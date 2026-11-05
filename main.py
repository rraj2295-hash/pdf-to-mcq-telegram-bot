
import os
import json
import asyncio
import logging
import tempfile
import secrets
import hmac

from aiohttp import web
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from google import genai
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]

RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
PORT = int(os.environ.get("PORT", "10000"))

if not RENDER_URL:
    raise RuntimeError("RENDER_EXTERNAL_URL is missing")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"

telegram_app = (
    ApplicationBuilder()
    .token(BOT_TOKEN)
    .build()
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to Bihar CHO PDF Quiz Bot!\n\n"
        "Please send a text-based PDF file.\n"
        "I will create MCQ Quiz from your PDF."
    )


def generate_questions(pdf_text):
    prompt = f"""
You are an expert nursing examination question creator.

Create up to 30 MCQs from the supplied PDF.

Rules:
1. Use only information in the PDF.
2. Each question must have exactly four options.
3. Give exactly one correct answer.
4. Do not invent facts.
5. Include rationale only if explicitly explained in PDF.
6. Otherwise rationale must be an empty string.
7. Avoid duplicate questions.
8. Return fewer questions if the PDF has insufficient content.

Return only valid JSON in this format:

{{
  "questions": [
    {{
      "number": 1,
      "question": "Question text",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "answer": "B",
      "rationale": ""
    }}
  ]
}}

PDF TEXT:
{pdf_text[:60000]}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.2,
        },
    )

    data = json.loads(response.text)
    return data.get("questions", [])


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    document = message.document

    if not document.file_name or not document.file_name.lower().endswith(".pdf"):
        await message.reply_text("Please send a PDF file.")
        return

    if document.file_size and document.file_size > 20 * 1024 * 1024:
        await message.reply_text("PDF must be smaller than 20 MB.")
        return

    await message.reply_text("PDF received! Reading and creating MCQs...")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as temp:
            temp_path = temp.name

        pdf_file = await context.bot.get_file(document.file_id)
        await pdf_file.download_to_drive(temp_path)

        reader = PdfReader(temp_path)
        pdf_text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

        if len(pdf_text.strip()) < 100:
            await message.reply_text(
                "PDF text could not be read. "
                "Please send a text-based PDF, not a scanned PDF."
            )
            return

        questions = await asyncio.to_thread(
            generate_questions, pdf_text
        )

        sent = 0

        for q in questions:
            options = q.get("options", {})
            answer = str(q.get("answer", "")).upper()

            keys = ["A", "B", "C", "D"]

            if (
                not q.get("question")
                or not isinstance(options, dict)
                or any(not options.get(k) for k in keys)
                or answer not in keys
            ):
                continue

            explanation = str(q.get("rationale", ""))[:200]

            await context.bot.send_poll(
                chat_id=message.chat_id,
                question=str(q["question"])[:300],
                options=[str(options[k])[:100] for k in keys],
                type="quiz",
                correct_option_id=keys.index(answer),
                explanation=explanation or None,
                is_anonymous=False,
            )

            sent += 1
            await asyncio.sleep(1)

        await message.reply_text(
            f"Quiz completed!\nQuestions sent: {sent}"
        )

    except Exception:
        logger.exception("PDF processing failed")
        await message.reply_text(
            "PDF processing failed. Please try again or check the PDF."
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


async def webhook_handler(request):
    supplied_secret = request.headers.get(
        "X-Telegram-Bot-Api-Secret-Token", ""
    )

    if not hmac.compare_digest(supplied_secret, WEBHOOK_SECRET):
        raise web.HTTPForbidden()

    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
        return web.Response(text="OK")
    except Exception:
        logger.exception("Webhook update failed")
        return web.Response(status=500, text="Error")


async def health(request):
    return web.Response(text="Bihar CHO Quiz Bot is running!")


async def on_startup(app):
    await telegram_app.initialize()
    await telegram_app.start()

    webhook_url = RENDER_URL.rstrip("/") + "/webhook"

    await telegram_app.bot.set_webhook(
        url=webhook_url,
        secret_token=WEBHOOK_SECRET,
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )

    logger.info("Telegram webhook configured")


async def on_cleanup(app):
    await telegram_app.bot.delete_webhook()
    await telegram_app.stop()
    await telegram_app.shutdown()


web_app = web.Application()
web_app.router.add_get("/", health)
web_app.router.add_get("/health", health)
web_app.router.add_post("/webhook", webhook_handler)

web_app.on_startup.append(on_startup)
web_app.on_cleanup.append(on_cleanup)


if __name__ == "__main__":
    web.run_app(
        web_app,
        host="0.0.0.0",
        port=PORT,
    )
