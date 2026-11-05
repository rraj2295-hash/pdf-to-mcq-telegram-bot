
import os
import json
import asyncio
import logging
import tempfile
import hmac
import re

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

# ================ CONFIGURATION ================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]

RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
PORT = int(os.environ.get("PORT", "10000"))

if not RENDER_URL:
    raise RuntimeError("RENDER_EXTERNAL_URL is missing")

if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", WEBHOOK_SECRET):
    raise RuntimeError("Invalid WEBHOOK_SECRET format")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-3.8-flash"

telegram_app = (
    ApplicationBuilder()
    .token(BOT_TOKEN)
    .build()
)


# ================ START COMMAND ================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.effective_message.reply_text(
        "Welcome to Bihar CHO PDF Quiz Bot! 🎓\n\n"
        "📚 Send a text-based PDF file.\n\n"
        "I will read your PDF and create up to 30 "
        "interactive MCQ Quiz questions.\n\n"
        "Best of luck for Bihar CHO preparation! 🩺"
    )


# ================ HELP COMMAND ================

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.effective_message.reply_text(
        "📖 Bihar CHO Quiz Bot Help\n\n"
        "/start - Start the Bot\n"
        "/help - Show instructions\n\n"
        "Send a PDF to generate MCQ quizzes."
    )


# ================ GEMINI MCQ GENERATION ================

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
8. Return fewer questions if PDF has insufficient content.

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

    if not response.text:
        raise ValueError("Gemini returned empty response")

    data = json.loads(response.text)

    if not isinstance(data, dict):
        raise ValueError("Invalid JSON response")

    questions = data.get("questions", [])

    if not isinstance(questions, list):
        raise ValueError("Invalid questions list")

    return questions


# ================ PDF HANDLER ================

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):

    message = update.effective_message
    document = message.document

    if not document or not document.file_name or not document.file_name.lower().endswith(".pdf"):

        await message.reply_text("Please send a PDF file.")

        return

    if document.file_size and document.file_size > 20 * 1024 * 1024:

        await message.reply_text("PDF must be smaller than 20 MB.")

        return

    await message.reply_text(
        "📥 PDF received!\n"
        "⏳ Reading PDF and creating MCQs. Please wait..."
    )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp:

            temp_path = temp.name

        pdf_file = await context.bot.get_file(document.file_id)

        await pdf_file.download_to_drive(temp_path)

        reader = PdfReader(temp_path)

        pdf_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        if len(pdf_text.strip()) < 100:

            await message.reply_text(
                "❌ PDF text could not be read.\n"
                "Please send a text-based PDF, not a scanned PDF."
            )

            return

        questions = await asyncio.to_thread(
            generate_questions,
            pdf_text
        )

        if not questions:

            await message.reply_text(
                "No MCQs could be generated from this PDF."
            )

            return

        sent = 0
        keys = ["A", "B", "C", "D"]

        for q in questions:

            if not isinstance(q, dict):
                continue

            options = q.get("options", {})

            if not isinstance(options, dict):
                continue

            question = str(q.get("question", "")).strip()
            answer = str(q.get("answer", "")).strip().upper()

            if (
                not question
                or len(question) > 300
                or answer not in keys
                or any(
                    not isinstance(options.get(k), str)
                    or not options[k].strip()
                    or len(options[k]) > 100
                    for k in keys
                )
            ):
                continue

            explanation = str(
                q.get("rationale", "")
            ).strip()[:200]

            await context.bot.send_poll(
                chat_id=message.chat_id,
                question=question,
                options=[options[k].strip() for k in keys],
                type="quiz",
                correct_option_id=keys.index(answer),
                explanation=explanation or None,
                is_anonymous=False,
            )

            sent += 1

            await asyncio.sleep(1)

        await message.reply_text(
            "🎉 Quiz completed!\n\n"
            f"Total Questions Sent: {sent}\n\n"
            "Best of luck for Bihar CHO preparation! 📚"
        )

    except Exception:

        logger.exception("PDF processing failed")

        await message.reply_text(
            "❌ PDF processing failed.\n"
            "Please check your PDF and API settings."
        )

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)


# ================ REGISTER HANDLERS ================

telegram_app.add_handler(
    CommandHandler("start", start)
)

telegram_app.add_handler(
    CommandHandler("help", help_command)
)

telegram_app.add_handler(
    MessageHandler(filters.Document.PDF, handle_pdf)
)


# ================ WEBHOOK HANDLER ================

async def webhook_handler(request):

    supplied_secret = request.headers.get(
        "X-Telegram-Bot-Api-Secret-Token",
        ""
    )

    if not hmac.compare_digest(
        supplied_secret,
        WEBHOOK_SECRET
    ):

        raise web.HTTPForbidden(
            text="Invalid webhook secret"
        )

    try:

        data = await request.json()

        update = Update.de_json(
            data,
            telegram_app.bot
        )

        await telegram_app.update_queue.put(update)

        return web.Response(text="OK")

    except Exception:

        logger.exception("Webhook update failed")

        return web.Response(
            status=500,
            text="Webhook processing error"
        )


# ================ HEALTH CHECK ================

async def health(request):

    return web.json_response({
        "status": "running",
        "service": "Bihar CHO Quiz Bot"
    })


# ================ STARTUP ================

async def on_startup(app):

    await telegram_app.initialize()

    await telegram_app.start()

    webhook_url = (
        RENDER_URL.rstrip("/") + "/webhook"
    )

    await telegram_app.bot.set_webhook(
        url=webhook_url,
        secret_token=WEBHOOK_SECRET,
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=False,
    )

    logger.info(
        "Telegram webhook configured successfully"
    )


# ================ CLEANUP ================

async def on_cleanup(app):

    # Never delete webhook during Render restart.

    await telegram_app.stop()

    await telegram_app.shutdown()

    logger.info("Telegram Bot stopped safely")


# ================ AIOHTTP SERVER ================

web_app = web.Application()

web_app.router.add_get("/", health)
web_app.router.add_get("/health", health)
web_app.router.add_post("/webhook", webhook_handler)

web_app.on_startup.append(on_startup)
web_app.on_cleanup.append(on_cleanup)


# ================ RUN SERVER ================

if __name__ == "__main__":

    web.run_app(
        web_app,
        host="0.0.0.0",
        port=PORT,
    )
