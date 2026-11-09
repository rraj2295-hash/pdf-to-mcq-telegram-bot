import os
import json
import asyncio
import logging
import tempfile
import hmac
import re
import time

from aiohttp import web
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
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

# Target Channel / Group (e.g. @CHO26_27 ya -100xxxxxxxxxx)
TARGET_CHANNEL_ID = os.environ.get("TARGET_CHANNEL_ID")

RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
PORT = int(os.environ.get("PORT", "10000"))

if not RENDER_URL:
    raise RuntimeError("RENDER_EXTERNAL_URL is missing")

if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", WEBHOOK_SECRET):
    raise RuntimeError("Invalid WEBHOOK_SECRET format")

client = genai.Client(api_key=GEMINI_API_KEY)

telegram_app = (
    ApplicationBuilder()
    .token(BOT_TOKEN)
    .build()
)


# ================ AUTO-DISCOVERY MODEL RESOLVER ================

def get_available_models():
    """Queries Google API to find exactly which models are supported on this key."""
    discovered = []
    try:
        for m in client.models.list():
            # Extract clean model name
            m_name = getattr(m, "name", str(m))
            clean_name = m_name.replace("models/", "")
            
            # Check supported actions if metadata is available
            methods = getattr(m, "supported_generation_methods", None) or getattr(m, "supported_actions", None)
            if methods:
                if any("generateContent" in str(action) for action in methods):
                    discovered.append(clean_name)
            else:
                discovered.append(clean_name)
    except Exception as exc:
        logger.warning("Auto model discovery via list() failed: %s", exc)

    # Sort discovered to prioritize flash versions
    flash_models = [m for m in discovered if "flash" in m.lower() and "8b" not in m.lower()]
    other_gemini = [m for m in discovered if "gemini" in m.lower() and m not in flash_models]
    
    final_candidates = flash_models + other_gemini + discovered
    
    # Static fallbacks just in case list API is restricted
    fallbacks = ["gemini-3.0-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for fb in fallbacks:
        if fb not in final_candidates:
            final_candidates.append(fb)

    return final_candidates


# ================ START COMMAND ================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Welcome to Bihar CHO PDF Quiz Bot! 🎓\n\n"
        "📚 Send a text-based PDF file.\n\n"
        "PDF bhejne ke baad bot timer select karne bolega.\n\n"
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

    candidate_models = get_available_models()
    logger.info("Candidate models to attempt: %s", candidate_models)

    last_error = None

    for model_name in candidate_models:
        logger.info("Testing candidate model: %s", model_name)
        try:
            response = client.models.generate_content(
                model=model_name,
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
                raise ValueError("Invalid JSON format received")

            questions = data.get("questions", [])
            if not isinstance(questions, list):
                raise ValueError("Invalid questions list")

            logger.info("Successfully generated %s questions with model: %s", len(questions), model_name)
            return questions[:30]

        except Exception as exc:
            last_error = exc
            err_msg = str(exc).upper()
            logger.warning("Model candidate %s failed: %s", model_name, exc)

            # If 404 or NOT_FOUND, immediately try next candidate model
            if "404" in err_msg or "NOT_FOUND" in err_msg:
                continue

            # If temporary rate limit / server error, retry once after short delay
            if any(code in err_msg for code in ("429", "500", "503", "504", "RESOURCE_EXHAUSTED")):
                time.sleep(3)
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            "temperature": 0.2,
                        },
                    )
                    if response.text:
                        data = json.loads(response.text)
                        questions = data.get("questions", [])
                        if isinstance(questions, list):
                            return questions[:30]
                except Exception as retry_exc:
                    last_error = retry_exc
                    continue

    logger.exception("All model candidates exhausted")
    raise last_error if last_error else RuntimeError("Gemini MCQ generation failed on all models")


# ================ PDF HANDLER ================

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    document = message.document

    if (
        not document
        or not document.file_name
        or not document.file_name.lower().endswith(".pdf")
    ):
        await message.reply_text("Please send a PDF file.")
        return

    if document.file_size and document.file_size > 20 * 1024 * 1024:
        await message.reply_text("PDF must be smaller than 20 MB.")
        return

    status_msg = await message.reply_text(
        "📥 PDF received!\n"
        "⏳ Reading PDF and creating MCQs with Gemini AI...\n"
        "Please wait..."
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
            temp_path = temp.name

        pdf_file = await context.bot.get_file(document.file_id)
        await pdf_file.download_to_drive(temp_path)

        reader = PdfReader(temp_path)

        if reader.is_encrypted:
            raise ValueError("Encrypted PDF is not supported")

        pdf_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        if len(pdf_text.strip()) < 100:
            await status_msg.edit_text(
                "❌ PDF text could not be read.\n"
                "Please send a text-based PDF, not a scanned PDF."
            )
            return

        questions = await asyncio.to_thread(
            generate_questions,
            pdf_text
        )

        if not questions:
            await status_msg.edit_text(
                "No MCQs could be generated from this PDF."
            )
            return

        context.user_data["pending_questions"] = questions
        context.user_data["reply_chat_id"] = message.chat_id

        # Timer selection buttons
        keyboard = [
            [
                InlineKeyboardButton("⏱️ 15s", callback_data="timer_15"),
                InlineKeyboardButton("⏱️ 30s", callback_data="timer_30"),
            ],
            [
                InlineKeyboardButton("⏱️ 45s", callback_data="timer_45"),
                InlineKeyboardButton("⏱️ 60s", callback_data="timer_60"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await status_msg.edit_text(
            f"✅ {len(questions)} Questions generate ho gaye hain!\n\n"
            "⏱️ Har question ke liye **Timer kitne seconds** ka rakhna hai? Select karein:",
            reply_markup=reply_markup
        )

    except Exception as exc:
        logger.exception("PDF processing failed")
        error_text = str(exc).upper()

        if any(
            code in error_text
            for code in ("503", "429", "500", "502", "504",
                         "UNAVAILABLE", "RESOURCE_EXHAUSTED")
        ):
            user_message = (
                "⚠️ Gemini AI is temporarily busy.\n\n"
                "Please try sending your PDF again after a few minutes."
            )
        elif "API KEY" in error_text or "API_KEY" in error_text:
            user_message = (
                "❌ Gemini API Key problem.\n"
                "Please check GEMINI_API_KEY in Render Environment."
            )
        else:
            user_message = (
                "❌ PDF processing failed.\n\n"
                "Please check the PDF and try again."
            )

        await status_msg.edit_text(user_message)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ================ TIMER BUTTON CALLBACK HANDLER ================

async def timer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith("timer_"):
        return

    seconds = int(data.split("_")[1])
    questions = context.user_data.get("pending_questions", [])
    reply_chat_id = context.user_data.get("reply_chat_id", query.message.chat_id)

    if not questions:
        await query.edit_message_text("❌ Questions expire ho chuke hain. Kripya PDF dobara bhejein.")
        return

    destination_chat_id = TARGET_CHANNEL_ID if TARGET_CHANNEL_ID else reply_chat_id
    target_name = TARGET_CHANNEL_ID if TARGET_CHANNEL_ID else "Chat"

    await query.edit_message_text(
        f"⏳ Timer set to {seconds} seconds.\n"
        f"Channel ({target_name}) me questions bhejna start ho raha hai..."
    )

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

        explanation = str(q.get("rationale", "")).strip()[:200]

        try:
            await context.bot.send_poll(
                chat_id=destination_chat_id,
                question=question,
                options=[options[k].strip() for k in keys],
                type="quiz",
                correct_option_id=keys.index(answer),
                explanation=explanation or None,
                is_anonymous=False,
                open_period=seconds,
            )
            sent += 1

            await asyncio.sleep(seconds + 2)

        except Exception:
            logger.exception("Failed to send an individual quiz question")

    context.user_data.pop("pending_questions", None)

    await context.bot.send_message(
        chat_id=reply_chat_id,
        text=(
            "🎉 Quiz processing completed!\n\n"
            f"Total Questions Sent to {target_name}: {sent}\n"
            f"Timer per Question: {seconds}s\n\n"
            "Best of luck for Bihar CHO preparation! 📚"
        )
    )


# ================ REGISTER HANDLERS ================

telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(CommandHandler("help", help_command))
telegram_app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
telegram_app.add_handler(CallbackQueryHandler(timer_callback))


# ================ WEBHOOK HANDLER ================

async def webhook_handler(request):
    supplied_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")

    if not hmac.compare_digest(supplied_secret, WEBHOOK_SECRET):
        raise web.HTTPForbidden(text="Invalid webhook secret")

    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.update_queue.put(update)
        return web.Response(text="OK")

    except Exception:
        logger.exception("Webhook update failed")
        return web.Response(status=500, text="Webhook processing error")


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

    webhook_url = RENDER_URL.rstrip("/") + "/webhook"

    await telegram_app.bot.set_webhook(
        url=webhook_url,
        secret_token=WEBHOOK_SECRET,
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=False,
    )
    logger.info("Telegram webhook configured successfully")


# ================ CLEANUP ================

async def on_cleanup(app):
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
