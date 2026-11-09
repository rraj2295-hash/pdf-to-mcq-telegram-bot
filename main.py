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
    PollAnswerHandler,
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

raw_channel = os.environ.get("TARGET_CHANNEL_ID", "")
TARGET_CHANNEL_ID = raw_channel.strip() if raw_channel else None

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

# Global trackers for live session
ACTIVE_POLLS = {}
USER_SCORES = {}


# ================ AUTO-DISCOVERY MODEL RESOLVER ================

def get_available_models():
    discovered = []
    try:
        for m in client.models.list():
            m_name = getattr(m, "name", str(m))
            clean_name = m_name.replace("models/", "")
            discovered.append(clean_name)
    except Exception as exc:
        logger.warning("Auto model discovery failed: %s", exc)

    flash_models = [m for m in discovered if "flash" in m.lower() and "8b" not in m.lower()]
    other_gemini = [m for m in discovered if "gemini" in m.lower() and m not in flash_models]
    
    final_candidates = flash_models + other_gemini + discovered
    fallbacks = ["gemini-3.0-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for fb in fallbacks:
        if fb not in final_candidates:
            final_candidates.append(fb)

    return final_candidates


# ================ START & HELP ================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Welcome to Bihar CHO PDF Quiz Bot! 🎓\n\n"
        "📚 Send a text-based PDF file.\n"
        "PDF bhejne ke baad timer select karein.\n\n"
        "Bot automated numbering, topic banner aur leaderboard create karega! 🩺"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "📖 Bihar CHO Quiz Bot Help\n\n"
        "/start - Start the Bot\n"
        "/help - Instructions\n"
        "Send a PDF to generate automated quizzes with leaderboards."
    )


# ================ GEMINI MCQ & TOPIC GENERATION ================

def generate_quiz_payload(pdf_text):
    prompt = f"""
You are an expert nursing examination question creator.

1. Identify a suitable short topic title for this nursing study material (max 6 words).
2. Create up to 30 MCQs from the supplied PDF text.

Rules:
- Use only facts from the PDF.
- Each question must have exactly four options (A, B, C, D).
- Specify the correct answer clearly.
- Add concise rationale if available.

Return valid JSON in this exact structure:
{{
  "topic": "Topic Title Here",
  "questions": [
    {{
      "number": 1,
      "question": "Question text without numbering prefix",
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
    last_error = None

    for model_name in candidate_models:
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
                raise ValueError("Empty response from model")

            data = json.loads(response.text)
            if not isinstance(data, dict):
                raise ValueError("Invalid JSON")

            questions = data.get("questions", [])
            topic = data.get("topic", "Bihar CHO Nursing Mock Test")

            if isinstance(questions, list) and len(questions) > 0:
                return topic, questions[:30]

        except Exception as exc:
            last_error = exc
            err_msg = str(exc).upper()
            if "404" in err_msg or "NOT_FOUND" in err_msg:
                continue

            time.sleep(2)
            continue

    raise last_error if last_error else RuntimeError("Gemini generation failed")


# ================ POLL ANSWER TRACKER (LEADERBOARD) ================

async def handle_poll_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = update.poll_answer
    poll_id = answer.poll_id
    user = answer.user

    if poll_id not in ACTIVE_POLLS:
        return

    poll_info = ACTIVE_POLLS[poll_id]
    correct_id = poll_info["correct_option_id"]
    start_time = poll_info["timestamp"]
    elapsed = max(1.0, round(time.time() - start_time, 1))

    user_id = user.id
    user_name = user.full_name or user.username or f"Student_{user_id}"

    if user_id not in USER_SCORES:
        USER_SCORES[user_id] = {
            "name": user_name,
            "correct": 0,
            "attempted": 0,
            "total_time": 0.0,
        }

    USER_SCORES[user_id]["attempted"] += 1
    USER_SCORES[user_id]["total_time"] += elapsed

    if answer.option_ids and answer.option_ids[0] == correct_id:
        USER_SCORES[user_id]["correct"] += 1


# ================ PDF HANDLER ================

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    document = message.document

    if not document or not document.file_name or not document.file_name.lower().endswith(".pdf"):
        await message.reply_text("Please send a valid PDF file.")
        return

    if document.file_size and document.file_size > 20 * 1024 * 1024:
        await message.reply_text("PDF must be smaller than 20 MB.")
        return

    status_msg = await message.reply_text(
        "📥 PDF received!\n"
        "⏳ Reading PDF, determining Topic and creating MCQs...\n"
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

        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if len(pdf_text.strip()) < 100:
            await status_msg.edit_text("❌ Text could not be read. Please send a text-based PDF.")
            return

        topic, questions = await asyncio.to_thread(generate_quiz_payload, pdf_text)

        if not questions:
            await status_msg.edit_text("No MCQs could be generated.")
            return

        context.user_data["pending_questions"] = questions
        context.user_data["pending_topic"] = topic
        context.user_data["reply_chat_id"] = message.chat_id

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
            f"📖 **Topic**: {topic}\n"
            f"📊 **Total Questions**: {len(questions)}\n\n"
            "⏱️ Har question ke liye **Timer kitne seconds** ka rakhna hai? Select karein:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    except Exception as exc:
        logger.exception("PDF processing failed")
        await status_msg.edit_text(f"❌ PDF processing failed: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ================ TIMER CALLBACK & DISPATCHER ================

async def timer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global ACTIVE_POLLS, USER_SCORES
    query = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith("timer_"):
        return

    seconds = int(data.split("_")[1])
    questions = context.user_data.get("pending_questions", [])
    topic = context.user_data.get("pending_topic", "Bihar CHO Nursing Mock Test")
    reply_chat_id = context.user_data.get("reply_chat_id", query.message.chat_id)

    if not questions:
        await query.edit_message_text("❌ Questions expire ho chuke hain. Kripya PDF dobara bhejein.")
        return

    target_destination = TARGET_CHANNEL_ID if TARGET_CHANNEL_ID else reply_chat_id
    total_q = len(questions)

    # Reset scoreboards for this test session
    ACTIVE_POLLS = {}
    USER_SCORES = {}

    await query.edit_message_text(
        f"⏳ Timer: {seconds}s per question.\n"
        f"Target: {target_destination}\n"
        "Sending announcement and initiating Quiz..."
    )

    # 1. Starting Announcement Message
    intro_banner = (
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🎯 **BIHAR CHO DAILY QUIZ TEST** 🩺\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📖 **Topic:** {topic}\n"
        f"❓ **Total Questions:** {total_q}\n"
        f"⏱️ **Time per Question:** {seconds} Seconds\n\n"
        "👉 *Har question ka countdown timer chalu rahega. Best of luck!*"
    )

    try:
        await context.bot.send_message(
            chat_id=target_destination,
            text=intro_banner,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.warning("Intro banner send failed: %s", e)

    await asyncio.sleep(2)

    sent = 0
    fail_reason = ""
    keys = ["A", "B", "C", "D"]

    # Channel chats strictly require is_anonymous=True, groups allow interactive false
    is_channel = str(target_destination).startswith("-100") or str(target_destination).startswith("@")
    is_anonymous_mode = True if is_channel else False

    for idx, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            continue

        options = q.get("options", {})
        if not isinstance(options, dict):
            continue

        clean_q = str(q.get("question", "")).strip()
        answer = str(q.get("answer", "")).strip().upper()

        if (
            not clean_q
            or len(clean_q) > 280
            or answer not in keys
            or any(not isinstance(options.get(k), str) or not options[k].strip() or len(options[k]) > 100 for k in keys)
        ):
            continue

        # Add question numbering prefix: Q1., Q2., etc.
        formatted_question = f"Q{idx}. {clean_q}"
        explanation = str(q.get("rationale", "")).strip()[:200]
        correct_opt_id = keys.index(answer)

        try:
            poll_msg = await context.bot.send_poll(
                chat_id=target_destination,
                question=formatted_question,
                options=[options[k].strip() for k in keys],
                type="quiz",
                correct_option_id=correct_opt_id,
                explanation=explanation or None,
                is_anonymous=is_anonymous_mode,
                open_period=seconds,
            )

            # Store poll for real-time answer tracking
            if poll_msg.poll:
                ACTIVE_POLLS[poll_msg.poll.id] = {
                    "correct_option_id": correct_opt_id,
                    "timestamp": time.time(),
                }

            sent += 1
            await asyncio.sleep(seconds + 2)

        except Exception as poll_err:
            fail_reason = str(poll_err)
            logger.exception("Poll failed: %s", poll_err)
            break

    context.user_data.pop("pending_questions", None)
    context.user_data.pop("pending_topic", None)

    # 2. Leaderboard Generation
    leaderboard_text = ""
    if USER_SCORES:
        # Rank by most correct answers, then least time taken
        sorted_users = sorted(
            USER_SCORES.values(),
            key=lambda u: (-u["correct"], u["total_time"])
        )

        ranks_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
        rows = []
        for i, u in enumerate(sorted_users[:10]):
            badge = ranks_emojis[i] if i < len(ranks_emojis) else f"#{i+1}"
            rows.append(
                f"{badge} **{u['name']}**\n"
                f"   └ Score: `{u['correct']}/{u['attempted']}` | Time: `{u['total_time']:.1f}s`"
            )

        leaderboard_text = (
            "\n\n🏆 **LEADERBOARD / TOPPERS** 🏆\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            + "\n".join(rows) +
            "\n━━━━━━━━━━━━━━━━━━━━\n"
            "👏 Shandar performance! Keep it up!"
        )

    closing_message = (
        "🏁 **QUIZ CONCLUDED!**\n\n"
        f"📖 **Topic:** {topic}\n"
        f"📊 **Total Questions Completed:** {sent}/{total_q}"
        + leaderboard_text
    )

    try:
        await context.bot.send_message(
            chat_id=target_destination,
            text=closing_message,
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.warning("Could not send closing leaderboard: %s", e)

    # Summary report to creator bot chat
    await context.bot.send_message(
        chat_id=reply_chat_id,
        text=(
            "🎉 Quiz processing completed!\n\n"
            f"Topic: {topic}\n"
            f"Total Questions Sent to {target_destination}: {sent}\n"
            f"Timer per Question: {seconds}s\n"
            f"{'Reason: ' + fail_reason if fail_reason else ''}"
        )
    )


# ================ REGISTER HANDLERS ================

telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(CommandHandler("help", help_command))
telegram_app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
telegram_app.add_handler(CallbackQueryHandler(timer_callback))
telegram_app.add_handler(PollAnswerHandler(handle_poll_answer))


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
