import os
import json
import asyncio
import logging
import tempfile
import hmac
import re
import time
import uuid
import certifi
from datetime import datetime

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
from motor.motor_asyncio import AsyncIOMotorClient

# ================ CONFIGURATION & LOGGING ================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BiharCHOBot")

BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGODB_DB_NAME", "bihar_cho_quiz_db")

raw_channel = os.environ.get("TARGET_CHANNEL_ID", "")
TARGET_CHANNEL_ID = raw_channel.strip() if raw_channel else None

RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
PORT = int(os.environ.get("PORT", "10000"))

if not RENDER_URL:
    raise RuntimeError("RENDER_EXTERNAL_URL environment variable is missing")

if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", WEBHOOK_SECRET):
    raise RuntimeError("Invalid WEBHOOK_SECRET format")

# Database & External Clients (SSL/TLS Fix with certifi)
if "mongodb+srv://" in MONGODB_URI or "ssl=true" in MONGODB_URI.lower():
    mongo_client = AsyncIOMotorClient(
        MONGODB_URI,
        tlsCAFile=certifi.where(),
        tls=True,
        serverSelectionTimeoutMS=5000
    )
else:
    mongo_client = AsyncIOMotorClient(MONGODB_URI)

db = mongo_client[DB_NAME]
client = genai.Client(api_key=GEMINI_API_KEY)

telegram_app = (
    ApplicationBuilder()
    .token(BOT_TOKEN)
    .build()
)

# Active background session tasks (session_id -> asyncio.Task)
ACTIVE_TASKS = {}


# ================ MODEL RESOLVER ================

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
    fallbacks = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for fb in fallbacks:
        if fb not in final_candidates:
            final_candidates.append(fb)

    return final_candidates


# ================ START & HELP ================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "🏥 **Welcome to Bihar CHO Nursing Mock Bot!** 🎓\n\n"
        "📚 Send a text-based PDF (Study material / Syllabus notes).\n"
        "Questions are automatically extracted, stored in Database, and scheduled.\n\n"
        "**Features:**\n"
        "• MongoDB Persistent Storage (Data safe on bot restarts)\n"
        "• Accurate Candidate-wise Toppers Leaderboard (Right, Wrong, Speed)\n"
        "• In-Chat Realtime Controls: Pause ⏸️ / Resume ▶️ / Restart 🔄\n"
        "• Multi-Session Support across study groups",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "📖 **Bihar CHO Quiz Bot Guide**\n\n"
        "1. Send your nursing notes in PDF format.\n"
        "2. Select the poll answer countdown timer (15s - 60s).\n"
        "3. Live quiz will launch in the designated group/channel.\n"
        "4. Use the control panel in private DM to Pause or Restart at any point.",
        parse_mode="Markdown"
    )


# ================ GEMINI MCQ EXTRACTION ================

def generate_quiz_payload(pdf_text):
    prompt = f"""
You are an expert nursing examination question creator for State CHO (Community Health Officer) exams.

1. Identify a concise topic title for this material (max 6 words).
2. Create up to 30 high-yield nursing MCQs based strictly on the provided text.

Rules:
- Each question must have exactly four options (A, B, C, D).
- Specify the correct answer option clearly.
- Provide a brief rationale (maximum 200 characters) for why the answer is correct.

Return strictly valid JSON matching this schema:
{{
  "topic": "Topic Title Here",
  "questions": [
    {{
      "number": 1,
      "question": "Question text here",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "answer": "B",
      "rationale": "Rationale explanation here"
    }}
  ]
}}

PDF CONTENT:
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
                raise ValueError("Empty response from AI")

            data = json.loads(response.text)
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")

            questions = data.get("questions", [])
            topic = data.get("topic", "Bihar CHO Nursing Mock Test")

            if isinstance(questions, list) and len(questions) > 0:
                return topic, questions[:30]

        except Exception as exc:
            last_error = exc
            if "404" in str(exc) or "NOT_FOUND" in str(exc):
                continue
            time.sleep(2)
            continue

    raise last_error if last_error else RuntimeError("Gemini AI extraction failed")


# ================ POLL ANSWER TRACKER (MONGODB) ================

async def handle_poll_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = update.poll_answer
    poll_id = answer.poll_id
    user = answer.user

    if not answer.option_ids:
        return

    poll_doc = await db.active_polls.find_one({"poll_id": str(poll_id)})
    if not poll_doc:
        return

    session_id = poll_doc["session_id"]
    correct_option_id = poll_doc["correct_option_id"]
    start_time = poll_doc["created_at"]
    elapsed = max(0.5, round(time.time() - start_time, 1))

    is_correct = (answer.option_ids[0] == correct_option_id)
    user_id = user.id
    user_name = user.full_name or user.first_name or f"Candidate_{user_id}"

    inc_query = {
        "attempted": 1,
        "total_time": elapsed,
        "correct": 1 if is_correct else 0,
        "wrong": 0 if is_correct else 1
    }

    await db.session_scores.update_one(
        {"session_id": session_id, "user_id": user_id},
        {
            "$set": {"user_name": user_name, "updated_at": datetime.utcnow()},
            "$inc": inc_query
        },
        upsert=True
    )


# ================ CONTROL KEYBOARD HELPER ================

def get_control_keyboard(session_id: str, is_paused: bool = False):
    toggle_btn = (
        InlineKeyboardButton("▶️ Resume Quiz", callback_data=f"ctrl_resume:{session_id}")
        if is_paused
        else InlineKeyboardButton("⏸️ Pause Quiz", callback_data=f"ctrl_pause:{session_id}")
    )
    restart_btn = InlineKeyboardButton("🔄 Restart Quiz", callback_data=f"ctrl_restart:{session_id}")
    return InlineKeyboardMarkup([[toggle_btn, restart_btn]])


# ================ PDF HANDLER ================

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    document = message.document

    if not document or not document.file_name or not document.file_name.lower().endswith(".pdf"):
        await message.reply_text("Please upload a valid text PDF document.")
        return

    if document.file_size and document.file_size > 20 * 1024 * 1024:
        await message.reply_text("File size exceeds 20 MB limit.")
        return

    status_msg = await message.reply_text(
        "📥 **PDF Received!**\n"
        "⏳ Reading document, structuring nursing MCQs, and caching to database...\n"
        "Please wait a moment...",
        parse_mode="Markdown"
    )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
            temp_path = temp.name

        pdf_file = await context.bot.get_file(document.file_id)
        await pdf_file.download_to_drive(temp_path)

        reader = PdfReader(temp_path)
        if reader.is_encrypted:
            raise ValueError("Password-protected PDF is not supported.")

        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if len(pdf_text.strip()) < 100:
            await status_msg.edit_text("❌ No readable text found. Please upload a digital/text-based PDF.")
            return

        topic, questions = await asyncio.to_thread(generate_quiz_payload, pdf_text)

        if not questions:
            await status_msg.edit_text("❌ Could not extract questions from this document.")
            return

        question_set_id = str(uuid.uuid4())[:8]
        await db.question_bank.insert_one({
            "set_id": question_set_id,
            "topic": topic,
            "questions": questions,
            "created_by": message.from_user.id,
            "created_at": datetime.utcnow()
        })

        context.user_data["pending_set_id"] = question_set_id
        context.user_data["pending_topic"] = topic
        context.user_data["reply_chat_id"] = message.chat_id

        keyboard = [
            [
                InlineKeyboardButton("⏱️ 15s", callback_data=f"timer_15:{question_set_id}"),
                InlineKeyboardButton("⏱️ 30s", callback_data=f"timer_30:{question_set_id}"),
            ],
            [
                InlineKeyboardButton("⏱️ 45s", callback_data=f"timer_45:{question_set_id}"),
                InlineKeyboardButton("⏱️ 60s", callback_data=f"timer_60:{question_set_id}"),
            ]
        ]

        await status_msg.edit_text(
            f"📖 **Topic**: `{topic}`\n"
            f"📊 **Total MCQs**: `{len(questions)}`\n"
            f"💾 **Saved to Bank ID**: `{question_set_id}`\n\n"
            "⏱️ Har question ke liye timer kitne seconds ka rakhna hai?",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    except Exception as exc:
        logger.exception("PDF processing failed")
        await status_msg.edit_text(f"❌ PDF processing failed: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ================ QUIZ RUNNER ENGINE ================

async def run_quiz_flow(bot, session_id: str):
    while True:
        session = await db.quiz_sessions.find_one({"session_id": session_id})
        if not session:
            break

        target_destination = session["target_destination"]
        reply_chat_id = session["reply_chat_id"]
        topic = session["topic"]
        questions = session["questions"]
        seconds = session["seconds"]
        total_q = len(questions)

        await db.active_polls.delete_many({"session_id": session_id})
        await db.session_scores.delete_many({"session_id": session_id})
        await db.quiz_sessions.update_one(
            {"session_id": session_id},
            {"$set": {"is_running": True, "is_paused": False, "should_restart": False}}
        )

        intro_banner = (
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🎯 **BIHAR CHO NURSING MOCK TEST** 🩺\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📖 **Topic:** {topic}\n"
            f"❓ **Total Questions:** {total_q}\n"
            f"⏱️ **Time per Question:** {seconds} Seconds\n\n"
            "👉 *Poll me sahi option chunein. Quiz ke baad Toppers Leaderboard declare hoga!*"
        )

        try:
            await bot.send_message(
                chat_id=target_destination,
                text=intro_banner,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.warning("Intro banner send failed: %s", e)

        await asyncio.sleep(2)

        sent = 0
        keys = ["A", "B", "C", "D"]
        restarted = False

        for idx, q in enumerate(questions, start=1):
            sess_state = await db.quiz_sessions.find_one({"session_id": session_id})
            if not sess_state or not sess_state.get("is_running", True):
                break

            if sess_state.get("should_restart", False):
                restarted = True
                await bot.send_message(
                    chat_id=target_destination,
                    text="🔄 **Quiz dobara shuru ho rahi hai... Question 1 par return kiya ja raha hai.**",
                    parse_mode="Markdown"
                )
                await asyncio.sleep(2)
                break

            while sess_state.get("is_paused", False):
                await asyncio.sleep(1)
                sess_state = await db.quiz_sessions.find_one({"session_id": session_id})
                if sess_state.get("should_restart", False):
                    break

            if sess_state.get("should_restart", False):
                restarted = True
                await bot.send_message(
                    chat_id=target_destination,
                    text="🔄 **Quiz dobara shuru ho rahi hai... Question 1 par return kiya ja raha hai.**",
                    parse_mode="Markdown"
                )
                await asyncio.sleep(2)
                break

            options = q.get("options", {})
            clean_q = str(q.get("question", "")).strip()
            answer = str(q.get("answer", "")).strip().upper()

            if (
                not clean_q
                or len(clean_q) > 280
                or answer not in keys
                or any(not isinstance(options.get(k), str) or not options[k].strip() or len(options[k]) > 100 for k in keys)
            ):
                continue

            formatted_question = f"Q{idx}. {clean_q}"
            explanation = str(q.get("rationale", "")).strip()[:200]
            correct_opt_id = keys.index(answer)

            try:
                poll_msg = await bot.send_poll(
                    chat_id=target_destination,
                    question=formatted_question,
                    options=[options[k].strip() for k in keys],
                    type="quiz",
                    correct_option_id=correct_opt_id,
                    explanation=explanation or None,
                    is_anonymous=False,
                    open_period=seconds,
                )

                if poll_msg.poll:
                    await db.active_polls.insert_one({
                        "poll_id": str(poll_msg.poll.id),
                        "session_id": session_id,
                        "correct_option_id": correct_opt_id,
                        "created_at": time.time()
                    })

                sent += 1

                control_msg_id = sess_state.get("control_message_id")
                if control_msg_id:
                    try:
                        await bot.edit_message_text(
                            chat_id=reply_chat_id,
                            message_id=control_msg_id,
                            text=(
                                f"🎮 **Live Quiz Control Panel**\n\n"
                                f"📖 Topic: `{topic}`\n"
                                f"📊 Progress: `Question {idx}/{total_q}`\n"
                                f"⏱️ Status: `{'PAUSED ⏸️' if sess_state.get('is_paused') else 'RUNNING ▶️'}`"
                            ),
                            reply_markup=get_control_keyboard(session_id, sess_state.get("is_paused", False)),
                            parse_mode="Markdown"
                        )
                    except Exception:
                        pass

                step_sleep = seconds + 2
                for _ in range(step_sleep):
                    curr = await db.quiz_sessions.find_one({"session_id": session_id})
                    if curr.get("should_restart", False):
                        break
                    while curr.get("is_paused", False):
                        await asyncio.sleep(1)
                        curr = await db.quiz_sessions.find_one({"session_id": session_id})
                        if curr.get("should_restart", False):
                            break
                    await asyncio.sleep(1)

            except Exception as poll_err:
                logger.exception("Poll dispatch failed: %s", poll_err)
                break

        if restarted:
            continue

        await asyncio.sleep(3)

        cursor = db.session_scores.find({"session_id": session_id}).sort([("correct", -1), ("total_time", 1)]).limit(10)
        sorted_users = await cursor.to_list(length=10)

        if sorted_users:
            ranks_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
            rows = []
            for i, u in enumerate(sorted_users):
                badge = ranks_emojis[i] if i < len(ranks_emojis) else f"#{i+1}"
                rows.append(
                    f"{badge} **{u['user_name']}**\n"
                    f"   ├ ✅ Sahi: `{u['correct']}` | ❌ Galat: `{u['wrong']}`\n"
                    f"   ├ 📝 Total Attempt: `{u['attempted']}/{total_q}`\n"
                    f"   └ ⏱️ Avg Speed / Total: `{u['total_time']:.1f}s`"
                )

            leaderboard_section = (
                "\n\n🏆 **TOPPERS LEADERBOARD** 🏆\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                + "\n\n".join(rows) +
                "\n━━━━━━━━━━━━━━━━━━━━\n"
                "👏 Bahut badhiya! Congratulations Candidates!"
            )
        else:
            leaderboard_section = (
                "\n\n🏆 **TOPPERS LEADERBOARD** 🏆\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Is test me kisi bhi student ne answer submit nahi kiya."
            )

        closing_message = (
            "🏁 **MOCK TEST CONCLUDED!**\n\n"
            f"📖 **Topic:** {topic}\n"
            f"📊 **Questions Completed:** {sent}/{total_q}"
            + leaderboard_section
        )

        try:
            await bot.send_message(
                chat_id=target_destination,
                text=closing_message,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.warning("Could not post closing leaderboard: %s", e)

        await db.quiz_sessions.update_one(
            {"session_id": session_id},
            {"$set": {"is_running": False, "concluded_at": datetime.utcnow()}}
        )

        final_sess = await db.quiz_sessions.find_one({"session_id": session_id})
        if final_sess and final_sess.get("control_message_id"):
            try:
                await bot.edit_message_text(
                    chat_id=reply_chat_id,
                    message_id=final_sess["control_message_id"],
                    text=(
                        "🎉 **Quiz Process Completed!**\n\n"
                        f"Topic: {topic}\n"
                        f"Questions Delivered: {sent}/{total_q}\n"
                        f"Target Chat: `{target_destination}`\n"
                        f"Leaderboard is successfully published."
                    ),
                    parse_mode="Markdown"
                )
            except Exception:
                pass

        break


# ================ CALLBACK HANDLER (TIMER & CONTROLS) ================

async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data

    if data.startswith("timer_"):
        parts = data.split(":")
        seconds = int(parts[0].replace("timer_", ""))
        question_set_id = parts[1]

        stored_set = await db.question_bank.find_one({"set_id": question_set_id})
        if not stored_set:
            await query.edit_message_text("❌ Question set expired or missing. Please upload the PDF again.")
            return

        questions = stored_set["questions"]
        topic = stored_set["topic"]
        reply_chat_id = query.message.chat_id
        target_destination = TARGET_CHANNEL_ID if TARGET_CHANNEL_ID else reply_chat_id

        session_id = str(uuid.uuid4())[:12]

        ctrl_msg = await query.edit_message_text(
            f"🎮 **Live Quiz Control Panel**\n\n"
            f"📖 Topic: `{topic}`\n"
            f"📊 Session ID: `{session_id}`\n"
            f"⏱️ Status: `RUNNING` ({seconds}s/question)\n"
            f"Target: `{target_destination}`",
            reply_markup=get_control_keyboard(session_id, is_paused=False),
            parse_mode="Markdown"
        )

        await db.quiz_sessions.insert_one({
            "session_id": session_id,
            "set_id": question_set_id,
            "topic": topic,
            "questions": questions,
            "seconds": seconds,
            "target_destination": target_destination,
            "reply_chat_id": reply_chat_id,
            "control_message_id": ctrl_msg.message_id,
            "is_running": True,
            "is_paused": False,
            "should_restart": False,
            "created_at": datetime.utcnow()
        })

        task = asyncio.create_task(run_quiz_flow(bot=context.bot, session_id=session_id))
        ACTIVE_TASKS[session_id] = task
        return

    if ":" in data and data.split(":")[0] in ("ctrl_pause", "ctrl_resume", "ctrl_restart"):
        action, session_id = data.split(":")
        session = await db.quiz_sessions.find_one({"session_id": session_id})
        if not session or not session.get("is_running"):
            await query.answer("This quiz session has already completed or does not exist.", show_alert=True)
            return

        if action == "ctrl_pause":
            await db.quiz_sessions.update_one(
                {"session_id": session_id},
                {"$set": {"is_paused": True}}
            )
            await query.edit_reply_markup(reply_markup=get_control_keyboard(session_id, is_paused=True))
            await query.answer("⏸️ Quiz paused!")

        elif action == "ctrl_resume":
            await db.quiz_sessions.update_one(
                {"session_id": session_id},
                {"$set": {"is_paused": False}}
            )
            await query.edit_reply_markup(reply_markup=get_control_keyboard(session_id, is_paused=False))
            await query.answer("▶️ Quiz resumed!")

        elif action == "ctrl_restart":
            await db.quiz_sessions.update_one(
                {"session_id": session_id},
                {"$set": {"should_restart": True, "is_paused": False}}
            )
            await query.answer("🔄 Restarting quiz from Question 1...", show_alert=True)


# ================ WEBHOOK & AIOHTTP HTTP SERVER ================

async def health_check(request: web.Request) -> web.Response:
    """Render port scan and uptime ping."""
    return web.Response(text="Bihar CHO Quiz Bot is active and running!", status=200)


async def telegram_webhook_handler(request: web.Request) -> web.Response:
    """Processes incoming Telegram updates via secure webhook."""
    received_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not hmac.compare_digest(received_token, WEBHOOK_SECRET):
        logger.warning("Unauthorized webhook request dropped (Invalid Secret Token).")
        return web.Response(status=403, text="Forbidden")

    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
        return web.Response(status=200, text="OK")
    except Exception as exc:
        logger.exception("Error processing webhook update: %s", exc)
        return web.Response(status=500, text="Internal Error")


# ================ APPLICATION LIFECYCLE ================

async def on_startup(app: web.Application):
    """Initializes Telegram Bot, handlers, and registers the webhook."""
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(MessageHandler(filters.Document.ALL, handle_pdf))
    telegram_app.add_handler(PollAnswerHandler(handle_poll_answer))
    telegram_app.add_handler(CallbackQueryHandler(callback_router))

    await telegram_app.initialize()
    await telegram_app.start()

    webhook_path = f"/webhook/{BOT_TOKEN}"
    full_webhook_url = f"{RENDER_URL.rstrip('/')}{webhook_path}"

    await telegram_app.bot.set_webhook(
        url=full_webhook_url,
        secret_token=WEBHOOK_SECRET,
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )
    logger.info("Webhook successfully registered to: %s", full_webhook_url)


async def on_cleanup(app: web.Application):
    """Graceful shutdown for background tasks, Telegram Bot, and Mongo connection."""
    logger.info("Shutting down bot application...")

    for session_id, task in list(ACTIVE_TASKS.items()):
        if not task.done():
            task.cancel()

    await telegram_app.bot.delete_webhook()
    await telegram_app.stop()
    await telegram_app.shutdown()
    mongo_client.close()
    logger.info("Cleanup completed successfully.")


def create_web_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", health_check)
    app.router.add_post(f"/webhook/{BOT_TOKEN}", telegram_webhook_handler)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


# ================ MAIN RUNNER ================

if __name__ == "__main__":
    web_app = create_web_app()
    web.run_app(web_app, host="0.0.0.0", port=PORT)
