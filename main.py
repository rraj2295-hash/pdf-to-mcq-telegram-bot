
import os
import json
import asyncio
import logging
import tempfile

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

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-flash-latest"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to Bihar CHO PDF Quiz Bot!\n\n"
        "Apni PDF file bhejiye. Main usse MCQ Quiz banaunga."
    )


def generate_questions(pdf_text):
    prompt = f"""
You are an expert nursing examination question creator.

Read the following PDF text and create 30 MCQs.

Rules:
1. Questions must be based only on the supplied PDF.
2. Each question must have exactly 4 options.
3. Give one correct answer.
4. Do not invent facts.
5. Include explanation only if the PDF explicitly provides it.
6. If no explanation is given, use an empty string.
7. Avoid duplicate questions.

Return ONLY valid JSON in this format:

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
            "temperature": 0.2
        }
    )

    data = json.loads(response.text)
    return data.get("questions", [])


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message

    if not message.document.file_name.lower().endswith(".pdf"):
        await message.reply_text("Please send a PDF file.")
        return

    await message.reply_text("PDF received! Processing... Please wait.")

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp:
            pdf_file = await context.bot.get_file(message.document.file_id)
            await pdf_file.download_to_drive(temp.name)

            reader = PdfReader(temp.name)
            pdf_text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

        if len(pdf_text.strip()) < 100:
            await message.reply_text(
                "PDF ka text read nahi ho raha. "
                "Please send a text-based PDF, not a scanned image PDF."
            )
            return

        questions = await asyncio.to_thread(
            generate_questions, pdf_text
        )

        if not questions:
            await message.reply_text("No MCQs generated. Try another PDF.")
            return

        sent = 0

        for q in questions:
            options = q.get("options", {})
            answer = str(q.get("answer", "")).upper()

            keys = ["A", "B", "C", "D"]

            if (
                not q.get("question")
                or any(not options.get(k) for k in keys)
                or answer not in keys
            ):
                continue

            explanation = str(q.get("rationale", ""))[:180]

            await context.bot.send_poll(
                chat_id=message.chat_id,
                question=str(q["question"])[:300],
                options=[str(options[k])[:100] for k in keys],
                type="quiz",
                correct_option_id=keys.index(answer),
                explanation=explanation or None,
                is_anonymous=False
            )

            sent += 1
            await asyncio.sleep(1)

        await message.reply_text(
            f"Quiz completed!\nTotal questions sent: {sent}"
        )

    except Exception as e:
        logger.exception("PDF processing failed")
        await message.reply_text(
            f"Error processing PDF: {str(e)[:300]}"
        )


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(filters.Document.PDF, handle_pdf)
    )

    print("Bihar CHO Quiz Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
