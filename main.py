import logging
import random
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

TOKEN = "7927045864:AAEuaDleP5MlCUHrE30Mxf5YWegY4EUWUXE"  # Ganti dengan token yang benar
MODEL_NAME = "microsoft/DialoGPT-medium"
DATA_FILE = "chatbot_memory.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🔄 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"✅ Model loaded! Running on: {device}")

if os.path.exists(DATA_FILE):
    try:
        with open(DATA_FILE, "r") as file:
            memory = json.load(file)
    except json.JSONDecodeError:
        memory = {}
else:
    memory = {}

chat_history = [] 

def get_response(user_input):
    """Menghasilkan respons menggunakan model transformer."""
    global chat_history
    new_input = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)

    chat_history.append(new_input)
    chat_history = chat_history[-5:]

    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt", padding=True).to(device)
    attention_mask = inputs.ne(tokenizer.pad_token_id).to(device)  

    output = model.generate(
        inputs,
        attention_mask=attention_mask,  
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        top_p=0.92,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id  
    )

    response = tokenizer.decode(output[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    # Simpan history response
    chat_history.append(output.detach().cpu())  
    return response

sad_words = ["capek", "lelah", "sedih", "gagal", "sendirian", "stress", "galau", "down", "kesepian", "pusing"]
motivations = [
    "Tetap semangat, ya! Aku yakin kamu bisa melewatinya. 💪",
    "Jangan nyerah! Kadang hari ini berat, tapi besok bisa lebih baik. 😊",
    "Kamu nggak sendirian. Aku di sini buat dengerin. 💙",
    "Kalau butuh istirahat, nggak apa-apa kok. Kamu juga butuh waktu buat diri sendiri. ✨",
    "Ingat, setiap masalah pasti ada jalan keluarnya. Kamu lebih kuat dari yang kamu kira!"
]

def is_sensitive_topic(text):
    """Mendeteksi apakah input mengandung topik sensitif."""
    sensitive_words = ["18+", "thriller"]
    return any(word in text.lower() for word in sensitive_words)

def get_transformers_response(user_input):
    """Menghasilkan respons dari model transformer dengan pengamanan tambahan."""
    try:
        user_input = user_input[:100]  
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)

        output = model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            top_p=0.92,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(output[:, inputs.shape[-1]:][0], skip_special_tokens=True).strip()

        if response.lower() == user_input.lower() or len(response) < 5:
            return "Aku belum punya jawaban yang pas buat itu. Ceritain lebih lanjut dong!"
        return response

    except Exception as e:
        logger.error(f"⚠️ ERROR: {e}")
        return "Maaf, aku lagi error nih. Coba lagi nanti ya!"

async def handle_message(update: Update, context: CallbackContext) -> None:
    """Menangani pesan dari user di Telegram."""
    user_input = update.message.text.lower()
    chat_id = update.message.chat_id

    if is_sensitive_topic(user_input):
        response = "Maaf, aku nggak bisa membahas topik itu. Tapi aku bisa bantu ngobrolin hal lain! 😊"
    elif user_input in memory:
        response = memory[user_input]
    elif any(word in user_input for word in sad_words):
        response = random.choice(motivations)
    else:
        response = get_transformers_response(user_input)
        if response not in ["Aku belum punya jawaban yang pas buat itu. Ceritain lebih lanjut dong!", ""]:
            memory[user_input] = response
            try:
                with open(DATA_FILE, "w") as file:
                    json.dump(memory, file)
            except Exception as e:
                logger.error(f"Gagal menyimpan ke file: {e}")

    await update.message.reply_text(response)

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Hai! Aku di sini buat dengerin kamu. 😊 Ketik aja apa yang kamu rasain!")

def main() -> None:
    """Menjalankan bot Telegram."""
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Chatbot is running on Telegram!")
    app.run_polling()

if __name__ == "__main__":
    main()
