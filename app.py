import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# Import API Key dari file rahasia
from config.secrets import TELEGRAM_BOT_TOKEN, DEEPSEEK_API_KEY

# === Load Models ===
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Helper Functions ===
def is_food_image(image):
    labels = ["a photo of food", "a photo of a random object", "a photo of a landscape", "a photo of an animal"]
    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return probs[0][0].item() > 0.9  # food score

def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def generate_recipe(caption):
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates easy-to-read recipes in plain text, without using markdown headings."},
                {"role": "user", "content": f"Generate a detailed and well-structured recipe for {caption}."},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Terjadi error saat membuat resep: {str(e)}"

# === Telegram Bot Handler ===
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Step 1: Ambil foto
        await update.message.reply_text("📥 Photo successfully received. Processing is underway...")

        file = await update.message.photo[-1].get_file()
        image_bytes = requests.get(file.file_path).content
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Step 2: Validasi gambar
        if not is_food_image(image):
            await update.message.reply_text("⚠️ This image is not detected as food.")
            return

        # Step 3: Caption oleh AI
        caption = generate_caption(image)
        await update.message.reply_text(f"📸 AI identified this image as:\n*{caption}*", parse_mode="Markdown")

        # Step 4: Notifikasi sedang proses
        await update.message.reply_text("🧠 Making a recipe... please wait a moment...")

        # Step 5: Generate resep
        recipe = generate_recipe(caption)

        # Step 6: Kirim resep
        await update.message.reply_text(f"📜 This recipe for*{caption}*:\n\n{recipe}", parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ There is an Error:\n{e}")

# === Start Bot ===
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("🤖 Bot sedang berjalan di PythonAnywhere...")
    app.run_polling()

if __name__ == "__main__":
    main()