import os
from datetime import datetime
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from flask import Flask, render_template, request
from pymongo import MongoClient

# Path to the locally downloaded vit-gpt2-image-captioning folder
local_model_path = "vit-gpt2-image-captioning"

# Load model, tokenizer, and feature extractor from local folder
model = VisionEncoderDecoderModel.from_pretrained(local_model_path)
feature_extractor = ViTImageProcessor.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/") 
db = client["image_captioning"]
collection = db["history"]

# Flask app initialization
app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    # Fetch the last 5 records from MongoDB
    history = list(collection.find().sort("timestamp", -1).limit(5))
    return render_template("index.html", history=history)

@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files:
        return "No file part", 400
    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    # Save the image temporarily
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Process the image and generate the caption
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    gen_kwargs = {"max_length": 16, "num_beams": 4}
    output_ids = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Save the result to MongoDB
    collection.insert_one({
        "image_path": image_path,
        "caption": caption,
        "timestamp": datetime.now()
    })

    return render_template("result.html", caption=caption, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
