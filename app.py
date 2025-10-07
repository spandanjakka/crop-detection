from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np
import json
import os
import gdown
import zipfile

app = Flask(__name__)

# ---------------------------
# Configuration for model download
# ---------------------------
GDRIVE_MODEL_ID = os.environ.get("1iwbyfuFVGcwDAWORLdZRHkeEB2VlP-z8")          # model.h5
GDRIVE_CLASS_ID = os.environ.get("1cfTP2ABfRuzCGaHR3sFth5FxUuac9NVM")          # class_indices.json
GDRIVE_DISEASE_ID = os.environ.get("1NaTVv_8OjCLQw9CMhyzJM6TlBblT3dwO")      # disease_info.json
MODEL_DIR = "model"
IMG_SIZE = 128

os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        # If zip file, extract
        if zipfile.is_zipfile(output_path):
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(MODEL_DIR)
            os.remove(output_path)

# ---------------------------
# Download model & JSON files
# ---------------------------
download_file(GDRIVE_MODEL_ID, os.path.join(MODEL_DIR, "model.h5"))
download_file(GDRIVE_CLASS_ID, os.path.join(MODEL_DIR, "class_indices.json"))
download_file(GDRIVE_DISEASE_ID, os.path.join(MODEL_DIR, "disease_info.json"))

# ---------------------------
# Load model & metadata
# ---------------------------
model = keras.models.load_model(os.path.join(MODEL_DIR, "model.h5"))

with open(os.path.join(MODEL_DIR, "class_indices.json"), "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

with open(os.path.join(MODEL_DIR, "disease_info.json"), "r") as f:
    disease_info = json.load(f)

# ---------------------------
# Flask routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    predictions = None
    file_path = None
    suggestion = None

    if request.method == "POST":
        file = request.files["leaf_image"]
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            # Preprocess image
            img = image.load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            pred = model.predict(img_array)[0]
            top_indices = pred.argsort()[-3:][::-1]
            predictions = [
                {"class": idx_to_class[i], "confidence": float(pred[i]) * 100}
                for i in top_indices
            ]

            top_class = predictions[0]["class"]
            suggestion = disease_info.get(top_class, None)

    return render_template(
        "index.html", predictions=predictions, file_path=file_path, suggestion=suggestion
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
