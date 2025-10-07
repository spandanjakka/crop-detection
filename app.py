from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np
import json
import os
import gdown

app = Flask(__name__)

# ===============================
# 🔹 STEP 1: DOWNLOAD FILES FROM GOOGLE DRIVE
# ===============================
# These environment variables must be set in Render dashboard
MODEL_ID = os.environ.get("GDRIVE_MODEL_ID")
CLASS_ID = os.environ.get("GDRIVE_CLASS_ID")
DISEASE_ID = os.environ.get("GDRIVE_DISEASE_ID")

os.makedirs("model", exist_ok=True)

MODEL_PATH = "model/plant_disease_model.h5"
CLASS_PATH = "model/class_indices.json"
DISEASE_PATH = "model/disease_info.json"

def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive using its file ID."""
    if not file_id:
        raise ValueError(f"Missing Google Drive ID for {output_path}")
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        print(f"⬇️ Downloading {output_path} from Google Drive...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"✅ {output_path} already exists, skipping download.")

# Download all model files
download_from_gdrive(MODEL_ID, MODEL_PATH)
download_from_gdrive(CLASS_ID, CLASS_PATH)
download_from_gdrive(DISEASE_ID, DISEASE_PATH)

# ===============================
# 🔹 STEP 2: LOAD MODEL AND DATA
# ===============================
print("✅ Loading model and class info...")
model = keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

with open(DISEASE_PATH, "r") as f:
    disease_info = json.load(f)

IMG_SIZE = 128

# ===============================
# 🔹 STEP 3: FLASK ROUTES
# ===============================
@app.route("/", methods=["GET", "POST"])
def home():
    predictions = None
    file_path = None
    suggestion = None

    if request.method == "POST":
        file = request.files["leaf_image"]
        if file:
            # Save uploaded file temporarily
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

            # Get suggestions for top prediction
            top_class = predictions[0]["class"]
            suggestion = disease_info.get(top_class, None)

    return render_template(
        "index.html", predictions=predictions, file_path=file_path, suggestion=suggestion
    )

# ===============================
# 🔹 STEP 4: PORT BINDING FOR RENDER
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT automatically
    app.run(host="0.0.0.0", port=port)
