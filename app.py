from flask import Flask, render_template, request
import os
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import gdown

app = Flask(__name__)

# ============================
# STEP 1: LOCAL FILE NAMES
# ============================

MODEL_FILE = "plant_disease_model.h5"
CLASS_FILE = "class_indices.json"
DISEASE_FILE = "disease_info.json"

# ============================
# STEP 2: DOWNLOAD FROM GOOGLE DRIVE SAFELY
# ============================

def download_file(file_id, output_file):
    """Download a file from Google Drive using gdown, with error handling"""
    if not os.path.exists(output_file):
        try:
            print(f"Downloading {output_file} from Google Drive...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)
            print(f"{output_file} downloaded successfully.")
        except Exception as e:
            print(f"Failed to download {output_file}. Check permissions and File ID.")
            print(e)

# Download model and JSON files
download_file(os.environ.get('GDRIVE_MODEL_ID'), MODEL_FILE)
download_file(os.environ.get('GDRIVE_CLASS_ID'), CLASS_FILE)
download_file(os.environ.get('GDRIVE_DISEASE_ID'), DISEASE_FILE)

# ============================
# STEP 3: LOAD MODEL AND JSON DATA
# ============================

try:
    model = keras.models.load_model(MODEL_FILE)
except Exception as e:
    print("Error loading model:", e)
    model = None

try:
    with open(CLASS_FILE, "r") as f:
        class_indices = json.load(f)
        idx_to_class = {v: k for k, v in class_indices.items()}
except Exception as e:
    print("Error loading class indices:", e)
    idx_to_class = {}

try:
    with open(DISEASE_FILE, "r") as f:
        disease_info = json.load(f)
except Exception as e:
    print("Error loading disease info:", e)
    disease_info = {}

IMG_SIZE = 128

# ============================
# STEP 4: FLASK ROUTE
# ============================

@app.route("/", methods=["GET", "POST"])
def home():
    predictions = None
    file_path = None
    suggestion = None

    if request.method == "POST":
        file = request.files.get("leaf_image")
        if file and model is not None:
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
                {"class": idx_to_class.get(i, "Unknown"), "confidence": float(pred[i]) * 100}
                for i in top_indices
            ]

            # Get suggestions for the top prediction
            top_class = predictions[0]["class"]
            suggestion = disease_info.get(top_class, None)

    return render_template(
        "index.html", predictions=predictions, file_path=file_path, suggestion=suggestion
    )

# ============================
# STEP 5: RUN APP
# ============================

if __name__ == "__main__":
    app.run(debug=True)
