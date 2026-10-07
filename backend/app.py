from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import joblib
import easyocr
import pandas as pd

# ==================================================
# PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "training",
    "model",
    "perfume_model.pkl"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "..",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================================================
# FLASK SETUP
# ==================================================

app = Flask(__name__)
CORS(app)

# ==================================================
# LOAD MODEL
# ==================================================

print("Loading Model...")

model = joblib.load(MODEL_PATH)

print("Model Loaded Successfully!")

# ==================================================
# LOAD OCR
# ==================================================

print("Loading OCR...")

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("OCR Ready!")

# ==================================================
# NOTE CATEGORIES
# ==================================================

CITRUS = [
    "lemon",
    "orange",
    "bergamot",
    "grapefruit",
    "lime",
    "mandarin",
    "citrus"
]

FLORAL = [
    "rose",
    "jasmine",
    "lavender",
    "lily",
    "orchid",
    "violet",
    "peony",
    "floral",
    "orange blossom"
]

WOODY = [
    "cedar",
    "cedarwood",
    "sandalwood",
    "vetiver",
    "woody",
    "wood",
    "patchouli",
    "cashmere wood",
    "cashmere"
]

AMBER = [
    "amber",
    "vanilla",
    "tonka",
    "benzoin"
]

MUSK = [
    "musk"
]

# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_features(text):

    text = text.lower()

    citrus_count = sum(
        1 for item in CITRUS
        if item in text
    )

    floral_count = sum(
        1 for item in FLORAL
        if item in text
    )

    woody_count = sum(
        1 for item in WOODY
        if item in text
    )

    amber_count = sum(
        1 for item in AMBER
        if item in text
    )

    musk_count = sum(
        1 for item in MUSK
        if item in text
    )

    return {
        "Citrus": citrus_count,
        "Floral": floral_count,
        "Woody": woody_count,
        "Amber": amber_count,
        "Musk": musk_count
    }

# ==================================================
# HOME
# ==================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "running",
        "message": "Perfume Longevity Predictor API"
    })

# ==================================================
# PREDICT
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image uploaded"
            }), 400

        image = request.files["image"]

        if image.filename == "":

            return jsonify({
                "success": False,
                "error": "No image selected"
            }), 400

        image_path = os.path.join(
            UPLOAD_FOLDER,
            image.filename
        )

        image.save(image_path)

        # ==========================================
        # OCR
        # ==========================================

        ocr_result = reader.readtext(
            image_path,
            detail=0
        )

        extracted_text = " ".join(ocr_result)

        print("\n========================")
        print("OCR TEXT")
        print("========================")
        print(extracted_text)

        # ==========================================
        # FEATURE EXTRACTION
        # ==========================================

        features = extract_features(
            extracted_text
        )

        print("\n========================")
        print("DETECTED FEATURES")
        print("========================")
        print(features)

        # ==========================================
        # MODEL INPUT
        # ==========================================

        input_df = pd.DataFrame(
            [[
                features["Citrus"],
                features["Floral"],
                features["Woody"],
                features["Amber"],
                features["Musk"]
            ]],
            columns=[
                "Citrus",
                "Floral",
                "Woody",
                "Amber",
                "Musk"
            ]
        )

        # ==========================================
        # PREDICTION
        # ==========================================

        prediction = model.predict(
            input_df
        )[0]

        prediction = round(
            float(prediction),
            2
        )

        print("\n========================")
        print("PREDICTION")
        print("========================")
        print(prediction)

        return jsonify({

            "success": True,

            "ocr_text": extracted_text,

            "detected_notes": {
                "citrus": features["Citrus"],
                "floral": features["Floral"],
                "woody": features["Woody"],
                "amber": features["Amber"],
                "musk": features["Musk"]
            },

            "predicted_longevity": prediction

        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ==================================================
# RUN SERVER
# ==================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )