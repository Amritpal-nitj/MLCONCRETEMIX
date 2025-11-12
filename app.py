import os
import joblib
import requests
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# -------------------------------
# CONFIG: File paths & env URLs
# -------------------------------
MODEL_PATH = "concrete_mix_model_checked.joblib"
ENC_PATH = "mix_encoders_checked.joblib"

MODEL_URL = os.environ.get("MODEL_URL")
ENC_URL = os.environ.get("ENC_URL")

# -------------------------------
# Function to download models if missing
# -------------------------------
def download_if_missing(url, path):
    """Download file from given URL if not already present locally."""
    if not url:
        print(f"⚠️ No URL provided for {path}. Skipping download.")
        return
    if not os.path.exists(path):
        print(f"⬇️ Downloading {path} from {url} ...")
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            print(f"✅ Downloaded and saved: {path}")
        except Exception as e:
            print(f"❌ Failed to download {path}: {e}")
    else:
        print(f"✅ {path} already exists, skipping download.")

# -------------------------------
# Download models (only first-time)
# -------------------------------
download_if_missing(MODEL_URL, MODEL_PATH)
download_if_missing(ENC_URL, ENC_PATH)

# -------------------------------
# Load models lazily to save memory
# -------------------------------
model_bundle = None
enc_data = None

def ensure_model_loaded():
    global model_bundle, enc_data
    if model_bundle is None or enc_data is None:
        print("🔄 Loading models into memory...")
        model_bundle = joblib.load(MODEL_PATH)
        enc_data = joblib.load(ENC_PATH)
        print("✅ Models loaded successfully.")
    return model_bundle, enc_data

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Load model on demand
        model_info, enc_info = ensure_model_loaded()
        model = model_info["model"]
        encoders = enc_info["encoders"]
        scaler = enc_info["scaler"]
        num_cols = enc_info["num_cols"]

        # Example of collecting user input (adjust for your actual UI fields)
        fck = float(request.form.get("Grade_fck", 30))
        cement_type = request.form.get("Cement_Type", "OPC43")
        agg_type = request.form.get("Aggregate_Type", "Crushed")
        max_size = float(request.form.get("Max_Aggregate_Size_mm", 20))
        fine_zone = request.form.get("Fine_Agg_Zone", "II")

        # Prepare a dummy input row — adapt as needed
        X_input = pd.DataFrame([{
            "Grade_fck": fck,
            "Cement_Type": cement_type,
            "Max_Aggregate_Size_mm": max_size,
            "Aggregate_Type": agg_type,
            "Fine_Agg_Zone": fine_zone,
            "Exposure_Condition": "Severe",
            "Workability_Slump_mm": 75,
            "Admixture_Type": "Superplasticizer",
            "Admixture_Dosage_%": 1.0,
            "Mineral_Admixture_%": 0.0,
            "w_c_ratio": 0.38,
            "Factor_X": 1.65,
            "Std_Deviation_S": 5.0
        }])

        # Encode categorical values
        for col, le in encoders.items():
            X_input[col] = le.transform(X_input[col].astype(str))
        X_input[num_cols] = scaler.transform(X_input[num_cols])

        # Predict
        y_pred = model.predict(X_input)[0]
        results = {
            "Cementitious_Content_kgm3": round(y_pred[0], 2),
            "Water_Content_kgm3": round(y_pred[1], 2),
            "Fine_Agg_kgm3": round(y_pred[2], 2),
            "Coarse_Agg_kgm3": round(y_pred[3], 2)
        }

        return render_template("index.html", results=results)

    return render_template("index.html")

# -------------------------------
# Render health route
# -------------------------------
@app.route("/health")
def health():
    return "ok", 200

# -------------------------------
# MAIN ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


