import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import time
import tensorflow as tf
import joblib

app = Flask(__name__)
CORS(app)

# ===============================
# CONFIG
# ===============================
IMG_HEIGHT = 30
IMG_WIDTH = 30
CONF_THRESHOLD = 60  # 🔥 change (50–70 best)

# ===============================
# LOAD MODELS
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

cnn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.h5"), compile=False)
rnn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "rnn_model.h5"), compile=False)
lstm_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"), compile=False)
knn_model = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))

# ===============================
# LABELS
# ===============================
classes_labels = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)', 9:'No passing',
    10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop',
    15:'No vehicles', 16:'Veh > 3.5 tons prohibited',
    17:'No entry', 18:'General caution',
    19:'Dangerous curve left', 20:'Dangerous curve right',
    21:'Double curve', 22:'Bumpy road',
    23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals',
    27:'Pedestrians', 28:'Children crossing',
    29:'Bicycles crossing', 30:'Beware of ice/snow',
    31:'Wild animals crossing', 32:'End speed + passing limits',
    33:'Turn right ahead', 34:'Turn left ahead',
    35:'Ahead only', 36:'Go straight or right',
    37:'Go straight or left', 38:'Keep right',
    39:'Keep left', 40:'Roundabout mandatory',
    41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

# ===============================
# PREPROCESS
# ===============================
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    normalized = resized.astype("float32") / 255.0

    return (
        normalized.reshape(1, 30, 30, 1),   # CNN
        normalized.reshape(1, 30, 30),      # RNN/LSTM
        normalized.reshape(1, 900)          # KNN
    )

# ===============================
# IMAGE DECODER
# ===============================
def decode_image():
    image = None

    # FILE UPLOAD
    if "image" in request.files:
        file = request.files["image"]
        npimg = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # CAMERA (BASE64)
    elif request.is_json:
        data = request.get_json()
        img_base64 = data.get("image")

        if img_base64:
            try:
                # remove base64 header if present
                if "," in img_base64:
                    img_base64 = img_base64.split(",")[1]

                decoded = base64.b64decode(img_base64)
                npimg = np.frombuffer(decoded, np.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            except:
                return None

    return image

# ===============================
# FORMAT OUTPUT
# ===============================
def format_dl(pred):
    idx = int(np.argmax(pred))
    conf = float(np.max(pred)) * 100

    if conf < CONF_THRESHOLD:
        return {
            "class": "Not Traffic Sign",
            "confidence": round(conf, 2)
        }

    return {
        "class": classes_labels[idx],
        "confidence": round(conf, 2)
    }

# ===============================
# API
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()

    image = decode_image()

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    cnn_in, seq_in, knn_in = preprocess(image)

    # FAST INFERENCE
    cnn_pred = cnn_model.predict(cnn_in, verbose=0)[0]
    rnn_pred = rnn_model.predict(seq_in, verbose=0)[0]
    lstm_pred = lstm_model.predict(seq_in, verbose=0)[0]
    knn_pred = knn_model.predict(knn_in)[0]

    response = {
        "cnn": format_dl(cnn_pred),
        "rnn": format_dl(rnn_pred),
        "lstm": format_dl(lstm_pred),
        "knn": {
            "class": classes_labels[int(knn_pred)],
            "confidence": None
        },
        "time_ms": round((time.time() - start) * 1000, 2)
    }

    return jsonify(response)

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True)