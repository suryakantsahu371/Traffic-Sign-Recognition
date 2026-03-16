import os
import zipfile
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    SimpleRNN, LSTM
)
from tensorflow.keras.optimizers import Adam  # type: ignore

# ==============================
# PATHS
# ==============================
zip_path = r"C:\Users\CYBORG\Downloads\archive.zip"
extract_path = r"C:\Users\CYBORG\Downloads\dataset"
DATA_DIR = os.path.join(extract_path, "Train")
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# EXTRACT DATASET (ONCE)
# ==============================
train_dir = os.path.join(DATA_DIR, "0")
if not os.path.exists(train_dir):
    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_path)
    print("✅ Dataset extracted successfully")
else:
    print("ℹ Dataset already extracted, skipping")

# ==============================
# CONFIG
# ==============================
IMG_HEIGHT = 30
IMG_WIDTH = 30
NUM_CLASSES = 43
EPOCHS = 15
BATCH_SIZE = 64

# ==============================
# LOAD DATA
# ==============================
data = []
labels = []

for class_id in range(NUM_CLASSES):
    class_path = os.path.join(DATA_DIR, str(class_id))
    if not os.path.exists(class_path):
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

        data.append(image)
        labels.append(class_id)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

print(f"📊 Total samples loaded: {len(data)}")

# ==============================
# TRAIN / TEST SPLIT (SINGLE SPLIT FOR ALL MODELS)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)

# ==============================
# CNN MODEL
# ==============================
X_train_cnn = X_train.reshape(-1, 30, 30, 1)
X_test_cnn = X_test.reshape(-1, 30, 30, 1)

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(30, 30, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

cnn_model.compile(
    optimizer=Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cnn_model.fit(
    X_train_cnn, y_train_cat,
    validation_data=(X_test_cnn, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

cnn_model.save(os.path.join(MODEL_DIR, "cnn_model.h5"))

# ==============================
# RNN MODEL
# ==============================
rnn_model = Sequential([
    SimpleRNN(128, input_shape=(30, 30)),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

rnn_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

rnn_model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

rnn_model.save(os.path.join(MODEL_DIR, "rnn_model.h5"))

# ==============================
# LSTM MODEL
# ==============================
lstm_model = Sequential([
    LSTM(128, input_shape=(30, 30)),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

lstm_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

lstm_model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))

# ==============================
# KNN MODEL (CLASSICAL ML)
# ==============================
print("🚀 Training KNN model...")

X_train_knn = X_train.reshape(-1, IMG_HEIGHT * IMG_WIDTH)
X_test_knn = X_test.reshape(-1, IMG_HEIGHT * IMG_WIDTH)

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="euclidean"
)

knn_model.fit(X_train_knn, y_train)

joblib.dump(knn_model, os.path.join(MODEL_DIR, "knn_model.pkl"))

print("✅ CNN, RNN, LSTM, and KNN models trained & saved successfully!")
