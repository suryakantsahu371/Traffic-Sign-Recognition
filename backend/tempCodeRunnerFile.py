import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model # type: ignore
import joblib  # for KNN

app = Flask(__name__)