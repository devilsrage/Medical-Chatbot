# app.py — Render / Railway / Render-proof
import os
# 1) Force CPU before anything else that might import TF
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging
import json
import random
import numpy as np

# Flask imports after env is set
from flask import Flask, request, jsonify, render_template

# Paths & base dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_H5 = os.path.join(BASE_DIR, "chatbot_model.h5")
MODEL_PATH_KERAS = os.path.join(BASE_DIR, "chatbot_model.keras")
INTENTS_PATH = os.path.join(BASE_DIR, "expanded_medical.json")
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Ensure nltk_data dir exists
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
# Tell NLTK where to look first
os.environ["NLTK_DATA"] = NLTK_DATA_DIR

# Import nltk AFTER setting NLTK_DATA
import nltk
# Ensure punkt and wordnet exist BEFORE tokenization
def ensure_nltk():
    missing = []
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        missing.append("punkt")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        missing.append("wordnet")
    if missing:
        logger.info("NLTK data missing: %s. Downloading to %s", missing, NLTK_DATA_DIR)
        # downloads into NLTK_DATA_DIR
        nltk.download("punkt", download_dir=NLTK_DATA_DIR, quiet=True)
        nltk.download("wordnet", download_dir=NLTK_DATA_DIR, quiet=True)
        logger.info("NLTK downloads finished.")
    else:
        logger.info("NLTK data already present.")

ensure_nltk()

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents (fail fast if missing)
logger.info("Loading intents from %s", INTENTS_PATH)
if not os.path.exists(INTENTS_PATH):
    logger.error("Intents file not found at %s", INTENTS_PATH)
    raise FileNotFoundError("expanded_medical.json not found")

with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)
logger.info("Intents loaded: %d tags", len(intents.get("intents", [])))

# Import TensorFlow after NLTK and after we forced CPU
from tensorflow.keras.models import load_model

# Load model safely with logging
model = None
try:
    if os.path.exists(MODEL_PATH_KERAS):
        model = load_model(MODEL_PATH_KERAS, compile=False)
        logger.info("Loaded model from .keras")
    elif os.path.exists(MODEL_PATH_H5):
        model = load_model(MODEL_PATH_H5, compile=False)
        logger.info("Loaded model from .h5")
    else:
        logger.error("No model file found. Looked for %s and %s", MODEL_PATH_KERAS, MODEL_PATH_H5)
        # don't raise here — allow app to startup but return 500 on /chat
except Exception as e:
    logger.exception("Model loading failed: %s", e)
    model = None

# Build vocabulary AFTER having NLTK available
words = []
classes = []
ignore_letters = ["?", "!", ".", ","]

for intent in intents.get("intents", []):
    tag = intent.get("tag")
    for pattern in intent.get("patterns", []):
        # tokenization now safe
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        if tag not in classes:
            classes.append(tag)

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]))
classes = sorted(set(classes))
logger.info("Vocabulary built: %d words, %d classes", len(words), len(classes))


# Helpers
def preprocess_input(sentence: str):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words]


def bag_of_words(sentence: str, words_list):
    sentence_words = preprocess_input(sentence)
    bag = [1 if w in sentence_words else 0 for w in words_list]
    return np.array(bag, dtype=np.float32)


def predict_class(sentence: str):
    if model is None:
        return []
    bow = bag_of_words(sentence, words)
    try:
        preds = model.predict(np.array([bow]), verbose=0)[0]
    except Exception as e:
        logger.exception("Model prediction failed: %s", e)
        return []
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(preds) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def get_response(intent_tag, intents_json):
    for i in intents_json.get("intents", []):
        if i.get("tag") == intent_tag:
            return random.choice(i.get("responses", ["I don't know what to say."]))
    return "I’m not sure how to respond to that."


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not request.is_json:
        return jsonify({"error": "Expected JSON"}), 400
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Please provide a message."}), 400

    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    intents_list = predict_class(user_input)
    if intents_list:
        intent_tag = intents_list[0]["intent"]
        response = get_response(intent_tag, intents)
    else:
        response = "I didn’t understand that. Can you rephrase?"

    return jsonify({"response": response})


# Healthcheck
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Use Flask dev server for local tests only
    app.run(host="0.0.0.0", port=port)
