import os
import json
import random
import logging
import nltk
import numpy as np
from flask import Flask, request, jsonify, render_template

# ---------------------------
# Force CPU only (no GPU)
# ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disables GPU

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_H5 = os.path.join(BASE_DIR, 'chatbot_model.h5')
MODEL_PATH_KERAS = os.path.join(BASE_DIR, 'chatbot_model.keras')
INTENTS_PATH = os.path.join(BASE_DIR, 'expanded_medical.json')
NLTK_DATA_DIR = os.path.join(BASE_DIR, 'nltk_data')

os.makedirs(NLTK_DATA_DIR, exist_ok=True)
os.environ['NLTK_DATA'] = NLTK_DATA_DIR

# ---------------------------
# NLTK setup
# ---------------------------
nltk.download('punkt', download_dir=NLTK_DATA_DIR)
nltk.download('wordnet', download_dir=NLTK_DATA_DIR)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Load intents
# ---------------------------
logger.info("Loading intents...")
with open(INTENTS_PATH, 'r') as file:
    intents = json.load(file)
logger.info("Intents loaded successfully.")

# ---------------------------
# Load model safely
# ---------------------------
from tensorflow.keras.models import load_model

try:
    if os.path.exists(MODEL_PATH_KERAS):
        model = load_model(MODEL_PATH_KERAS, compile=False)
        logger.info("Loaded model from .keras")
    elif os.path.exists(MODEL_PATH_H5):
        model = load_model(MODEL_PATH_H5, compile=False)
        logger.info("Loaded model from .h5")
    else:
        raise FileNotFoundError("No model file found!")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None

# ---------------------------
# Build vocabulary
# ---------------------------
words, classes = [], []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]))
classes = sorted(set(classes))

# ---------------------------
# Helper functions
# ---------------------------
def preprocess_input(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence, words):
    sentence_words = preprocess_input(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

def predict_class(sentence):
    if model is None:
        return []
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intent, intents_json):
    for i in intents_json['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I’m not sure how to respond to that."

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({"error": "Please provide a message."}), 400

        if model is None:
            return jsonify({"error": "Model not loaded."}), 500

        intents_list = predict_class(user_input)
        if intents_list:
            intent = intents_list[0]['intent']
            response = get_response(intent, intents)
        else:
            response = "I didn’t understand that. Can you rephrase?"

        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in /chat: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
