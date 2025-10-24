import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
import json
import random
import logging
import nltk
import numpy as np
from flask import Flask, request, jsonify, render_template
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Setup Flask
app = Flask(__name__, template_folder='templates', static_folder='static')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NLTK setup (avoid runtime downloads on Render)
lemmatizer = WordNetLemmatizer()

# --- Download required data once ---
nltk.data.path.append("/usr/share/nltk_data")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# --- Load intents ---
with open('expanded_medical.json', 'r') as file:
    intents = json.load(file)
logger.info("Loaded intents.")

# --- Load Model ---
MODEL_PATH_KERAS = 'chatbot_model.keras'
MODEL_PATH_H5 = 'chatbot_model.h5'

model = None
if os.path.exists(MODEL_PATH_KERAS):
    model = load_model(MODEL_PATH_KERAS, compile=False)
    logger.info("Loaded model from .keras file")
elif os.path.exists(MODEL_PATH_H5):
    model = load_model(MODEL_PATH_H5, compile=False)
    logger.info("Loaded model from .h5 file")
else:
    raise FileNotFoundError("No model file found!")

# --- Build vocabulary ---
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

def preprocess_input(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = preprocess_input(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

def predict_class(sentence):
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({"error": "Please provide a message."}), 400

        intents_list = predict_class(user_input)
        if intents_list:
            intent = intents_list[0]['intent']
            response = get_response(intent, intents)
        else:
            response = "I didn’t understand that. Can you rephrase?"

        return jsonify({"response": response})
    except Exception as e:
        logger.exception("Error in /chat")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # Don't use debug or threaded mode on Render
    app.run(host="0.0.0.0", port=port)
