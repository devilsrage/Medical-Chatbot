import os
import json
import random
import logging
import nltk
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder='templates', static_folder='static')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()

logger.info("Loading model...")
try:
    model = load_model('chatbot_model.keras', compile=False)
    logger.info("Loaded model from .keras file")
except Exception as e:
    logger.error(f"Error loading .keras model, trying .h5: {e}")
    try:
        model = load_model('chatbot_model.h5', compile=False)
        logger.info("Loaded model from .h5 file")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load any model: {e}")
        raise e

logger.info("Loading vocabulary (words.pkl)...")
try:
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    logger.info("Loaded words.pkl")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load 'words.pkl': {e}")
    raise e

logger.info("Loading labels (classes.pkl)...")
try:
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    logger.info("Loaded classes.pkl")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load 'classes.pkl': {e}")
    raise e
    
logger.info("Loading intents (expanded_medical.json)...")
try:
    with open('expanded_medical.json', 'r') as file:
        intents = json.load(file)
    logger.info("Intents loaded successfully.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load 'expanded_medical.json': {e}")
    raise e

logger.info("All resources loaded. App is ready.")

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
    if len(bow) != len(words):
        logger.warning(f"Bag of words length ({len(bow)}) does not match words list length ({len(words)}). This may be an error.")
        bow_correct_len = np.zeros(len(words))
        for i, word in enumerate(words):
            if word in sentence_words:
                 bow_correct_len[i] = 1
        bow = bow_correct_len

    model_input = np.array([bow])
    
    try:
        res = model.predict(model_input, verbose=0)[0]
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        logger.error(f"Model expected input shape: {model.input_shape}, but got: {model_input.shape}")
        return []

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        if r[0] < len(classes):
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        else:
            logger.warning(f"Prediction index {r[0]} is out of bounds for classes list (len {len(classes)}).")
            
    return return_list


def get_response(intent_tag, intents_json):
    for i in intents_json['intents']:
        if i['tag'] == intent_tag:
            return random.choice(i['responses'])
    logger.warning(f"No response found for intent tag: {intent_tag}")
    return "I’m not sure how to respond to that."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(silent=True)
        if not data or 'message' not in data:
            logger.warning("Bad request: 'message' key missing or invalid JSON.")
            return jsonify({"error": "Please provide a 'message' in the JSON body."}), 400

        user_input = data['message']
        if not user_input:
            return jsonify({"error": "Message cannot be empty."}), 400

        intents_list = predict_class(user_input)
        
        if intents_list:
            intent_tag = intents_list[0]['intent']
            response = get_response(intent_tag, intents)
        else:
            response = "I didn’t understand that. Can you rephrase?"

        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Unhandled error in /chat: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
