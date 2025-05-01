from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from waitress import serve
import logging

app = Flask(__name__, static_folder='static', template_folder='templates')

logging.basicConfig(level=logging.INFO)


app = Flask(__name__, template_folder='templates', static_folder='static')

lemmatizer = WordNetLemmatizer()
logging.info("Loading intents...")
with open('expanded_medical.json', 'r') as file:
    intents = json.load(file)
logging.info("Intents loaded successfully.")

logging.info("Loading model...")
model = load_model('chatbot_model.h5') 
logging.info("Model loaded successfully.")

words = []
classes = []

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
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intent, intents_json):
    for i in intents_json['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "Please provide a message."}), 400

    intents_list = predict_class(user_input)
    if intents_list:
        intent = intents_list[0]['intent']
        response = get_response(intent, intents)
        return jsonify({"response": response})
    else:
        return jsonify({"response": "I didn't understand that. Can you rephrase?"})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
