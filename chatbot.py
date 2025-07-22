import random
import json
import pickle
import nltk

nltk.download('punkt')

with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def get_response(user_input):
    tokens = nltk.word_tokenize(user_input.lower())
    input_vector = vectorizer.transform([" ".join(tokens)])

    predicted_tag = model.predict(input_vector)[0]
    confidence = max(model.predict_proba(input_vector)[0])

    if confidence > 0.5:
        for intent in intents["intents"]:
            if intent["tag"] == predicted_tag:
                return random.choice(intent["responses"])
    else:
        return "I'm not sure I understand. Can you rephrase?"

