import json
import random
import nltk
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')

with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

all_patterns = []
all_tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern.lower())
        all_patterns.append(" ".join(tokens))
        all_tags.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(all_patterns)
y = np.array(all_tags)

model = MultinomialNB()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Training completed successfully!")
