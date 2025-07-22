# 🤖 Chatbot with NLP

A simple rule-based chatbot built with Python and Natural Language Processing using NLTK and scikit-learn.

## 📁 Project Structure

- `intents.json`: Holds patterns and responses for intents.
- `train.py`: Trains the model on defined intents.
- `chatbot.py`: Contains the NLP logic to process user input.
- `main.py`: Main interaction loop for user-chatbot conversation.
- `requirements.txt`: List of dependencies.

## ⚙️ Installation

```bash
git clone https://github.com/Nomahk25/chatbot-nlp.git
cd chatbot-nlp
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train the Model

Before running the chatbot, train the model with:
```
python train.py
```

### 2. Run the Chatbot

Start chatting with the bot by running:
```
python main.py
```
Type your messages and get intelligent responses based on the trained intents.

## 🧠 Features

- Intent classification with machine learning
- Pattern matching using NLTK
- Rule-based responses with context awareness
- Easy to extend with new intents and responses
- Lightweight and fast interaction

## 🤝 Contributing

Feel free to fork the project, improve it, and submit pull requests!

Ideas:
- Add support for more languages
- Integrate with voice input/output
- Connect chatbot to messaging platforms like Telegram or Slack
- Implement advanced context tracking for multi-turn conversations

## 💡 Inspiration

Inspired by the need for accessible AI-powered chatbots that can be customized easily for various applications, from customer service to learning assistants.
