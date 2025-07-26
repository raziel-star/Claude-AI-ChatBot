import torch
import torch.nn as nn
import torch.optim as optim
import re
import random
from flask import Flask, request, jsonify, render_template_string

# ----------------------------- Simple NLP Helper -----------------------------

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# ----------------------------- Advanced LSTM Model -----------------------------

class TorchMindLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        out = self.dropout(hidden[-1])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ----------------------------- Intent Management -----------------------------

class IntentModel:
    def __init__(self, intents_list):
        self.intents = intents_list
        self.all_words = []
        self.tags = []
        self.patterns = []
        self.labels = []
        self.label_map = {}
        self.reverse_map = {}
        self.model = None
        self.embed_size = 128
        self.hidden_size = 256
        self.num_layers = 3
        self.dropout = 0.5
        self.trained = False

        self.prepare_training_data()
        if self.labels:
            self.train()

    def prepare_training_data(self):
        self.all_words = []
        self.tags = []
        self.patterns = []
        self.labels = []
        for intent in self.intents:
            tag = intent['tag']
            self.tags.append(tag)
            for pattern in intent['patterns']:
                tokens = simple_tokenize(pattern)
                self.patterns.append(tokens)
                self.labels.append(tag)
                for token in tokens:
                    if token not in self.all_words:
                        self.all_words.append(token)
        for idx, tag in enumerate(sorted(set(self.labels))):
            self.label_map[tag] = idx
            self.reverse_map[idx] = tag
        self.word2idx = {word: idx+1 for idx, word in enumerate(self.all_words)}  # +1 for padding

    def encode_sentence(self, tokens):
        return [self.word2idx.get(token, 0) for token in tokens]

    def pad_sequence(self, seq, max_len):
        return seq + [0]*(max_len - len(seq))

    def train(self, epochs=500, lr=0.006):
        max_len = max(len(p) for p in self.patterns)
        X_indices = [self.encode_sentence(tokens) for tokens in self.patterns]
        X_padded = [self.pad_sequence(seq, max_len) for seq in X_indices]
        X_tensor = torch.tensor(X_padded, dtype=torch.long)
        y = torch.tensor([self.label_map[label] for label in self.labels], dtype=torch.long)

        vocab_size = len(self.all_words) + 1
        output_size = len(self.label_map)

        self.model = TorchMindLSTM(vocab_size, self.embed_size, self.hidden_size, output_size, self.num_layers, self.dropout)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.02)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 50 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

        self.trained = True
        print(f"🧠 Training done - Vocab: {vocab_size}, Intents: {output_size}")

    def predict_intent(self, text, threshold=0.80):
        if not self.trained:
            return None, 0.0
        tokens = simple_tokenize(text)
        max_len = max(len(p) for p in self.patterns)
        seq = self.encode_sentence(tokens)
        seq_padded = self.pad_sequence(seq, max_len)
        X = torch.tensor([seq_padded], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            output = self.model(X)[0]
            probs = torch.softmax(output, dim=0)
            conf, pred = torch.max(probs, 0)
            if conf.item() < threshold:
                return None, conf.item()
            return self.reverse_map[pred.item()], conf.item()

    def get_response(self, intent_tag):
        for intent in self.intents:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])
        return "Sorry, I didn't understand. Please try again."

# ----------------------------- Intents (English) -----------------------------

intents = [
    {"tag": "greeting", "patterns": ["hello", "hi", "hey", "how are you?", "what's up?"], 
     "responses": ["Hello!", "Hi there!", "Hey, how can I help you?"]},
    {"tag": "goodbye", "patterns": ["bye", "see you", "goodbye", "later"], 
     "responses": ["Goodbye!", "Have a great day!", "See you soon!"]},
    {"tag": "thanks", "patterns": ["thanks", "thank you", "appreciate it"], 
     "responses": ["You're welcome!", "Glad to help!", "No problem :)"]},
    {"tag": "bot_info", "patterns": ["what's your name?", "who made you?", "are you a bot?"], 
     "responses": ["I'm TorchMind, your AI chatbot.", "I was built with Python and Torch.", "Yes, I'm an AI bot!"]},
    {"tag": "joke", "patterns": ["tell me a joke", "joke", "make me laugh"], 
     "responses": [
         "Why did the programmer quit his job? Because he didn't get arrays.",
         "Why do Java developers wear glasses? Because they don't see sharp.",
         "Why was the computer cold? It left its Windows open!"
     ]},
    {"tag": "help", "patterns": ["what can you do?", "help me", "how do you work?"], 
     "responses": [
         "I can answer questions, tell jokes, and chat with you!",
         "Ask me anything about tech, fun, or general knowledge."
     ]},
    {"tag": "weather", "patterns": ["what's the weather?", "weather today", "is it raining?"], 
     "responses": ["Check the weather here: https://weather.com"]},
    {"tag": "AI", "patterns": ["are you AI?", "what is AI?", "are you artificial intelligence?"], 
     "responses": ["Yes, I'm based on artificial intelligence!", "AI helps computers understand and help humans smartly."]},
]

intent_model = IntentModel(intents)

# ----------------------------- Flask + Modern HTML -----------------------------

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
    <meta charset="UTF-8">
    <title>TorchMind Model</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #232526 0%, #414345 100%);
            min-height: 100vh;
            font-family: 'Varela Round', Arial, sans-serif;
            color: #fff;
        }
        .chat-container {
            max-width: 600px;
            margin: 40px auto;
            background: #23272f;
            border-radius: 18px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.25);
            padding: 0;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(90deg, #4e54c8 0%, #8f94fb 100%);
            color: #fff;
            padding: 24px 32px;
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            letter-spacing: 2px;
            display: flex;
            align-items: center;
            gap: 18px;
            justify-content: center;
        }
        .bot-avatar {
            width: 54px;
            height: 54px;
            border-radius: 50%;
            border: 2px solid #fff;
            background: #fff;
            object-fit: cover;
            box-shadow: 0 2px 8px #0002;
        }
        .chat-box {
            height: 420px;
            overflow-y: auto;
            padding: 24px;
            background: #23272f;
        }
        .chat-msg {
            margin-bottom: 18px;
            display: flex;
            flex-direction: row;
        }
        .chat-msg.user {
            justify-content: flex-end;
        }
        .chat-msg.bot {
            justify-content: flex-start;
        }
        .msg-bubble {
            max-width: 70%;
            padding: 14px 20px;
            border-radius: 18px;
            font-size: 1.1rem;
            line-height: 1.5;
            box-shadow: 0 2px 8px #0001;
        }
        .chat-msg.user .msg-bubble {
            background: linear-gradient(90deg, #8f94fb 0%, #4e54c8 100%);
            color: #fff;
            border-bottom-right-radius: 4px;
        }
        .chat-msg.bot .msg-bubble {
            background: #fff;
            color: #23272f;
            border-bottom-left-radius: 4px;
        }
        .chat-msg.bot .bot-avatar {
            margin-right: 10px;
        }
        .chat-input-area {
            display: flex;
            border-top: 1px solid #2e3240;
            background: #23272f;
            padding: 18px 24px;
        }
        .chat-input-area input {
            flex: 1;
            border: none;
            border-radius: 8px;
            padding: 12px;
            font-size: 1.1rem;
            margin-right: 12px;
            background: #2e3240;
            color: #fff;
        }
        .chat-input-area input::placeholder {
            color: #bfc4d1;
        }
        .chat-input-area button {
            background: linear-gradient(90deg, #4e54c8 0%, #8f94fb 100%);
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 12px 28px;
            font-size: 1.1rem;
            font-weight: bold;
            transition: background 0.2s;
        }
        .chat-input-area button:hover {
            background: linear-gradient(90deg, #8f94fb 0%, #4e54c8 100%);
        }
        @media (max-width: 600px) {
            .chat-container { max-width: 100%; margin: 0; border-radius: 0; }
            .chat-header { font-size: 1.3rem; padding: 16px 8px; }
            .chat-box { padding: 12px; height: 320px; }
            .chat-input-area { padding: 10px 8px; }
            .bot-avatar { width: 36px; height: 36px; }
        }
    </style>
</head>
<body>
    <div class="chat-container shadow">
        <div class="chat-header">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" class="bot-avatar" alt="bot">
            <span>🤖 TorchMind - English Model</span>
        </div>
        <div class="chat-box" id="chat-box">
            <div class="chat-msg bot">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" class="bot-avatar" alt="bot">
                <div class="msg-bubble" id="first-bot-msg">
                    Hello! I'm TorchMind, your AI chatbot. How can I help you?
                </div>
            </div>
        </div>
        <form class="chat-input-area" id="chat-form" autocomplete="off">
            <input type="text" id="user-input" placeholder="Type your message..." autofocus required />
            <button type="submit">Send</button>
        </form>
    </div>
    <script>
        const chatBox = document.getElementById('chat-box');
        const chatForm = document.getElementById('chat-form');
        const userInput = document.getElementById('user-input');

        function appendMessage(text, sender) {
            const msgDiv = document.createElement('div');
            msgDiv.className = 'chat-msg ' + sender;
            if(sender === 'bot') {
                const avatar = document.createElement('img');
                avatar.src = "https://cdn-icons-png.flaticon.com/512/4712/4712035.png";
                avatar.className = "bot-avatar";
                avatar.alt = "bot";
                msgDiv.appendChild(avatar);
            }
            const bubble = document.createElement('div');
            bubble.className = 'msg-bubble';
            bubble.innerText = text;
            msgDiv.appendChild(bubble);
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            return bubble;
        }

        function appendBotTyping() {
            const msgDiv = document.createElement('div');
            msgDiv.className = 'chat-msg bot';
            const avatar = document.createElement('img');
            avatar.src = "https://cdn-icons-png.flaticon.com/512/4712/4712035.png";
            avatar.className = "bot-avatar";
            avatar.alt = "bot";
            msgDiv.appendChild(avatar);
            const bubble = document.createElement('div');
            bubble.className = 'msg-bubble';
            bubble.innerHTML = '<span class="typing"></span>';
            msgDiv.appendChild(bubble);
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            return bubble;
        }

        function typeWriterEffect(element, text, speed=28) {
            element.innerHTML = '';
            let i = 0;
            function type() {
                if (i < text.length) {
                    element.innerHTML += text.charAt(i);
                    i++;
                    setTimeout(type, speed);
                }
            }
            type();
        }

        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const text = userInput.value.trim();
            if (!text) return;
            appendMessage(text, 'user');
            userInput.value = '';
            const bubble = appendBotTyping();
            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            })
            .then(res => res.json())
            .then(data => {
                setTimeout(() => {
                    typeWriterEffect(bubble, data.reply);
                }, 400);
            })
            .catch(() => {
                typeWriterEffect(bubble, 'Server error. Please try again later.');
            });
        });

        // Typewriter effect for first bot message
        window.onload = function() {
            const firstMsg = document.getElementById('first-bot-msg');
            if(firstMsg) {
                const text = firstMsg.innerText;
                firstMsg.innerText = '';
                typeWriterEffect(firstMsg, text, 22);
            }
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")
    intent_tag, confidence = intent_model.predict_intent(user_msg, threshold=0.80)
    if intent_tag:
        reply = intent_model.get_response(intent_tag)
    else:
        reply = "Sorry, I didn't understand. Please try again."
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(port=5000, debug=True)