# 🤖 TorchMind - Intent-Based AI Chatbot

TorchMind is an intelligent intent-based chatbot built with **PyTorch** and **Flask**.  
It uses a deep **LSTM neural network** to classify user intents and generate relevant responses.  
Includes a beautiful modern web interface and fully functional backend.

---

## 🔧 Features

- 🧠 Deep LSTM network with embeddings
- 🔎 Simple tokenizer (regex-based)
- 🎯 Intent classification with confidence threshold
- 🌐 Flask web server with HTML/CSS frontend
- 💬 Easy-to-extend intent structure
- ⚡ Fully self-contained — no external services required

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourname/TorchMind.git
cd TorchMind
2. Install Dependencies
bash
Copy
Edit
pip install torch flask
Works with Python 3.8+ (Tested with Python 3.11)

3. Run the App
bash
Copy
Edit
python model.py
Open your browser and go to:

arduino
Copy
Edit
http://localhost:5000
🧠 Model Architecture
The core model is a custom-built deep LSTM:

python
Copy
Edit
class TorchMindLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=3, dropout=0.5)
Embedding layer

Multi-layer LSTM (default: 3 layers)

Dropout regularization

Fully connected layers

ReLU activations

CrossEntropy loss + AdamW optimizer

📦 Intents Structure
Defined as a simple Python list:

python
Copy
Edit
intents = [
    {
        "tag": "greeting",
        "patterns": ["hello", "hi", "hey"],
        "responses": ["Hello!", "Hi there!"]
    },
    ...
]
Add new intents by simply extending this list.

🖥️ Web Interface
Beautiful, responsive frontend using:

HTML5

Bootstrap 5

Custom CSS

JavaScript (with typewriter effect and async chat)

🧪 Example API Usage
Send POST requests to /chat:

bash
Copy
Edit
curl -X POST http://localhost:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "hello"}'
Returns:

json
Copy
Edit
{ "reply": "Hi there!" }
🛠️ Customization Tips
Change the confidence threshold in:

python
Copy
Edit
intent_model.predict_intent(user_msg, threshold=0.80)
To expand vocabulary or improve accuracy, add more training phrases to each intent.

📁 File Structure
csharp
Copy
Edit
TorchMind/
├── model.py           # Main training + Flask server file
├── static/            # (Optional) External assets
├── templates/         # (Optional) HTML templates
└── README.md
❤️ Credits
Built with 💡 by [Your Name] using:

PyTorch

Flask

Bootstrap

Your brain 😉

