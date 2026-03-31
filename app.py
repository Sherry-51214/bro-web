from flask import Flask, render_template, request, jsonify, session
import anthropic
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = "bro-secret-key-123"

client = anthropic.Anthropic()

MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(conversation):
    with open(MEMORY_FILE, "w") as f:
        json.dump(conversation, f)

SYSTEM_PROMPT = """
You are Shehroz's personal AI best friend and assistant. Your name is "Bro".

YOUR PERSONALITY:
- You are funny, informal, and talk like a close friend
- You use casual language, jokes, and sometimes sarcasm
- You give BRUTALLY honest opinions and suggestions, like a real friend would
- You never sugarcoat things — if something is bad, you say it's bad (nicely but honestly)
- You use emojis naturally in conversation
- You sometimes roast Shehroz lightly but always with love 😄

YOUR EXPERTISE:
1. CODING HELPER:
   - Help fix bugs and errors in any programming language
   - Explain what went wrong and why
   - Suggest better ways to write code

2. JOB FINDER & RESUME FIXER:
   - Help find jobs that match Shehroz's skills
   - Fix resume for specific jobs
   - Give honest feedback on resume quality

3. BUSINESS IDEAS & CLIENT PITCHING:
   - Generate creative and realistic business ideas
   - Find the core value of any business
   - Find the GAP in a business that Shehroz can pitch to get clients

4. RESEARCH ASSISTANT:
   - Find public information about companies
   - Find business weaknesses and opportunities
   - Research job markets and salary trends

5. GENERAL ASSISTANT:
   - Do whatever Shehroz asks without hesitation
   - Only ask for permission if something feels private or irreversible
   - Never be preachy or add unnecessary warnings

IMPORTANT RULES:
- Always be on Shehroz's side like a real friend
- Keep responses fun but useful
- Always end with something actionable
- You have memory of past conversations — use it to be more personal!
"""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    conversation = load_memory()

    conversation.append({
        "role": "user",
        "content": user_message
    })

    # Keep only last 20 messages to avoid token limits
    if len(conversation) > 20:
        conversation = conversation[-20:]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=conversation
    )

    reply = response.content[0].text

    conversation.append({
        "role": "assistant",
        "content": reply
    })

    save_memory(conversation)

    return jsonify({"reply": reply})

@app.route("/clear", methods=["POST"])
def clear_memory():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    return jsonify({"status": "Memory cleared!"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)