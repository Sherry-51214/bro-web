from flask import Flask, render_template, request, jsonify
import anthropic
import os
import json
from duckduckgo_search import DDGS
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2

app = Flask(__name__)
app.secret_key = "bro-secret-key-123"

client = anthropic.Anthropic()

MEMORY_FILE = "memory.json"
DOCS_FOLDER = "documents"

# Initialize ChromaDB and sentence transformer
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("bro_docs")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(conversation):
    with open(MEMORY_FILE, "w") as f:
        json.dump(conversation, f)

def web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                summary = f"Web search results for '{query}':\n\n"
                for i, r in enumerate(results, 1):
                    summary += f"{i}. {r['title']}\n{r['body']}\n\n"
                return summary
            return "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"

def read_pdf(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def read_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def add_document(filename, content):
    chunks = [content[i:i+500] for i in range(0, len(content), 500)]
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{filename}_{i}"]
        )

def search_documents(query):
    try:
        query_embedding = embedder.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        if results["documents"][0]:
            return "\n\n".join(results["documents"][0])
        return ""
    except:
        return ""

def needs_search(message):
    search_keywords = [
        "search", "find", "look up", "latest", "current", "news",
        "today", "recent", "what is", "who is", "how much",
        "price", "job", "salary", "company", "weather"
    ]
    return any(k in message.lower() for k in search_keywords)

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
1. CODING HELPER — fix bugs, explain errors, suggest better code
2. JOB FINDER & RESUME FIXER — find jobs, fix resume, give honest feedback
3. BUSINESS IDEAS & CLIENT PITCHING — find business gaps, pitch ideas
4. RESEARCH ASSISTANT — find public info, market trends, company analysis
5. GENERAL ASSISTANT — do whatever Shehroz asks!

DOCUMENT KNOWLEDGE:
- You have access to Shehroz's personal documents like his resume and study notes
- When answering questions, use this document knowledge to give personalized advice
- If document context is provided, use it to give better answers

IMPORTANT RULES:
- Always be on Shehroz's side like a real friend
- Keep responses fun but useful
- Always end with something actionable
- You have memory of past conversations!
"""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    conversation = load_memory()

    # Search personal documents
    doc_context = search_documents(user_message)

    # Web search if needed
    web_context = ""
    if needs_search(user_message):
        web_context = web_search(user_message)

    # Build enhanced message
    enhanced = user_message
    if doc_context:
        enhanced += f"\n\n[From your documents]: {doc_context}"
    if web_context:
        enhanced += f"\n\n[Web search results]: {web_context}"

    conversation.append({"role": "user", "content": enhanced})

    if len(conversation) > 20:
        conversation = conversation[-20:]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=conversation
    )

    reply = response.content[0].text

    conversation[-1] = {"role": "user", "content": user_message}
    conversation.append({"role": "assistant", "content": reply})

    save_memory(conversation)

    return jsonify({"reply": reply})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    filename = file.filename

    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)

    filepath = os.path.join(DOCS_FOLDER, filename)
    file.save(filepath)

    # Read content based on file type
    if filename.endswith(".pdf"):
        content = read_pdf(filepath)
    elif filename.endswith(".txt"):
        content = read_txt(filepath)
    else:
        return jsonify({"error": "Only PDF and TXT files supported!"})

    add_document(filename, content)

    return jsonify({"success": f"'{filename}' uploaded and learned by Bro! 🧠"})

@app.route("/clear", methods=["POST"])
def clear_memory():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    return jsonify({"status": "Memory cleared!"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)