from flask import Flask, request, jsonify,send_file, render_template
from flask_socketio import SocketIO, emit
import os
import urllib.request
import threading
import torch
from model import generate_text, SimpleGPT, create_dataloader_v1, tokenizer
import torch.nn.functional as F

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template("index.html")

# Hyperparameters
hyperparams = {
    "batch_size": 16,
    "max_length": 128,
    "stride": 64,
    "embed_dim": 256,
    "epochs": 10,
    "lr": 3e-4,
}
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = tokenizer.n_vocab

# Global variables
dataloader = None
model = None
embedding_info = []

@app.route("/update_hyperparams", methods=["POST"])
def update_hyperparams():
    data = request.json
    for k in hyperparams:
        if k in data:
            hyperparams[k] = type(hyperparams[k])(data[k])
    return jsonify(hyperparams)

@app.route("/upload_text", methods=["POST"])
def upload_text():
    file_path = "uploaded_text.txt"
    uploaded_file = request.files.get("file")
    url = request.form.get("url")

    # Handle file upload with non-empty filename
    if uploaded_file and uploaded_file.filename:
        try:
            content = uploaded_file.read().decode("utf-8")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"Failed to process uploaded file: {str(e)}", 500

    # Fallback to URL download
    elif url:
        try:
            print(f"Downloading file from URL: {url}")
            urllib.request.urlretrieve(url, file_path)
        except Exception as e:
            return f"Failed to download file from URL: {str(e)}", 500

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    global dataloader
    dataloader = create_dataloader_v1(raw_text, hyperparams["batch_size"])
    if dataloader is None:
        return "Failed to create dataloader", 500
    print(f"dataloader length = {len(dataloader)}")
    return "File processed and dataloader created", 200

@app.route("/train", methods=["POST"])
def train_model():
    thread = threading.Thread(target=training_loop)
    thread.start()
    return "Training started", 200

def training_loop():
    global model, embedding_info
    model = SimpleGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(hyperparams["epochs"]):
        model.train()
        total_loss = 0
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Collect token embedding info from a batch
            with torch.no_grad():
                embeddings = model.token_embedding(input_ids).cpu().numpy()
                tokens = [tokenizer.decode([tok.item()]) for tok in input_ids[0]]
                embedding_info = [
                    {
                        "token": tokens[i],
                        "token_id": input_ids[0][i].item(),
                        "embedding": embeddings[0][i].tolist()
                    }
                    for i in range(len(tokens))
                ]

        socketio.emit("training_status", {"epoch": str(epoch + 1)+"/"+str(hyperparams["epochs"]), "loss": total_loss / len(dataloader)})

    torch.save(model.state_dict(), "simple_gpt.pth")
    socketio.emit("training_complete", {"message": "Training complete."})
    socketio.emit("embedding_info", embedding_info)

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "")
    generated = generate_text(model, prompt)
    return jsonify({"generated_text": generated})

@app.route("/list_models", methods=["GET"])
def list_models():
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    return jsonify(model_files)

@app.route("/test_prompt", methods=["POST"])
def test_prompt():
    data = request.get_json()
    prompt = data.get("prompt")
    model_file = data.get("model")

    if not prompt or not model_file:
        return jsonify({"error": "Prompt and model_file are required"}), 400

    try:
        model = SimpleGPT().to(device)
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
        model.eval()

        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
        for _ in range(50):
            if input_ids.size(1) > hyperparams["max_length"]:
                input_ids = input_ids[:, -1*hyperparams["max_length"]:]
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_text = tokenizer.decode(input_ids[0].tolist())

        return jsonify({"response": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    socketio.run(app, debug=True)
