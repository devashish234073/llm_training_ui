import os
import urllib.request
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# -------------------------------
# Dataset and DataLoader Classes
# -------------------------------
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )
    return dataloader


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Get form inputs: file input and URL text field.
    file_obj = request.files.get("file_input")
    url_input = request.form.get("url_input")
    filename = ""
    
    # Determine file source and file path
    if file_obj and file_obj.filename:
        filename = secure_filename(file_obj.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_obj.save(file_path)
    elif url_input:
        filename = os.path.basename(url_input)
        if not filename.strip():
            filename = "downloaded_file.txt"
        filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # If file does not exist locally, download it
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url_input, file_path)
    else:
        return jsonify({"error": "No file or URL provided."}), 400

    # Read file content
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        return jsonify({"error": f"File read error: {str(e)}"}), 500

    # Process file to extract vocabulary. Here, using a simple space-split.
    preprocessed = raw_text.split()
    all_words = sorted(set(preprocessed))
    
    # Return the raw content and vocabulary list to the frontend.
    return jsonify({
        "raw_text": raw_text,
        "vocab": all_words,
        "file_path": file_path
    })


@app.route('/create_dataloader', methods=['POST'])
def create_dataloader():
    # Retrieve parameters from the frontend (including the file path)
    file_path = request.form.get("file_path")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500

    # Create dataloader using the provided configuration
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=4,
        stride=4,
        shuffle=False
    )

    # Prepare visualization data: process one batch for demonstration
    dataloader_data = []
    for batch_idx, batch in enumerate(dataloader):
        inputs, targets = batch
        inputs_list = inputs.tolist()
        targets_list = targets.tolist()
        dataloader_data.append({
            "batch": batch_idx,
            "input_ids": inputs_list,
            "target_ids": targets_list
        })
        break

    # Also compute the vocabulary and create embedding layers
    preprocessed = raw_text.split()
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    output_dim = 16  # You can adjust this dimension as needed

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    # Extract the weights from the embedding layer as a Python list
    embeddings_weight = token_embedding_layer.weight.detach().tolist()

    # Map each vocabulary word to its embedding vector (assuming the order corresponds)
    embeddings = []
    for word, vector in zip(all_words, embeddings_weight):
        embeddings.append({
            "word": word,
            "embedding": vector
        })

    # Return both dataloader visualization and embeddings info
    return jsonify({
        "dataloader": dataloader_data,
        "embeddings": embeddings
    })


@app.route('/tokenize', methods=['POST'])
def tokenize():
    word = request.form.get("word", "")
    if not word:
        return jsonify({"error": "No word provided."}), 400
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        # Tokenize the single word
        token_ids = tokenizer.encode(word)
        return jsonify({"token_ids": token_ids})
    except Exception as e:
        return jsonify({"error": f"Tokenization error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
