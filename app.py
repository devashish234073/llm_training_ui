import os
import urllib.request
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken
# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# -------------------------------
# Dataset and DataLoader Classes
# -------------------------------
class GPTDatasetV1(Dataset):
    def __init__(self, txt, max_length, stride):
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
    dataset = GPTDatasetV1(txt, max_length, stride)
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
    file_obj = request.files.get("file_input")
    url_input = request.form.get("url_input")
    filename = ""

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
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url_input, file_path)
    else:
        return jsonify({"error": "No file or URL provided."}), 400

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        return jsonify({"error": f"File read error: {str(e)}"}), 500

    # Tokenizer-based vocabulary (token IDs)
    token_ids = tokenizer.encode(raw_text)
    unique_token_ids = sorted(set(token_ids))
    decoded_tokens = [tokenizer.decode([tid]) for tid in unique_token_ids]

    # Return raw text, decoded token vocab and token ids
    return jsonify({
        "raw_text": raw_text,
        "vocab": decoded_tokens,
        "token_ids": unique_token_ids,
        "file_path": file_path
    })



@app.route('/create_dataloader', methods=['POST'])
def create_dataloader():
    file_path = request.form.get("file_path")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=4,
        stride=4,
        shuffle=False
    )

    dataloader_data = []
    for batch_idx, batch in enumerate(dataloader):
        inputs, targets = batch
        inputs_list = inputs.tolist()
        targets_list = targets.tolist()
        decoded_inputs = [tokenizer.decode(input_id_seq) for input_id_seq in inputs_list]
        decoded_targets = [tokenizer.decode(target_id_seq) for target_id_seq in targets_list]
        dataloader_data.append({
            "batch": batch_idx,
            "input_ids": inputs_list,
            "target_ids": targets_list,
            "decoded_inputs":decoded_inputs,
            "decoded_targets":decoded_targets
        })
        if batch_idx > 3:
            break

    # Tokenizer-based vocabulary
    token_ids = tokenizer.encode(raw_text)
    unique_token_ids = sorted(set(token_ids))
    vocab_size = len(unique_token_ids)
    print("vocab_size:", vocab_size)    
    print("tokenizer.n_vocab:", tokenizer.n_vocab)
    vocab_size = tokenizer.n_vocab
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    embeddings_weight = token_embedding_layer.weight.detach().tolist()

    # Decode each token_id to a token string
    decoded_tokens = [tokenizer.decode([tid]) for tid in unique_token_ids]

    embeddings = []
    for tok_str, vector in zip(decoded_tokens, embeddings_weight):
        embeddings.append({
            "token": tok_str,
            "embedding": vector
        })

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
        # Tokenize the single word
        token_ids = tokenizer.encode(word)
        return jsonify({"token_ids": token_ids})
    except Exception as e:
        return jsonify({"error": f"Tokenization error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
