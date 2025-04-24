import torch
import torch.nn as nn
import random
import tiktoken
import torch.nn.functional as F

tokenizer = tiktoken.get_encoding("gpt2")

# Hyperparameters
batch_size = 16
max_length = 128
stride = 64
embed_dim = 256
epochs = 10
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = tokenizer.n_vocab


class GPTDatasetV1:
    def __init__(self, txt):
        self.samples = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.samples.append((
                torch.tensor(input_chunk, dtype=torch.long),
                torch.tensor(target_chunk, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            batch = [self.dataset[idx] for idx in batch_indices]
            input_batch, target_batch = zip(*batch)
            yield torch.stack(input_batch), torch.stack(target_batch)

    def __len__(self):
        full_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            full_batches += 1
        return full_batches


def create_dataloader_v1(txt, batch_size, shuffle=True, drop_last=True):
    print(f"creating dataloader with batch_size={batch_size}, shuffle={shuffle}, drop_last={drop_last}")
    print(f"txt length = {len(txt)}")
    print(f"txt[:10] = {txt[:10]}")
    dataset = GPTDatasetV1(txt)
    return CustomDataLoader(dataset, batch_size, shuffle, drop_last)

def generate_text(model, prompt, max_new_tokens=50):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, -max_length:]

        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    output_text = tokenizer.decode(input_ids[0].tolist())
    return output_text

class SimpleGPT(nn.Module):
    def __init__(self):
        super(SimpleGPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),
            num_layers=4
        )
        self.head = nn.Linear(embed_dim, vocab_size)
        self.max_length = max_length

    def forward(self, x):
        B, T = x.size()
        assert T <= self.max_length, f"Sequence length {T} > max_length {self.max_length}"
        token_emb = self.token_embedding(x)
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb
        x = self.ln(x)
        x = x.permute(1, 0, 2)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = self.transformer(x, mask=causal_mask)
        x = x.permute(1, 0, 2)
        logits = self.head(x)
        return logits
