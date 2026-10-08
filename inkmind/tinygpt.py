"""TinyGPT as a reusable module.

This mirrors `notebooks/01_TinyGPT.ipynb` exactly - same layers, same forward
pass, same data cleaning, same checkpoint format - so that Task 4/5 can load
the checkpoint produced by that notebook (`inkmind_checkpoint_1000_steps.pth`)
without changing the original notebook.

The original notebook builds TinyGPT from separate layers that are stored in an
`nn.ModuleList` in this order:

    0 token_embedding_table   4 value_layer
    1 position_embedding_table 5 ffn
    2 query_layer             6 layer_norm
    3 key_layer               7 lm_head

`TinyGPT.model_parts()` rebuilds that ModuleList, so the saved
`checkpoint["model_parts"]` state dict loads directly.
"""

import math
import os
import time

import torch
import torch.nn as nn

CHECKPOINT_NAME = "inkmind_checkpoint_1000_steps.pth"


class CharTokenizer:
    """Character-level tokenizer built from the notebook's `stoi` / `itos`."""

    def __init__(self, stoi, itos):
        self.stoi = dict(stoi)
        self.itos = {int(i): ch for i, ch in itos.items()}
        self.vocab_size = len(self.stoi)
        self.last_unknown_chars = []

    @classmethod
    def from_corpus(cls, corpus):
        # Same construction as the notebook: sorted set of characters
        chars = sorted(list(set(corpus)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi, itos)

    def encode(self, text):
        # The notebook raises KeyError on unseen characters; here they are
        # skipped and recorded so a prompt never crashes the comparison.
        self.last_unknown_chars = sorted({ch for ch in text if ch not in self.stoi})
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def tokenize(self, text):
        return [ch for ch in text if ch in self.stoi]


class TinyGPT(nn.Module):
    """One transformer block, single-head causal self-attention."""

    # Structural facts of this architecture (not tunable hyperparameters)
    n_layer = 1
    n_head = 1

    def __init__(self, vocab_size, n_embd=128, block_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.query_layer = nn.Linear(n_embd, n_embd)
        self.key_layer = nn.Linear(n_embd, n_embd)
        self.value_layer = nn.Linear(n_embd, n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def model_parts(self):
        """The same ModuleList (same order) as the original notebook."""
        return nn.ModuleList([
            self.token_embedding_table,
            self.position_embedding_table,
            self.query_layer,
            self.key_layer,
            self.value_layer,
            self.ffn,
            self.layer_norm,
            self.lm_head
        ])

    def forward(self, X):
        # Identical to `forward_pass` in 01_TinyGPT.ipynb

        # 1. Token + position embeddings
        token_embeddings = self.token_embedding_table(X)

        T = X.size(1)

        positions = torch.arange(T, device=X.device)

        position_embeddings = self.position_embedding_table(positions)

        x = token_embeddings + position_embeddings

        # 2. Self-attention
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        attention_scores = Q @ K.transpose(-2, -1)

        d_k = K.size(-1)

        scaled_scores = attention_scores / math.sqrt(d_k)

        causal_mask = torch.tril(torch.ones(T, T, device=X.device))

        scaled_scores = scaled_scores.masked_fill(
            causal_mask == 0,
            float("-inf")
        )

        attention_weights = torch.softmax(scaled_scores, dim=-1)

        context = attention_weights @ V

        # 3. LayerNorm + Feed-Forward Network
        normalized_context = self.layer_norm(context)

        ffn_output = self.ffn(normalized_context)

        # 4. Residual connection
        residual_output = context + ffn_output

        # 5. Predict next character
        logits = self.lm_head(residual_output)

        return logits


# --------------------------------------------------------------------------
# Data preparation (same steps as sections 2-5 of 01_TinyGPT.ipynb)
# --------------------------------------------------------------------------

def load_poetry_poems(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df[["Title", "Poem"]]
    df = df.drop_duplicates(subset="Poem").reset_index(drop=True)

    df["Poem"] = (
        df["Poem"]
        .str.replace(r"\r\n", "\n", regex=False)
        .str.replace(r"\r", "\n", regex=False)
        .str.strip()
    )

    for char in [" ", " ", " ", " ", " "]:
        df["Poem"] = df["Poem"].str.replace(char, " ", regex=False)

    df["Poem"] = df["Poem"].str.replace(" ", "\n", regex=False)
    df["Poem"] = df["Poem"].str.replace("\r", "\n", regex=False)

    for char in ["\x9f", "\xad", "​", "⁠", "﻿"]:
        df["Poem"] = df["Poem"].str.replace(char, "", regex=False)

    df["Poem"] = df["Poem"].str.strip()
    df = df[df["Poem"].str.strip() != ""].reset_index(drop=True)

    return df["Poem"].tolist()


def load_poetry_corpus(csv_path):
    return "\n<|ENDOFPOEM|>\n".join(load_poetry_poems(csv_path))


# --------------------------------------------------------------------------
# Checkpoint loading / saving
# --------------------------------------------------------------------------

def find_checkpoint(extra_paths=()):
    """Look for the TinyGPT checkpoint in the places the team uses."""
    candidates = list(extra_paths) + [
        CHECKPOINT_NAME,
        os.path.join("/content", CHECKPOINT_NAME),
        os.path.join("/content/drive/MyDrive", CHECKPOINT_NAME),
        os.path.join("checkpoints", CHECKPOINT_NAME),
        os.path.join("..", "checkpoints", CHECKPOINT_NAME),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def load_tinygpt(path, device="cpu"):
    """Load a checkpoint saved by 01_TinyGPT.ipynb (or by `train_tinygpt`).

    Returns (model, tokenizer, info). `info["load_time_s"]` is measured here.
    """
    t0 = time.perf_counter()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # The team's own checkpoint also stores optimizer state; older torch
        # versions may need the full unpickler for that.
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        n_embd=checkpoint["n_embd"],
        block_size=checkpoint["block_size"]
    )
    model.model_parts().load_state_dict(checkpoint["model_parts"])
    model = model.to(device).eval()

    tokenizer = CharTokenizer(checkpoint["stoi"], checkpoint["itos"])
    info = {
        "source": f"checkpoint: {path}",
        "load_time_s": time.perf_counter() - t0,
        "training_steps": checkpoint.get("training_steps", "unknown (notebook: 1000)"),
    }
    return model, tokenizer, info


def save_tinygpt(path, model, tokenizer, optimizer=None, **extra):
    """Save in the same format as 01_TinyGPT.ipynb."""
    payload = {
        "model_parts": model.model_parts().state_dict(),
        "vocab_size": model.vocab_size,
        "n_embd": model.n_embd,
        "block_size": model.block_size,
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    payload.update(extra)
    torch.save(payload, path)


# --------------------------------------------------------------------------
# Fallback training (only used when no checkpoint is available)
# --------------------------------------------------------------------------

def train_tinygpt(
    corpus,
    steps=1000,
    batch_size=32,
    block_size=256,
    n_embd=128,
    lr=3e-4,
    seed=42,
    device="cpu",
    log_every=100
):
    """Retrain TinyGPT with the notebook's recipe (90/10 split, AdamW 3e-4,
    batch 32, block 256, 1000 steps total).

    The original notebook did not set a seed, so a retrained model will not be
    bit-identical to the team's checkpoint - only the recipe is the same.
    """
    torch.manual_seed(seed)

    tokenizer = CharTokenizer.from_corpus(corpus)
    data = torch.tensor([tokenizer.stoi[ch] for ch in corpus], dtype=torch.long)

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    def get_batch(split_name):
        data_source = train_data if split_name == "train" else val_data
        ix = torch.randint(len(data_source) - block_size, (batch_size,))
        x = torch.stack([data_source[i:i + block_size] for i in ix])
        y = torch.stack([data_source[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    model = TinyGPT(tokenizer.vocab_size, n_embd=n_embd, block_size=block_size).to(device)
    optimizer = torch.optim.AdamW(model.model_parts().parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        X, Y = get_batch("train")
        logits = model(X)
        B, T, C = logits.shape
        loss = loss_fn(logits.view(B * T, C), Y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            history.append({"step": step, "train_loss": loss.item()})
            print(f"Step {step}: Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        X_val, Y_val = get_batch("val")
        logits_val = model(X_val)
        B, T, C = logits_val.shape
        val_loss = loss_fn(logits_val.view(B * T, C), Y_val.view(B * T)).item()
    print(f"Validation Loss: {val_loss:.4f}")

    info = {
        "source": f"retrained in this session ({steps} steps, seed={seed})",
        "train_time_s": time.perf_counter() - t0,
        "training_steps": steps,
        "val_loss": val_loss,
        "history": history,
    }
    return model, tokenizer, optimizer, info
