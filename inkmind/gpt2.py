"""GPT-2 loading, matching `notebooks/GPT2_Exploration_(1).ipynb`.

Uses the same Hugging Face classes (`GPT2Tokenizer`, `GPT2LMHeadModel`) and the
same pretrained "gpt2" checkpoint. `model_name` can point to a fine-tuned
checkpoint directory later without changing anything else.
"""

import time

import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gpt2(model_name="gpt2", device=None):
    """Returns (model, tokenizer, info). `info["load_time_s"]` covers loading
    both tokenizer and weights (from local cache if already downloaded)."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = device or get_device()

    t0 = time.perf_counter()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_time = time.perf_counter() - t0

    info = {
        "source": f"Hugging Face pretrained: {model_name}",
        "load_time_s": load_time,
        "training_steps": "pretrained (WebText)",
    }
    return model, tokenizer, info
