"""Model-agnostic autoregressive generation (Task 4) used by both models.

Both TinyGPT and GPT-2 are wrapped in a small adapter exposing:

    encode(text) -> list[int]       decode(ids) -> str
    next_token_logits(ids) -> (1, vocab) logits for the last position

`generate()` then runs the *same* loop and the *same* sampling code for both
models, so differences in the output come from the models, not from the
decoding implementation.

Methodology notes (important for the Task 5 comparison):
- Each step recomputes the forward pass over the last `context_length` tokens
  (no KV cache) for BOTH models. This is simple and identical across models;
  it is slower than Hugging Face's cached `model.generate`.
- Timing is split into: tokenization, generation loop (model inference +
  sampling), and decoding. `tokens_per_s` and `chars_per_s` use the
  generation-loop time only.
- Token counts are in each model's OWN tokens (TinyGPT: characters, GPT-2:
  BPE subwords) and are not directly comparable; `chars_per_s` is.
"""

import time
from dataclasses import asdict, dataclass, field

import torch


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

class TinyGPTAdapter:
    name = "TinyGPT"

    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.context_length = model.block_size
        self.eos_token_id = None

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def next_token_logits(self, ids):
        # Keep only the latest context (same as generate_text in the notebook)
        return self.model(ids[:, -self.context_length:])[:, -1, :]


class GPT2Adapter:
    name = "GPT-2"

    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.context_length = model.config.n_positions
        self.eos_token_id = tokenizer.eos_token_id

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def next_token_logits(self, ids):
        return self.model(ids[:, -self.context_length:]).logits[:, -1, :]


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def filter_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """Apply temperature, then top-k, then top-p (nucleus) filtering.

    Removed tokens get -inf so softmax gives them probability 0.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use strategy='greedy' for argmax)")
    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_best = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))

    if top_p is not None and top_p < 1.0:
        if top_p <= 0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        # Probability mass of all better-ranked tokens. Once it reaches top_p,
        # the nucleus is complete; the most likely token is always kept.
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
        sorted_logits = sorted_logits.masked_fill(mass_before >= top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    return logits


def next_token_distribution(logits, temperature=1.0, top_k=None, top_p=None):
    """Probabilities after filtering - handy for visualising what sampling sees."""
    return torch.softmax(filter_logits(logits, temperature, top_k, top_p), dim=-1)


def select_next_token(logits, strategy="sample", temperature=1.0, top_k=None,
                      top_p=None, generator=None):
    if strategy == "greedy":
        # Greedy ignores temperature / top-k / top-p: always the argmax
        return torch.argmax(logits, dim=-1, keepdim=True)
    if strategy == "sample":
        probs = next_token_distribution(logits, temperature, top_k, top_p)
        return torch.multinomial(probs, num_samples=1, generator=generator)
    raise ValueError(f"unknown strategy: {strategy!r} (use 'greedy' or 'sample')")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

@dataclass
class GenerationResult:
    model: str
    prompt: str
    strategy: str
    temperature: float
    top_k: object
    top_p: object
    seed: object
    max_new_tokens: int
    max_new_chars: object
    prompt_tokens: int
    generated_tokens: int
    generated_chars: int
    completion: str
    full_text: str
    stop_reason: str
    tokenize_time_s: float
    generation_time_s: float
    decode_time_s: float
    total_time_s: float
    tokens_per_s: float
    chars_per_s: float
    device: str
    dropped_prompt_chars: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def generate(adapter, prompt, max_new_tokens=50, strategy="sample",
             temperature=1.0, top_k=None, top_p=None, seed=None,
             stop_at_eos=False, max_new_chars=None):
    """Autoregressive generation: predict, sample, append, repeat.

    max_new_chars: optional extra stop rule on the length of the generated
    text, which gives both models the same *text* budget even though their
    tokens have very different lengths.
    """
    device = adapter.device
    generator = None
    if seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)

    # 1) Tokenization
    t0 = time.perf_counter()
    prompt_ids = adapter.encode(prompt)
    t1 = time.perf_counter()
    dropped = list(getattr(getattr(adapter, "tokenizer", None), "last_unknown_chars", []) or [])
    if not prompt_ids:
        raise ValueError(f"{adapter.name}: prompt has no encodable tokens")

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 2) Generation loop (model inference + sampling)
    _sync(device)
    t2 = time.perf_counter()
    new_ids = []
    approx_chars = 0
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        logits = adapter.next_token_logits(ids)
        next_id = select_next_token(logits, strategy, temperature, top_k, top_p, generator)
        token = int(next_id.item())

        if stop_at_eos and adapter.eos_token_id is not None and token == adapter.eos_token_id:
            stop_reason = "eos"
            break

        ids = torch.cat((ids, next_id.to(device)), dim=1)
        new_ids.append(token)

        if max_new_chars is not None:
            approx_chars += len(adapter.decode([token]))
            if approx_chars >= max_new_chars:
                stop_reason = "max_new_chars"
                break
    _sync(device)
    t3 = time.perf_counter()

    # 3) Decoding
    completion = adapter.decode(new_ids)
    full_text = adapter.decode(prompt_ids + new_ids)
    t4 = time.perf_counter()

    gen_time = t3 - t2
    return GenerationResult(
        model=adapter.name,
        prompt=prompt,
        strategy=strategy,
        temperature=temperature if strategy == "sample" else None,
        top_k=top_k if strategy == "sample" else None,
        top_p=top_p if strategy == "sample" else None,
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_new_chars=max_new_chars,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(new_ids),
        generated_chars=len(completion),
        completion=completion,
        full_text=full_text,
        stop_reason=stop_reason,
        tokenize_time_s=t1 - t0,
        generation_time_s=gen_time,
        decode_time_s=t4 - t3,
        total_time_s=t4 - t0,
        tokens_per_s=len(new_ids) / gen_time if gen_time > 0 else float("nan"),
        chars_per_s=len(completion) / gen_time if gen_time > 0 else float("nan"),
        device=str(device),
        dropped_prompt_chars=dropped,
    )


def results_to_dataframe(results):
    import pandas as pd
    return pd.DataFrame([r.to_dict() for r in results])
