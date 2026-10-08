"""Comparison engine for TinyGPT vs GPT-2 (Task 5).

    Input prompt
         |
         +------------------+
         v                  v
      TinyGPT             GPT-2        <- same generate() loop + same sampler
         |                  |
      decode             decode        <- compare DECODED TEXT, never token IDs
         +--------+---------+
                  v
         Comparison engine
       speed | output quality | model statistics

Everything here is measured, not assumed. Metrics that cannot be measured
fairly are labelled as such in the returned tables.
"""

import itertools
import os
import re
import time

import torch

from .generation import generate, results_to_dataframe


# --------------------------------------------------------------------------
# Model statistics
# --------------------------------------------------------------------------

def count_parameters(model):
    """Unique parameters (GPT-2's tied lm_head/wte weight is counted once)."""
    return sum(p.numel() for p in model.parameters())


def parameter_memory_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2


def model_statistics(tiny_model, tiny_tokenizer, tiny_info, gpt2_model, gpt2_tokenizer, gpt2_info):
    """Side-by-side table; values are read from the loaded model objects."""
    import pandas as pd

    cfg = gpt2_model.config
    rows = [
        ("Model type", "Decoder-only Transformer (from scratch)", "Decoder-only Transformer (pretrained)"),
        ("Source", tiny_info["source"], gpt2_info["source"]),
        ("Tokenizer", "Character-level", "Byte-level BPE"),
        ("Vocabulary size", tiny_tokenizer.vocab_size, len(gpt2_tokenizer)),
        ("Context length (tokens)", tiny_model.block_size, cfg.n_positions),
        ("Embedding dimension", tiny_model.n_embd, cfg.n_embd),
        ("Transformer layers", tiny_model.n_layer, cfg.n_layer),
        ("Attention heads", tiny_model.n_head, cfg.n_head),
        ("Head dimension", tiny_model.n_embd // tiny_model.n_head, cfg.n_embd // cfg.n_head),
        ("FFN inner size", tiny_model.ffn[0].out_features, 4 * cfg.n_embd),
        ("LayerNorm placement", "After attention (before FFN only)", "Pre-LN (before attn and MLP)"),
        ("Residual connections", "Around FFN only", "Around attention and MLP"),
        ("Parameters", count_parameters(tiny_model), count_parameters(gpt2_model)),
        ("Parameter memory (MB, fp32)", round(parameter_memory_mb(tiny_model), 2),
         round(parameter_memory_mb(gpt2_model), 2)),
        ("Training data", "Poetry Foundation poems (~21M chars)", "WebText (~40 GB, OpenAI)"),
        ("Training", str(tiny_info.get("training_steps")), str(gpt2_info.get("training_steps"))),
        ("Load time (s)", round(tiny_info["load_time_s"], 3) if "load_time_s" in tiny_info else "n/a (trained in session)",
         round(gpt2_info["load_time_s"], 3)),
    ]
    return pd.DataFrame(rows, columns=["Property", "TinyGPT", "GPT-2"]).set_index("Property")


# --------------------------------------------------------------------------
# Resource measurement
# --------------------------------------------------------------------------

def environment_info():
    import platform
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "not installed"
    return info


def measure(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and record wall time, process CPU time and memory.

    cpu_utilisation = process CPU seconds / wall seconds (can exceed 1.0 when
    PyTorch uses several threads). RSS delta is noisy and only indicative.
    GPU peak memory is recorded when CUDA is available.
    """
    try:
        import psutil
        proc = psutil.Process(os.getpid())
    except ImportError:
        proc = None

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    rss0 = proc.memory_info().rss if proc else None
    cpu0 = sum(proc.cpu_times()[:2]) if proc else None
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if on_cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    stats = {"wall_time_s": wall}
    if proc:
        cpu = sum(proc.cpu_times()[:2]) - cpu0
        stats["process_cpu_time_s"] = cpu
        stats["cpu_utilisation"] = cpu / wall if wall > 0 else float("nan")
        stats["rss_after_mb"] = proc.memory_info().rss / 1024 ** 2
        stats["rss_delta_mb"] = (proc.memory_info().rss - rss0) / 1024 ** 2
    if on_cuda:
        stats["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    return out, stats


# --------------------------------------------------------------------------
# Text metrics (computed on decoded text, so they are tokenizer-neutral)
# --------------------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def words(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(word_list, n):
    """Unique n-grams / total n-grams (higher = more varied wording)."""
    grams = ngrams(word_list, n)
    return len(set(grams)) / len(grams) if grams else float("nan")


def repeated_ngram_rate(word_list, n=3):
    """Fraction of word n-grams that already appeared earlier in the text
    (higher = more looping / repetition)."""
    grams = ngrams(word_list, n)
    if not grams:
        return float("nan")
    seen, repeats = set(), 0
    for g in grams:
        repeats += g in seen
        seen.add(g)
    return repeats / len(grams)


def build_reference_vocabulary(poems):
    """Set of lowercase words appearing in the poetry corpus."""
    vocab = set()
    for poem in poems:
        vocab.update(words(poem))
    return vocab


def known_word_rate(word_list, reference_vocab):
    """Share of generated words found in the reference vocabulary.

    A *proxy* for "produces real words", not a measure of quality. It is
    biased in TinyGPT's favour (the vocabulary comes from its training data).
    """
    if not word_list:
        return float("nan")
    return sum(w in reference_vocab for w in word_list) / len(word_list)


def text_metrics(text, reference_vocab=None):
    w = words(text)
    m = {
        "n_words": len(w),
        "distinct_1": distinct_n(w, 1),
        "distinct_2": distinct_n(w, 2),
        "repeated_3gram_rate": repeated_ngram_rate(w, 3),
    }
    if reference_vocab is not None:
        m["known_word_rate"] = known_word_rate(w, reference_vocab)
    return m


def add_text_metrics(df, reference_vocab=None):
    import pandas as pd
    metrics = pd.DataFrame([text_metrics(t, reference_vocab) for t in df["completion"]], index=df.index)
    return pd.concat([df, metrics], axis=1)


def cross_sample_diversity(df, group_cols=("model", "prompt", "config")):
    """Diversity ACROSS seeds for the same model/prompt/config:
    pooled distinct-2 and mean pairwise word-bigram Jaccard distance."""
    import pandas as pd

    rows = []
    for key, g in df.groupby(list(group_cols)):
        texts = list(g["completion"])
        bigram_sets = [set(ngrams(words(t), 2)) for t in texts]
        pooled = list(itertools.chain.from_iterable(ngrams(words(t), 2) for t in texts))
        dists = []
        for a, b in itertools.combinations(bigram_sets, 2):
            union = a | b
            if union:
                dists.append(1 - len(a & b) / len(union))
        rows.append(dict(zip(group_cols, key)) | {
            "n_samples": len(texts),
            "unique_outputs": len(set(texts)),
            "pooled_distinct_2": len(set(pooled)) / len(pooled) if pooled else float("nan"),
            "mean_pairwise_jaccard_distance": sum(dists) / len(dists) if dists else float("nan"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Comparison runner
# --------------------------------------------------------------------------

DEFAULT_CONFIGS = {
    "greedy":            dict(strategy="greedy"),
    "sample T=1.0":      dict(strategy="sample", temperature=1.0),
    "sample T=0.7":      dict(strategy="sample", temperature=0.7),
    "top-k=40 T=0.8":    dict(strategy="sample", temperature=0.8, top_k=40),
    "top-p=0.9 T=0.8":   dict(strategy="sample", temperature=0.8, top_p=0.9),
}


def run_comparison(adapters, prompts, configs=None, seeds=(0,), max_new_chars=200,
                   max_new_tokens=None, stop_at_eos=True, reference_vocab=None,
                   verbose=True):
    """Generate with every adapter x prompt x config x seed.

    Every model gets the same TEXT budget (`max_new_chars`). The token cap
    `max_new_tokens` defaults to max_new_chars, which is always enough because
    every token decodes to at least one character.
    Greedy decoding is deterministic, so it is run once (seed ignored).
    """
    configs = configs or DEFAULT_CONFIGS
    max_new_tokens = max_new_tokens or max_new_chars

    results, config_names = [], []
    for adapter in adapters:
        for prompt in prompts:
            for cfg_name, cfg in configs.items():
                run_seeds = seeds[:1] if cfg.get("strategy") == "greedy" else seeds
                for seed in run_seeds:
                    r = generate(adapter, prompt, max_new_tokens=max_new_tokens,
                                 max_new_chars=max_new_chars, stop_at_eos=stop_at_eos,
                                 seed=seed, **cfg)
                    results.append(r)
                    config_names.append(cfg_name)
                    if verbose:
                        print(f"[{adapter.name:7s}] {cfg_name:16s} seed={seed} "
                              f"{r.generated_tokens:4d} tok {r.generation_time_s:6.2f}s | {prompt!r}")

    df = results_to_dataframe(results)
    df.insert(1, "config", config_names)
    return add_text_metrics(df, reference_vocab)


def speed_summary(df):
    cols = ["generated_tokens", "generated_chars", "generation_time_s",
            "tokens_per_s", "chars_per_s", "tokenize_time_s", "decode_time_s"]
    return df.groupby("model")[cols].mean().round(4)


def quality_summary(df):
    cols = ["n_words", "distinct_1", "distinct_2", "repeated_3gram_rate"]
    if "known_word_rate" in df:
        cols.append("known_word_rate")
    return df.groupby(["model", "config"])[cols].mean().round(3)


def qualitative_rating_template(df):
    """Blank 1-5 rating sheet for human evaluation (one row per output).
    Automatic metrics cannot judge coherence or relevance reliably."""
    sheet = df[["model", "config", "seed", "prompt", "completion"]].copy()
    for col in ["coherence", "relevance", "grammar", "context_consistency", "diversity", "notes"]:
        sheet[col] = ""
    return sheet
