"""InkMind - TinyGPT vs GPT-2 side-by-side generation (Task 5 interface).

Run locally:   streamlit run app/streamlit_app.py
Run in Colab:  see README ("Streamlit interface").

Uses exactly the same generation loop and sampler as the notebooks
(`inkmind.generate`), so the app and the comparison notebook agree.
"""

import os
import sys

import pandas as pd
import streamlit as st
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from inkmind import (CHECKPOINT_NAME, GPT2Adapter, TinyGPTAdapter, find_checkpoint,  # noqa: E402
                     generate, load_gpt2, load_tinygpt)
from inkmind.comparison import count_parameters, parameter_memory_mb, text_metrics  # noqa: E402

st.set_page_config(page_title="InkMind: TinyGPT vs GPT-2", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_gpt2():
    model, tokenizer, info = load_gpt2("gpt2", device=DEVICE)
    return GPT2Adapter(model, tokenizer), info


@st.cache_resource
def get_tinygpt(path):
    model, tokenizer, info = load_tinygpt(path, device=DEVICE)
    return TinyGPTAdapter(model, tokenizer), info


st.title("InkMind: TinyGPT vs GPT-2")
st.caption(
    "Both models use the same generation loop and sampling code. Outputs are compared as decoded "
    "text with the same character budget, because the two models use different tokenizers."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Generation settings")
    strategy = st.radio("Method", ["sample", "greedy"], horizontal=True)
    sampling = strategy == "sample"
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.05, disabled=not sampling)
    use_top_k = st.checkbox("Top-k", value=False, disabled=not sampling)
    top_k = st.slider("k", 1, 200, 40, disabled=not (sampling and use_top_k))
    use_top_p = st.checkbox("Top-p (nucleus)", value=True, disabled=not sampling)
    top_p = st.slider("p", 0.05, 1.0, 0.9, 0.05, disabled=not (sampling and use_top_p))
    max_chars = st.slider("Characters to generate", 20, 600, 200, 10)
    seed = st.number_input("Seed", min_value=0, value=0, step=1)

    st.header("TinyGPT checkpoint")
    default_ckpt = find_checkpoint([os.path.join(REPO_DIR, CHECKPOINT_NAME),
                                    os.path.join(REPO_DIR, "notebooks", CHECKPOINT_NAME)]) or ""
    ckpt_path = st.text_input("Path", value=default_ckpt,
                              help=f"{CHECKPOINT_NAME} from 01_TinyGPT.ipynb or 05_Comparison_Integration.ipynb")

# ---------------------------------------------------------------- models
adapters = []
with st.spinner("Loading GPT-2..."):
    gpt2, gpt2_info = get_gpt2()
if ckpt_path and os.path.exists(ckpt_path):
    tiny, tiny_info = get_tinygpt(ckpt_path)
    adapters.append(tiny)
else:
    tiny = None
    st.warning("TinyGPT checkpoint not found: only GPT-2 will run. Run 05_Comparison_Integration.ipynb "
               "(it retrains and saves the checkpoint if missing) or set the path in the sidebar.")
adapters.append(gpt2)

prompt = st.text_area("Prompt", "Once upon a time", height=80)
cfg = dict(
    strategy=strategy,
    temperature=temperature,
    top_k=top_k if (sampling and use_top_k) else None,
    top_p=top_p if (sampling and use_top_p) else None,
)

if st.button("Generate", type="primary") and prompt.strip():
    cols = st.columns(len(adapters))
    rows = []
    for col, adapter in zip(cols, adapters):
        with col:
            st.subheader(adapter.name)
            with st.spinner(f"{adapter.name} generating..."):
                r = generate(adapter, prompt, max_new_tokens=max_chars, max_new_chars=max_chars,
                             stop_at_eos=True, seed=int(seed), **cfg)
            st.markdown(f"**{prompt}**")
            st.text(r.completion)
            if r.dropped_prompt_chars:
                st.caption(f"Characters unknown to {adapter.name} were dropped: {r.dropped_prompt_chars}")
            m = text_metrics(r.completion)
            rows.append({
                "model": adapter.name,
                "generated tokens (own units)": r.generated_tokens,
                "generated chars": r.generated_chars,
                "generation time (s)": round(r.generation_time_s, 3),
                "chars / s": round(r.chars_per_s, 1),
                "distinct-2": round(m["distinct_2"], 3),
                "repeated 3-gram rate": round(m["repeated_3gram_rate"], 3),
                "stop reason": r.stop_reason,
            })
    st.subheader("Run metrics")
    st.dataframe(pd.DataFrame(rows).set_index("model"), width="stretch")
    st.caption("Generation time = model inference + sampling (tokenization and decoding excluded). "
               "Token counts use each model's own units and are not comparable; chars/s is.")

# ---------------------------------------------------------------- model stats
with st.expander("Model statistics", expanded=False):
    stats = []
    for adapter in adapters:
        model = adapter.model
        is_gpt2 = adapter is gpt2
        stats.append({
            "model": adapter.name,
            "tokenizer": "Byte-level BPE" if is_gpt2 else "Character-level",
            "vocab size": len(adapter.tokenizer) if is_gpt2 else adapter.tokenizer.vocab_size,
            "context length": adapter.context_length,
            "layers": model.config.n_layer if is_gpt2 else model.n_layer,
            "heads": model.config.n_head if is_gpt2 else model.n_head,
            "embedding dim": model.config.n_embd if is_gpt2 else model.n_embd,
            "parameters": f"{count_parameters(model):,}",
            "parameter memory (MB)": round(parameter_memory_mb(model), 1),
            "load time (s)": round((gpt2_info if is_gpt2 else tiny_info)["load_time_s"], 2),
        })
    st.dataframe(pd.DataFrame(stats).set_index("model"), width="stretch")
    st.caption(f"Device: {DEVICE}. GPT-2 is the pretrained 'gpt2' checkpoint (not fine-tuned on poetry).")
