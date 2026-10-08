# InkMind

InkMind is a small GPT-style language model trained on poetry to generate
creative poem continuations from user-provided prompts.

## Project Objective

The goal of this project is to explore how Transformer-based language
models learn patterns from poetry and generate text from a given prompt.

The project compares two approaches:

- **TinyGPT** — a small GPT-style Transformer built from scratch.
- **GPT-2** — a larger pretrained GPT-style language model fine-tuned on poetry.

## Dataset

The project uses the Poetry Foundation Poems dataset from Kaggle.

The poems were cleaned, normalized, and prepared for language-model training.

## TinyGPT

TinyGPT was implemented from scratch using PyTorch.

The model includes:

- Character-level tokenization
- Token embeddings
- Positional embeddings
- Self-attention
- Causal masking
- Layer normalization
- Feed-forward network
- Residual connection
- Next-token prediction
- Cross-entropy loss
- Autoregressive text generation

## Example Prompt

`Love is a beautiful thing`

TinyGPT generates a continuation by predicting one character at a time based on
the context it has already seen.

## Project Structure

```text
Ink-mind_group2/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── PoetryFoundationData.csv
├── notebooks/
│   ├── 01_TinyGPT.ipynb                 # TinyGPT from scratch
│   ├── GPT2_Exploration_(1).ipynb       # GPT-2 tokenization, architecture, attention heads
│   ├── 04_GPT2_Generation.ipynb         # Task 4: GPT-2 generation & decoding experiments
│   └── 05_Comparison_Integration.ipynb  # Task 5: TinyGPT vs GPT-2 comparison
├── inkmind/                             # reusable code shared by notebooks 04/05 and the app
│   ├── tinygpt.py                       # TinyGPT model (same as 01_TinyGPT), checkpoint load/save
│   ├── gpt2.py                          # GPT-2 loading (same HF classes as the exploration notebook)
│   ├── generation.py                    # one generation loop + greedy/temperature/top-k/top-p sampling
│   ├── comparison.py                    # comparison engine: model stats, speed, resources, text metrics
│   └── plotting.py                      # shared chart style
├── app/
│   └── streamlit_app.py                 # side-by-side generation interface
└── results/                             # CSVs written by notebooks 04/05 (created when run)
```

## Running the Notebooks in Google Colab

The notebooks are developed for Google Colab. Notebooks 04 and 05 start with a
setup cell that, in Colab, clones this repository to `/content/Ink-mind_group2`
(or pulls the latest version) and imports the `inkmind/` package from it. The
same notebooks also run locally from the `notebooks/` folder.

**TinyGPT checkpoint:** `inkmind_checkpoint_1000_steps.pth` (saved by
`01_TinyGPT.ipynb`) is not in Git (`*.pth` is ignored). Notebook 05 looks for
it in the working directory, `/content` and `/content/drive/MyDrive`, can
upload it (`UPLOAD_CHECKPOINT = True`), or retrains TinyGPT with the same
recipe as `01_TinyGPT.ipynb` if it is missing (about 2 minutes on CPU).

## Task 4: GPT-2 Generation

`04_GPT2_Generation.ipynb` shows the autoregressive loop step by step, checks
that our loop matches Hugging Face `model.generate` for greedy decoding, and
runs experiments on the same four prompts for decoding method, temperature,
top-k, top-p, maximum length and seeds.

## Task 5: Comparison & Integration

`05_Comparison_Integration.ipynb` runs TinyGPT and GPT-2 through the same
generation loop and sampler, on the same prompts, seeds and device, and
compares:

- **Model statistics:** parameters, layers, heads, embedding size, context, vocabulary, memory
- **Speed and resources:** load time, latency, characters per second, CPU and GPU memory
- **Output:** decoded text side by side, known-word rate, distinct-n, repetition, diversity across seeds
- **Human ratings:** an exported sheet for coherence, relevance, grammar, consistency and diversity

Because the models use different tokenizers (characters vs BPE), outputs are
compared as decoded text with an equal character budget, and token counts and
tokens per second are not treated as comparable.

## Streamlit Interface

Locally:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

In Colab (after running the setup cell of notebook 05, so the repo and the
TinyGPT checkpoint exist):

```python
!pip install -q streamlit
!cp /content/inkmind_checkpoint_1000_steps.pth /content/Ink-mind_group2/ 2>/dev/null
!streamlit run /content/Ink-mind_group2/app/streamlit_app.py &>/content/streamlit.log &
!npx --yes localtunnel --port 8501
```

Open the printed URL. LocalTunnel asks for a password, which is the Colab VM's
public IP (`!curl -s ipv4.icanhazip.com`).
