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
├── notebooks/
│   └── 01_TinyGPT.ipynb
