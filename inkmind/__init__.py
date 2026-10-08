"""InkMind helpers for Task 4 (GPT-2 generation) and Task 5 (comparison).

Reuses the TinyGPT design from notebooks/01_TinyGPT.ipynb and the GPT-2
loading code from notebooks/GPT2_Exploration_(1).ipynb.
"""

from .generation import (
    GPT2Adapter,
    TinyGPTAdapter,
    filter_logits,
    generate,
    next_token_distribution,
    results_to_dataframe,
)
from .gpt2 import get_device, load_gpt2
from .tinygpt import (
    CHECKPOINT_NAME,
    CharTokenizer,
    TinyGPT,
    find_checkpoint,
    load_poetry_corpus,
    load_poetry_poems,
    load_tinygpt,
    save_tinygpt,
    train_tinygpt,
)
