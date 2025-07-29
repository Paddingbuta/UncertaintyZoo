# UncertaintyZoo ✅

_A Unified Toolkit for Predictive Uncertainty Quantification_

## Overview

**UncertaintyZoo** is a unified, extensible, and easy-to-use Python toolkit for estimating predictive uncertainty in deep learning systems. It supports a wide variety of uncertainty quantification (UQ) methods — from simple probability-based scores to sampling and information-theoretic approaches — and can be easily integrated into existing deep learning pipelines.

### 🧪 Usage 1: Basic Setup

```python
from uncertainty import Quantifier

# Initialize with model and tokenizer
uq = Quantifier(model, tokenizer, methods=["mc_dropout_var"])

# Compute uncertainty for an input
score = uq.quantify(code_str)
```

### 🧠 Usage 2: Advanced Setup (for CoT / Embedding / Reasoning-based Methods)

```python
uq = Quantifier(
    model=base_model,
    tokenizer=base_tokenizer,
    gen_model=generative_model,
    gen_tokenizer=generative_tokenizer,
    embed_model=embed_model,
    embed_tokenizer=embed_tokenizer,
    methods=["topologyuq", "cotuq"]
)
```
