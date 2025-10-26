<p align="center">
  <img src="https://img.shields.io/badge/UncertaintyZoo-%F0%9F%90%8A%20Unified%20UQ%20Toolkit-blueviolet?style=for-the-badge">
</p>

<h1 align="center">🐘 UncertaintyZoo</h1>

<p align="center">
  <i>A Unified Toolkit for Predictive Uncertainty Quantification</i><br>
  <b>for Discriminative and Generative Models</b>
</p>

---

## 🌍 Overview

**UncertaintyZoo** is a unified, extensible, and easy-to-use Python toolkit for estimating **predictive uncertainty** in deep learning systems.  
It supports a wide range of **uncertainty quantification (UQ) methods** — from probability-based scores to **input-level**, **reasoning-level**, and **information-theoretic** approaches — making it a one-stop solution for uncertainty analysis in both **classification** and **generative** tasks.

---

## 🧪 Quick Start

### 1️⃣ Basic Setup

```python
from uncertainty import Quantifier

# Initialize with model and tokenizer
uq = Quantifier(model, tokenizer, methods=["mc_dropout_var"])

# Compute uncertainty for an input
score = uq.quantify("def add(a, b): return a + b")
print("Uncertainty:", score)
```

### 2️⃣ Advanced Setup (Hybrid / Reasoning / Embedding-level)

```python
uq = Quantifier(
    model=disc_model,
    tokenizer=disc_tokenizer,
    gen_model=gen_model,
    gen_tokenizer=gen_tokenizer,
    embed_model=embed_model,
    embed_tokenizer=embed_tokenizer,
    methods=["topologyuq", "cotuq", "spuq"]
)

score = uq.quantify(prompt="Analyze the following code function:")
```

---

## 🧩 Supported Uncertainty Quantification Methods

<p align="center">
  <b>Total Methods:</b> 29 | <b>Categories:</b> Predictive Distribution · Ensemble · Input Sampling · Reasoning · Representation
</p>

<details>
<summary><b>📊 Expand Table of Methods</b></summary>

| Category                         | Method                                                                                                                                                                               | Task Type |       Level        |   Supported by    |
| :------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------: | :----------------: | :---------------: |
| **Predictive Distribution (11)** | Avg. NLL / Avg. Probability / Perplexity / Max Token Entropy / Avg. Prediction Entropy / Token Impossibility / Margin / Max Prob. / Least Confidence / Predictive Entropy / DeepGini |     C     |   Token / Output   |      🧠 Both      |
| **Ensemble (9)**                 | Expected Entropy / Mutual Info (BALD) / MC Dropout Var / Class Prediction Var / Class Probability Var / Sample Var / Max Diff Var / Min Var / Cosine Similarity                      |     C     | Output / Embedding |      🧠 Both      |
| **Input-Level Sampling (3)**     | SPUQ / ICL-Sample / ICE                                                                                                                                                              |    NC     |       Input        |   🤖 Generative   |
| **Reasoning (5)**                | UAG / CoT-UQ / ToT-UQ / TopologyUQ / SEC                                                                                                                                             |   Both    |     Reasoning      |   🤖 Generative   |
| **Representation (1)**           | Logit Lens Entropy                                                                                                                                                                   |     C     |    Hidden-State    | 🧩 Discriminative |

</details>

---

## 🧠 Model-Specific Usage

### 💎 ① Discriminative Models (e.g., CodeBERT, RoBERTa)

```python
from uncertainty import Quantifier

uq = Quantifier(model, tokenizer, methods=[
    "average_negative_log_likelihood",
    "maximum_probability",
    "deepgini"
])

inputs = <task>
score = uq.quantify(inputs)
print("Discriminative UQ Score:", score)
```

---

### 🔮 ② Generative Models (e.g., ChatGLM, Qwen, LLaMA)

```python
from uncertainty import Quantifier

uq = Quantifier(
    model=gen_model,
    tokenizer=gen_tokenizer,
    methods=["mutual_information", "spuq", "cotuq"]
)

prompt = <prompt> + code_snippet
result = uq.quantify(prompt)

print("Generative UQ:", result)
```

---

## 🧩 Input-Level Sampling Examples

```python
uq = Quantifier(gen_model, gen_tokenizer, methods=["spuq", "icl_sample", "ice"])

# SPUQ: Self-Perturbation UQ
uq.quantify(prompt, N=6)

# ICL-Sample: Few-shot Sampling
uq.quantify(prompt, icl_examples=examples, k_shot=2, n_contexts=5)

# ICE: Input Clarification Ensemble
uq.quantify(prompt, paraphrase_prompts=True)
```

---

## 🧭 Reasoning-Level UQ Examples

```python
uq = Quantifier(gen_model, gen_tokenizer, methods=["uag", "cotuq", "tout", "topologyuq", "sec"])

# Gradient-based attention uncertainty
uag = uq.quantify(prompt, method="uag")

# Chain-of-Thought and Tree-of-Thought
cot = uq.quantify(prompt, method="cotuq", n_paths=5)
tout = uq.quantify(prompt, method="tout", depth=3)

# Structural reasoning
topo = uq.quantify(prompt, method="topologyuq")

# Stable Explanation Consistency
sec = uq.quantify(prompt, method="sec")
```

---

## 🧱 Parameter Legend

| Parameter                        | Description                                                   |
| :------------------------------- | :------------------------------------------------------------ |
| `model`, `tokenizer`             | The primary discriminative or generative model and tokenizer  |
| `gen_model`, `gen_tokenizer`     | Optional: for CoT / ToT / reasoning-level uncertainty         |
| `embed_model`, `embed_tokenizer` | Optional: for embedding-level uncertainty                     |
| `methods`                        | List of UQ method names (e.g., `["mc_dropout_var", "cotuq"]`) |
| `N`                              | Number of perturbations (SPUQ)                                |
| `icl_examples`                   | In-context examples for ICL-Sample                            |
| `num_samples`                    | Ensemble sample count (default=10)                            |
| `prompt`                         | Text or code input string                                     |

---

## 🧾 Citation

If you use **UncertaintyZoo** in your research, please cite:

```
@software{uncertaintyzoo,
  title = {UncertaintyZoo: A Unified Toolkit for Predictive Uncertainty Quantification},
  author = {Wu, Xianzong and Contributors},
  year = {2025},
  url = {https://github.com/UncertaintyZoo/UncertaintyZoo}
}
```

---

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square">
  <img src="https://img.shields.io/badge/python-3.8+-blue?style=flat-square">
  <img src="https://img.shields.io/badge/framework-PyTorch-lightgrey?style=flat-square">
</p>

<p align="center" style="font-size:13px;color:#777;">
  🌟 Designed for research on model reliability, interpretability, and robust reasoning.
</p>
```
