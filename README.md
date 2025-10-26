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
It supports 29+ **Uncertainty Quantification (UQ)** methods — covering probability-based, ensemble-based, input-level, reasoning-level, and topological uncertainty — for both **discriminative** (e.g., CodeBERT) and **generative** (e.g., ChatGLM, Qwen, LLaMA) models.

---

## 🧪 Quick Start

### 1️⃣ Basic Setup

```python
from uncertainty import Quantifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load a discriminative model (e.g., CodeBERT)
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
model.tokenizer = tokenizer

# Initialize Quantifier with a chosen method
uq = Quantifier(model, methods=["mc_dropout_variance"])

# Compute uncertainty for an input
score = uq.quantify(
    input_text="int main(){ char buf[8]; gets(buf); }",
    model_type="discriminative",
    label_tokens=["Yes", "No"]
)

print("Uncertainty:", score)
```

---

### 2️⃣ Generative Model Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from uncertainty import Quantifier

# Load a generative model (e.g., ChatGLM)
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).eval()
model.tokenizer = tokenizer

# Initialize Quantifier
uq = Quantifier(model, methods=["mutual_information", "spuq", "cot_uq"])

# Run uncertainty analysis on a reasoning prompt
prompt = (
    "You are a vulnerability detection assistant.\n"
    "Decide whether the following C code is vulnerable.\n\n"
    "Code:\nint main(){ char buf[8]; gets(buf); }\n\n"
    "Answer only with 'Yes' or 'No'."
)

score = uq.quantify(
    input_text=prompt,
    model_type="generative",
    num_samples=5,
    label_tokens=["Yes", "No"]
)

print(score)
```

---

## 🧩 Supported Uncertainty Quantification Methods

<p align="center">
  <b>Total Methods:</b> 29 | <b>Categories:</b> Predictive · Ensemble · Input Sampling · Reasoning · Representation
</p>

<details>
<summary><b>📊 Expand Table of Methods</b></summary>

| Category                         | Method                                                                                                                                                                                                                                 |     Task Type      |       Level        | Supported Model |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------: | :----------------: | :-------------: |
| **Predictive Distribution (11)** | `average_neg_log_likelihood`, `average_probability`, `perplexity`, `max_token_entropy`, `avg_prediction_entropy`, `token_impossibility_score`, `margin_score`, `max_probability`, `least_confidence`, `predictive_entropy`, `deepgini` |   Classification   |   Token / Output   |      Both       |
| **Ensemble (9)**                 | `expected_entropy`, `mutual_information`, `mc_dropout_variance`, `class_prediction_variance`, `class_probability_variance`, `sample_variance`, `max_diff_variance`, `min_variance`, `cosine_similarity_embeddings`                     |   Classification   | Output / Embedding |      Both       |
| **Input-Level (3)**              | `spuq`, `ice`, `icl_sample`                                                                                                                                                                                                            | Non-Classification |       Input        |   Generative    |
| **Reasoning (5)**                | `uag`, `cot_uq`, `topology_uq`, `tout`, `stable_explanation_conf`                                                                                                                                                                      |        Both        |     Reasoning      |   Generative    |
| **Representation (1)**           | `logit_lens_entropy`                                                                                                                                                                                                                   |   Classification   |    Hidden-State    | Discriminative  |

</details>

---

## 💎 Discriminative Models (e.g., CodeBERT, RoBERTa)

```python
from uncertainty import Quantifier

uq = Quantifier(model, methods=[
    "average_neg_log_likelihood",
    "max_probability",
    "deepgini"
])

score = uq.quantify(
    input_text="int main(){ char buf[8]; gets(buf); }",
    model_type="discriminative",
    label_tokens=["Yes", "No"]
)

print("UQ Score:", score)
```

---

## 🔮 Generative Models (e.g., ChatGLM, Qwen, LLaMA)

```python
from uncertainty import Quantifier

uq = Quantifier(model, methods=["mutual_information", "spuq", "cot_uq"])

prompt = (
    "You are a vulnerability detection assistant.\n"
    "Determine whether this code is vulnerable.\n\n"
    "int main(){ char buf[8]; gets(buf); }\n\n"
    "Answer only 'Yes' or 'No'."
)

score = uq.quantify(
    input_text=prompt,
    model_type="generative",
    num_samples=5,
    label_tokens=["Yes", "No"]
)

print("Generative UQ:", score)
```

---

## 🧩 Input-Level Sampling Methods

```python
uq = Quantifier(model, methods=["spuq", "icl_sample", "ice"])

# Self-Perturbation Uncertainty (SPUQ)
uq.quantify(
    input_text=prompt,
    model_type="generative",
    num_perturb=5,
    label_tokens=["Yes", "No"]
)

# In-Context Learning Sampling (ICL-Sample)
uq.quantify(
    input_text=prompt,
    model_type="generative",
    num_clarifications=6,
    label_tokens=["Yes", "No"]
)

# Input Clarification Ensemble (ICE)
uq.quantify(
    input_text=prompt,
    model_type="generative",
    label_tokens=["Yes", "No"]
)
```

---

## 🧭 Reasoning-Level Uncertainty

```python
uq = Quantifier(model, methods=["uag", "cot_uq", "topology_uq", "stable_explanation_conf", "tout"])

# Attention Gradient Sensitivity
uag = uq.quantify(input_text=prompt, model_type="generative")

# Chain-of-Thought UQ
cot = uq.quantify(input_text=prompt, model_type="generative", num_chains=5)

# Tree-of-Thought UQ
tout = uq.quantify(input_text=prompt, model_type="generative", depth=3, branching=4)

# Topology-based Reasoning Stability
topo = uq.quantify(input_text=prompt, model_type="generative")

# Stable Explanation Confidence
sec = uq.quantify(input_text=prompt, model_type="generative")
```

---

## 🧱 Parameter Legend

| Parameter                                         | Description                                                    |
| :------------------------------------------------ | :------------------------------------------------------------- |
| `model`                                           | The model instance (CodeBERT, ChatGLM, etc.)                   |
| `methods`                                         | List of UQ methods, e.g. `["mc_dropout_variance", "deepgini"]` |
| `input_text`                                      | Text or code input                                             |
| `prompt`                                          | Optional task instruction (mainly for generative models)       |
| `model_type`                                      | `"discriminative"` or `"generative"`                           |
| `label_tokens`                                    | Class tokens for classification-like tasks                     |
| `num_samples`                                     | Number of stochastic forward passes for ensemble methods       |
| `num_perturb`                                     | Number of perturbations for SPUQ                               |
| `num_clarifications`                              | Number of paraphrased prompts for ICL-Sample                   |
| `depth`, `branching`                              | Parameters for tree-of-thought reasoning                       |
| `max_new_tokens`, `temperature`, `top_p`, `top_k` | Sampling parameters for generative inference                   |

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
