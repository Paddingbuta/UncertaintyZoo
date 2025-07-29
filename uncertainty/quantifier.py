"""
Quantifier Class (Extended Version)
====================================

Supports unified uncertainty quantification over code/text inputs using various scoring methods.
Now supports advanced reasoning-based and representation-based methods requiring multiple models.

Supports 29 UQ methods:
------------------------
🧮 Predictive Distribution (11 methods)
🧠 Ensemble-Based (9 methods)
📝 Input-Level Sampling (3 methods)
🔗 Reasoning-Level (5 methods)
🔍 Representation-Based (1 method)

Usage Example:
--------------
from uq.quantifier import Quantifier
quant = Quantifier(
    model=base_model,
    tokenizer=base_tokenizer,
    gen_model=chatglm,
    gen_tokenizer=chatglm_tokenizer,
    embed_model=codebert,
    embed_tokenizer=codebert_tokenizer,
    methods=["predictive_entropy", "topologyuq"]
)
scores = quant.quantify("def foo(): ...")

Returns:
--------
Dictionary mapping each method name to its uncertainty score.
"""

import torch
import numpy as np
from .scoring import (
    avg_negative_log_likelihood,
    avg_probability,
    perplexity,
    max_token_entropy,
    avg_prediction_entropy,
    token_impossibility_score,
    margin_score,
    max_probability,
    least_confidence,
    predictive_entropy,
    deepgini,

    expected_entropy,
    mutual_information,
    mc_dropout_var,
    class_prediction_variance,
    class_probability_variance,
    sample_variance,
    max_difference_variance,
    min_variance,
    cosine_similarity_embeddings,

    spuq,
    icl_sampling,
    input_clarification_ensembles,

    uag_attention,
    cotuq,
    tout,
    topologyuq,
    stable_explanation_confidence,

    logit_lens_entropy,
    prompt_confidence,
)

class Quantifier:
    def __init__(self, model, tokenizer, methods=None, device="cpu",
                 gen_model=None, gen_tokenizer=None,
                 embed_model=None, embed_tokenizer=None):
        """
        Args:
            model: Base model for scoring (e.g. CodeBERT).
            tokenizer: Corresponding tokenizer for base model.
            gen_model: (Optional) Generative model for reasoning-based methods.
            gen_tokenizer: Tokenizer for gen_model.
            embed_model: (Optional) Embedding model (for CoT or similarity-based methods).
            embed_tokenizer: Tokenizer for embed_model.
            methods: List of UQ method names to use.
            device: 'cuda' or 'cpu'.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.gen_model = gen_model
        self.gen_tokenizer = gen_tokenizer
        self.embed_model = embed_model
        self.embed_tokenizer = embed_tokenizer
        self.methods = methods if methods else []
        self.device = device

    def quantify(self, code_str):
        """Compute all specified uncertainty scores for a given code input."""
        results = {}
        inputs = self.tokenizer(code_str, return_tensors="pt", truncation=True, padding=True).to(self.device)

        for method in self.methods:
            try:
                if method == "mc_dropout_var":
                    _, var = mc_dropout_var(self.model, inputs, n_forward=10)
                    results[method] = var.mean().item()

                elif method == "predictive_entropy":
                    probs = self._sample_probs(inputs)
                    results[method] = predictive_entropy(probs)

                elif method == "mutual_information":
                    probs = self._sample_probs(inputs)
                    results[method] = mutual_information(probs)

                elif method == "expected_entropy":
                    probs = self._sample_probs(inputs)
                    results[method] = expected_entropy(probs)

                elif method == "class_prediction_variance":
                    results[method] = class_prediction_variance(self.model, inputs, n_samples=10)

                elif method == "class_probability_variance":
                    results[method] = class_probability_variance(self.model, inputs, n_samples=10)

                elif method == "sample_variance":
                    results[method] = sample_variance(self.model, inputs, n_samples=10)

                elif method == "max_difference_variance":
                    results[method] = max_difference_variance(self.model, inputs, n_samples=10)

                elif method == "min_variance":
                    results[method] = min_variance(self.model, inputs, n_samples=10)

                elif method == "cosine_similarity_embeddings":
                    results[method] = cosine_similarity_embeddings(self.model, self.tokenizer, code_str, self.device)

                elif method == "avg_negative_log_likelihood":
                    results[method] = avg_negative_log_likelihood(self.model, inputs)

                elif method == "avg_probability":
                    results[method] = avg_probability(self.model, inputs)

                elif method == "perplexity":
                    results[method] = perplexity(self.model, inputs)

                elif method == "max_token_entropy":
                    results[method] = max_token_entropy(self.model, inputs)

                elif method == "avg_prediction_entropy":
                    results[method] = avg_prediction_entropy(self.model, inputs)

                elif method == "token_impossibility_score":
                    results[method] = token_impossibility_score(self.model, inputs)

                elif method == "margin_score":
                    results[method] = margin_score(self.model, inputs)

                elif method == "max_probability":
                    results[method] = max_probability(self.model, self.tokenizer, code_str, self.device)

                elif method == "least_confidence":
                    results[method] = least_confidence(self.model, inputs)

                elif method == "deepgini":
                    results[method] = deepgini(self.model, inputs)

                elif method == "prompt_confidence":
                    score, _ = prompt_confidence(code_str, self.model, self.tokenizer, device=self.device)
                    results[method] = 1 - score if score is not None else None

                elif method == "spuq":
                    results[method] = spuq(self.model, self.tokenizer, code_str, self.device)

                elif method == "icl_sampling":
                    results[method] = icl_sampling(self.model, self.tokenizer, code_str, self.device)

                elif method == "input_clarification_ensembles":
                    results[method] = input_clarification_ensembles(self.model, self.tokenizer, code_str, self.device)

                elif method == "uag_attention":
                    results[method] = uag_attention(self.model, self.tokenizer, code_str, self.device)

                elif method == "cotuq":
                    results[method] = cotuq(self.gen_model, self.gen_tokenizer, code_str, self.device)

                elif method == "tout":
                    results[method] = tout(self.gen_model, self.gen_tokenizer, code_str, self.device)

                elif method == "topologyuq":
                    results[method] = topologyuq(self.gen_model, self.gen_tokenizer,
                                                  self.embed_model, self.embed_tokenizer,
                                                  code_str, self.device)

                elif method == "stable_explanation_confidence":
                    results[method] = stable_explanation_confidence(self.gen_model, self.gen_tokenizer,
                                                                    self.embed_model, self.embed_tokenizer,
                                                                    code_str, self.device)

                elif method == "logit_lens_entropy":
                    results[method] = logit_lens_entropy(self.model, inputs, layer=6)

                else:
                    results[method] = None  # unknown method

            except Exception as e:
                results[method] = None  # safe fallback
                print(f"[Warning] Method '{method}' failed with error: {e}")

        return results

    def _sample_probs(self, inputs, n=10):
        """Monte Carlo sampling from model with dropout enabled."""
        self.model.train()
        probs_list = []
        with torch.no_grad():
            for _ in range(n):
                out = self.model(**inputs)
                probs = torch.softmax(out.logits, dim=-1)
                probs_list.append(probs[0].cpu().numpy())
        self.model.eval()
        return np.stack(probs_list, axis=0)
