"""
Uncertainty Method: Predictive Entropy

This method computes the entropy over the mean predicted probabilities
from multiple stochastic forward passes (e.g., with dropout enabled).
It is commonly used in Bayesian deep learning to estimate epistemic uncertainty.

Given:
    mc_probs: [n_forward, num_classes]
    mean_probs = mean(mc_probs, axis=0)
    predictive_entropy = -sum(mean_probs * log(mean_probs))

Higher entropy → greater uncertainty.
"""

import torch
import numpy as np

class PredictiveEntropy:
    """
    Predictive Entropy Uncertainty Estimator.

    Uses multiple forward passes with dropout to estimate uncertainty
    by computing the entropy of the mean probability distribution.
    """

    def __init__(self, model, tokenizer, device="cpu", n_forward=10):
        """
        Initialize the estimator.

        Args:
            model (torch.nn.Module): Classification model with dropout layers.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
            device (str): 'cpu' or 'cuda'
            n_forward (int): Number of MC forward passes.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_forward = n_forward

    def quantify(self, code_str):
        """
        Compute predictive entropy for the given input.

        Args:
            code_str (str): Source code snippet.

        Returns:
            float: Predictive entropy value (higher = more uncertain).
        """
        inputs = self.tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        mc_probs = self._sample_probs(inputs, n_forward=self.n_forward)

        # Mean over MC samples
        mean_probs = np.mean(mc_probs, axis=0)  # shape: [num_classes]

        # Predictive entropy: -sum(p * log p)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12))
        return float(entropy)

    def _sample_probs(self, inputs, n_forward=10):
        """
        Perform multiple stochastic forward passes with dropout.

        Args:
            inputs (Dict[str, torch.Tensor]): Tokenized input tensors.
            n_forward (int): Number of stochastic passes.

        Returns:
            np.ndarray: Array of shape [n_forward, num_classes]
        """
        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_forward):
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)  # [1, num_classes]
                preds.append(probs.squeeze(0).cpu().numpy())
        self.model.eval()
        return np.stack(preds, axis=0)
