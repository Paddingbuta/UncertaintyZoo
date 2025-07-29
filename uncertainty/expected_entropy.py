"""
Uncertainty Method: Expected Entropy

Expected Entropy measures the average entropy over multiple model predictions.
It reflects the **expected uncertainty** of predictions under model stochasticity (e.g., dropout).

Formula:
    expected_entropy = mean_t [ -sum(p_t * log(p_t)) ]
    where t is each MC sample.

This is different from Predictive Entropy, which calculates entropy after averaging probabilities.
"""

import torch
import numpy as np

class ExpectedEntropy:
    """
    Expected Entropy Uncertainty Estimator.

    Computes the average entropy across multiple forward passes with dropout.
    """

    def __init__(self, model, tokenizer, device="cpu", n_forward=10):
        """
        Initialize the estimator.

        Args:
            model (torch.nn.Module): Model with dropout layers.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer.
            device (str): Device to run inference on.
            n_forward (int): Number of MC forward passes.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_forward = n_forward

    def quantify(self, code_str):
        """
        Compute expected entropy over MC samples.

        Args:
            code_str (str): Input source code.

        Returns:
            float: Expected entropy score.
        """
        inputs = self.tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        mc_probs = self._sample_probs(inputs)

        # Compute entropy for each MC sample
        entropies = -np.sum(mc_probs * np.log(mc_probs + 1e-12), axis=1)  # shape: [n_forward]
        return float(np.mean(entropies))  # scalar

    def _sample_probs(self, inputs):
        """
        Perform stochastic forward passes (MC Dropout) and collect probabilities.

        Args:
            inputs (Dict[str, torch.Tensor]): Tokenized input tensors.

        Returns:
            np.ndarray: Shape [n_forward, num_classes]
        """
        self.model.train()  # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(self.n_forward):
                logits = self.model(**inputs).logits  # [1, num_classes]
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [num_classes]
                preds.append(probs)
        self.model.eval()  # Restore eval mode
        return np.stack(preds, axis=0)
