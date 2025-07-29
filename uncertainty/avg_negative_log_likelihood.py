# uncertainty/methods/avg_negative_log_likelihood.py

"""
Uncertainty Method: Average Negative Log-Likelihood (NLL)

This method computes the negative log-probability of the true class under the predicted distribution.
"""

import torch
import numpy as np

class AvgNegativeLogLikelihood:
    """
    Class for computing Average Negative Log-Likelihood (NLL) uncertainty.
    """
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def quantify(self, code_str, label_id=None):
        """
        Args:
            code_str (str): Input code snippet.
            label_id (int): Optional. If None, will use predicted label.

        Returns:
            float: Negative log-likelihood score.
        """
        self.model.eval()
        inputs = self.tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # [1, num_classes]
            probs = torch.softmax(logits, dim=-1)  # [1, num_classes]

            if label_id is None:
                label_id = torch.argmax(probs, dim=-1).item()

            logp = torch.log(probs + 1e-12)
            nll = -logp[0, label_id].item()

        return nll
