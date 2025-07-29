"""
Uncertainty Method: Average Prediction Entropy

This method computes the entropy of the model's predicted probability distribution over classes,
which reflects the average uncertainty in classification. For classification tasks with a single
output step, this is equivalent to the token-level entropy (i.e., MaxTokenEntropy).

Entropy H(p) = -sum(p_i * log(p_i))

Higher entropy implies greater uncertainty.
"""

import torch
import numpy as np

class AvgPredictionEntropy:
    """
    Average Prediction Entropy Uncertainty Estimator.

    This method computes the entropy of the predicted class probabilities.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the estimator.

        Args:
            model (torch.nn.Module): A classification model (e.g., CodeBERT).
            tokenizer (transformers.PreTrainedTokenizer): Associated tokenizer.
            device (str): 'cpu' or 'cuda' depending on where the model is loaded.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def quantify(self, code_str):
        """
        Compute the average prediction entropy.

        Args:
            code_str (str): Source code snippet.

        Returns:
            float: Entropy score (higher = more uncertain).
        """
        self.model.eval()
        inputs = self.tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # shape: [1, num_classes]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # shape: [num_classes]
            entropy = -np.sum(probs * np.log(probs + 1e-12))  # scalar entropy value

        return entropy
