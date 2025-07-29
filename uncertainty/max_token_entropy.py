"""
Uncertainty Method: Maximum Token Entropy

This method calculates the entropy of the predicted class distribution (softmax output).
It quantifies uncertainty based on the information content in the predicted probabilities.
Higher entropy indicates greater uncertainty in the model's prediction.

Entropy H(p) = -sum(p_i * log(p_i)), where p is the predicted probability distribution.
"""

import torch
import numpy as np

class MaxTokenEntropy:
    """
    Max Token Entropy Uncertainty Estimator.

    This method computes the entropy of the softmax output over classes.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the entropy estimator.

        Args:
            model (torch.nn.Module): Pretrained classification model.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
            device (str): 'cpu' or 'cuda'
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def quantify(self, code_str):
        """
        Compute the entropy over the predicted class distribution.

        Args:
            code_str (str): Source code snippet as input.

        Returns:
            float: Entropy score (higher = more uncertain).
        """
        self.model.eval()
        inputs = self.tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # shape: [1, num_classes]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # shape: [num_classes]
            entropy = -np.sum(probs * np.log(probs + 1e-12))  # Prevent log(0)

        return entropy
