"""
Uncertainty Method: Average Probability

This method computes the average predicted probability across all output classes.
Lower values indicate a less confident (i.e., more uncertain) prediction distribution.
It is typically used as a baseline confidence measure for classification tasks.
"""

import torch
import numpy as np

class AvgProbability:
    """
    Average Probability Uncertainty Estimator.

    This method assumes the output is a softmax over class logits and
    returns the average probability over all classes.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the estimator with model, tokenizer, and device.

        Args:
            model (torch.nn.Module): Pretrained classification model.
            tokenizer (transformers.PreTrainedTokenizer): Corresponding tokenizer.
            device (str): 'cpu' or 'cuda'
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def quantify(self, code_str):
        """
        Compute the average predicted probability over all output classes.

        Args:
            code_str (str): Input source code (function) string.

        Returns:
            float: Average predicted probability (higher = more confident, lower = more uncertain).
        """
        self.model.eval()
        inputs = self.tokenizer(code_str, return_tensors="pt", truncation=True, padding=True, max_length=256).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)  # shape: [1, num_classes]
            avg_prob = probs.mean().item()  # scalar

        return avg_prob
