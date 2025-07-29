"""
Uncertainty Method: Least Confidence

This method computes the least confidence score, defined as:
    1 - max(p_i)

Where p_i is the predicted probability for each class.
Lower values indicate high confidence; higher values indicate greater uncertainty.
"""

import torch
import numpy as np

class LeastConfidence:
    """
    Least Confidence Uncertainty Estimator.

    Computes (1 - max predicted probability) as a measure of uncertainty.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the estimator.

        Args:
            model (torch.nn.Module): Classification model.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
            device (str): Device to run model on ('cpu' or 'cuda').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def quantify(self, code_str):
        """
        Compute the least confidence score.

        Args:
            code_str (str): Source code snippet to analyze.

        Returns:
            float: 1 - max class probability.
        """
        self.model.eval()
        inputs = self.tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # shape: [1, num_classes]
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [num_classes]
            max_prob = np.max(probs)
            least_conf = 1.0 - max_prob

        return float(least_conf)
