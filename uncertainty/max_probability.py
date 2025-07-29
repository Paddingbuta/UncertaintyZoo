"""
Uncertainty Method: Maximum Probability

This method computes the maximum class probability from the softmax output.
It reflects the model’s top-1 confidence: higher values = more confident.

max_prob = max(p_i), where p is the predicted class distribution.
"""

import torch
import numpy as np

class MaxProbability:
    """
    Max Probability Uncertainty Estimator.

    Returns the top-1 predicted probability (confidence).
    """

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the estimator.

        Args:
            model (torch.nn.Module): A classification model.
            tokenizer (transformers.PreTrainedTokenizer): Associated tokenizer.
            device (str): Device for inference ('cpu' or 'cuda').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def quantify(self, code_str):
        """
        Compute the maximum predicted probability.

        Args:
            code_str (str): The input source code snippet.

        Returns:
            float: The maximum predicted probability (higher = more confident).
        """
        self.model.eval()
        inputs = self.tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # [batch_size=1, num_classes]
            probs = torch.softmax(logits, dim=-1)  # [1, num_classes]
            probs = probs.squeeze(0).cpu().numpy()  # ensure shape: [num_classes]
            max_prob = np.max(probs)

        return float(max_prob)
