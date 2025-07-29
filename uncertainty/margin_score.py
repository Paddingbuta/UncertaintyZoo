"""
Uncertainty Method: Margin Score

The margin score is defined as the difference between the top-1 and top-2 class probabilities.
A small margin indicates high uncertainty (i.e., the model is unsure between multiple classes),
while a large margin indicates confident prediction.

margin = p_1 - p_2
"""

import torch
import numpy as np

class MarginScore:
    """
    Margin-based Uncertainty Estimator.

    Computes the difference between the highest and second-highest predicted class probabilities.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the estimator.

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
        Compute the margin between top-1 and top-2 predicted class probabilities.

        Args:
            code_str (str): Input source code string.

        Returns:
            float: Margin score (higher = more confident, lower = more uncertain).
        """
        self.model.eval()
        inputs = self.tokenizer(code_str, return_tensors="pt", truncation=True, padding=True, max_length=256).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # shape: [1, num_classes]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # shape: [num_classes]

            if np.allclose(probs, 0):
                # Softmax returned all zeros → invalid prediction
                print("[Warning] All-zero probability distribution detected in margin_score.")
                return 0.0  # or np.nan

            sorted_probs = np.sort(probs)[::-1]  # Descending sort
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            return float(margin)
