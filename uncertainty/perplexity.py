"""
Uncertainty Method: Perplexity

This method computes the perplexity of the predicted class distribution, which is 
a commonly used confidence measure in language modeling and classification tasks.
Perplexity is defined as the exponential of the negative log-likelihood (NLL),
and represents how "surprised" the model is by its own prediction.

Lower perplexity = higher confidence.
"""

import torch
import numpy as np

class Perplexity:
    """
    Perplexity-based Uncertainty Estimation for Classification Models.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the estimator.

        Args:
            model (torch.nn.Module): Pretrained classification model (e.g., CodeBERT).
            tokenizer (transformers.PreTrainedTokenizer): Corresponding tokenizer.
            device (str): Device identifier (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def quantify(self, code_str, label_id=None):
        """
        Compute perplexity for the given code sample.

        Args:
            code_str (str): Source code snippet as input.
            label_id (int, optional): True label index. If None, use predicted label.

        Returns:
            float: Perplexity score (lower = more confident).
        """
        self.model.eval()
        inputs = self.tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # [1, num_classes]
            probs = torch.softmax(logits, dim=-1)  # [1, num_classes]

            if label_id is None:
                label_id = torch.argmax(probs, dim=-1).item()

            logp = torch.log(probs + 1e-12)  # Avoid log(0)
            nll = -logp[0, label_id].item()
            perplexity = np.exp(nll)

        return perplexity
