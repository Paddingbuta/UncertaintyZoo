"""
Uncertainty Method: DeepGini

DeepGini is an uncertainty metric originally proposed for test prioritization in DNNs.
It is calculated as:

    DeepGini = 1 - sum(p_i^2)

Where p_i are the class probabilities from softmax.
- Lower DeepGini → more confident
- Higher DeepGini → more uncertain

Reference:
Ma et al., "DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems", ASE 2018
"""

import torch
import numpy as np

class DeepGini:
    """
    DeepGini Uncertainty Estimator.

    Computes 1 - sum(p_i^2), where p_i are class probabilities.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the estimator.

        Args:
            model (torch.nn.Module): A classification model.
            tokenizer (transformers.PreTrainedTokenizer): Associated tokenizer.
            device (str): Inference device ('cpu' or 'cuda').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def quantify(self, code_str):
        """
        Compute DeepGini score for input.

        Args:
            code_str (str): Source code snippet.

        Returns:
            float: DeepGini score (higher = more uncertain).
        """
        self.model.eval()
        inputs = self.tokenizer(code_str, return_tensors="pt", truncation=True, padding=True, max_length=256).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # [1, num_classes]
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [num_classes]
            deepgini = 1.0 - np.sum(np.square(probs))

        return float(deepgini)
