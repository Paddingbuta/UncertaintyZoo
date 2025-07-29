"""
Uncertainty Method: Monte Carlo Dropout Variance

Performs multiple stochastic forward passes with dropout enabled,
and measures the variance of the predicted class probabilities
to quantify epistemic uncertainty.

Reference:
- Gal and Ghahramani, 2016. Dropout as a Bayesian Approximation.
"""

import torch
import numpy as np

class MCDropoutVariance:
    """
    Class for computing Monte Carlo Dropout Variance uncertainty.
    """
    def __init__(self, model, tokenizer, device="cpu", n_forward=10):
        """
        Args:
            model: torch.nn.Module, classification model with dropout layers
            tokenizer: tokenizer for input text/code
            device: str, device to run the model on ("cpu" or "cuda")
            n_forward: int, number of stochastic forward passes
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_forward = n_forward

    def _sample_probs(self, inputs):
        """
        Perform multiple stochastic forward passes with dropout enabled.

        Args:
            inputs: tokenized inputs on the correct device

        Returns:
            np.ndarray: Array of shape (n_forward, num_classes) with predicted probabilities.
        """
        self.model.train()  # Enable dropout
        probs_list = []
        with torch.no_grad():
            for _ in range(self.n_forward):
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)  # [batch=1, num_classes]
                probs_list.append(probs[0].cpu().numpy())  # Extract batch=0
        self.model.eval()  # Back to eval mode
        return np.stack(probs_list, axis=0)

    def quantify(self, code_str):
        """
        Compute the MC Dropout Variance for the input code string.

        Args:
            code_str (str): Input code snippet.

        Returns:
            float: The mean variance across classes.
        """
        # Tokenize input and move to device
        inputs = self.tokenizer(
            code_str,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        # Sample probabilities with dropout
        probs_samples = self._sample_probs(inputs)  # shape: [n_forward, num_classes]

        # Calculate variance across samples per class
        var_per_class = np.var(probs_samples, axis=0)  # shape: [num_classes]

        # Return mean variance as scalar uncertainty score
        return float(np.mean(var_per_class))
