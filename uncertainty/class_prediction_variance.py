"""
Uncertainty Method: Class Prediction Variance

Computes the variance of predicted class labels across multiple stochastic forward passes.
A higher variance indicates more disagreement among predictions, hence higher epistemic uncertainty.
"""

import torch
import numpy as np

class ClassPredictionVariance:
    """
    Class for computing variance of predicted class labels from multiple samples.
    """

    def __init__(self, model, tokenizer, device="cpu", n_forward=10):
        """
        Args:
            model: torch.nn.Module, classification model with dropout layers
            tokenizer: tokenizer for input text/code
            device: str, device to run model on ("cpu" or "cuda")
            n_forward: int, number of stochastic forward passes
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_forward = n_forward

    def _sample_labels(self, inputs):
        """
        Sample predicted class labels multiple times.

        Args:
            inputs: tokenized input batch on correct device

        Returns:
            np.ndarray: Array of predicted labels with shape (n_forward,)
        """
        self.model.train()  # Enable dropout
        labels = []
        with torch.no_grad():
            for _ in range(self.n_forward):
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)  # [batch=1, num_classes]
                pred_label = torch.argmax(probs, dim=-1)[0].cpu().item()  # batch=0 label
                labels.append(pred_label)
        self.model.eval()  # Back to eval mode
        return np.array(labels)

    def quantify(self, code_str):
        """
        Compute the variance of predicted class labels.

        Args:
            code_str (str): Input code snippet.

        Returns:
            float: Variance of predicted labels (as uncertainty measure).
        """
        inputs = self.tokenizer(
            code_str,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        labels = self._sample_labels(inputs)  # shape: (n_forward,)
        # Variance of discrete labels can be computed by converting labels to integers and calculating variance
        # Alternatively, variance of class counts normalized by number of samples can be used
        variance = np.var(labels)
        return float(variance)
