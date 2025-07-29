"""
Uncertainty Method: Token Impossibility Score

This method computes the Negative Log-Likelihood (NLL) of the predicted distribution
for a given (true or predicted) class. It is identical in value to average NLL but is
interpreted as a "degree of impossibility" of the correct class under the model.

The higher the NLL, the more 'impossible' (i.e., less likely) the model considers
the correct class to be.

Impossibility = -log(p(y)), where p(y) is the predicted probability for class y.
"""

import torch
import numpy as np

class TokenImpossibilityScore:
    """
    Token Impossibility Score based on Negative Log-Likelihood (NLL).
    """

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the estimator.

        Args:
            model (torch.nn.Module): Classification model with softmax output.
            tokenizer (transformers.PreTrainedTokenizer): Associated tokenizer.
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def quantify(self, code_str, label_id=None):
        """
        Compute the impossibility score (i.e., NLL) for the input.

        Args:
            code_str (str): Input source code snippet.
            label_id (int, optional): Ground-truth label index. If not provided, uses predicted label.

        Returns:
            float: Negative log-probability of the (true or predicted) class.
        """
        self.model.eval()
        inputs = self.tokenizer(code_str, return_tensors="pt", truncation=True, padding=True, max_length=256).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # [1, num_classes]
            probs = torch.softmax(logits, dim=-1)  # [1, num_classes]

            if label_id is None:
                label_id = torch.argmax(probs, dim=-1).item()

            logp = torch.log(probs + 1e-12)
            nll = -logp[0, label_id].item()  # Negative log-probability of class

        return nll
