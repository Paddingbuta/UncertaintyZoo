"""
Uncertainty Method: Mutual Information (a.k.a. BALD)

Mutual Information (MI) quantifies epistemic uncertainty — uncertainty from the model itself.
It is defined as the difference between predictive entropy (PE) and expected entropy (EE):

    MI = Predictive Entropy - Expected Entropy

- High MI implies the model is uncertain due to lack of knowledge (e.g., OOD).
- Commonly used in Bayesian Active Learning.

References:
- Houlsby et al., "Bayesian Active Learning by Disagreement" (BALD), 2011
"""

import torch
import numpy as np

class MutualInformation:
    """
    Mutual Information (BALD) Estimator.

    Computes MI = Predictive Entropy - Expected Entropy from multiple forward passes.
    """

    def __init__(self, model, tokenizer, device="cpu", n_forward=10):
        """
        Initialize the estimator.

        Args:
            model (torch.nn.Module): Dropout-enabled model.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer.
            device (str): 'cpu' or 'cuda'
            n_forward (int): Number of MC Dropout passes.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_forward = n_forward

    def quantify(self, code_str):
        """
        Compute mutual information (epistemic uncertainty).

        Args:
            code_str (str): Source code snippet.

        Returns:
            float: Mutual information score.
        """
        inputs = self.tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        mc_probs = self._sample_probs(inputs)

        mean_probs = np.mean(mc_probs, axis=0)  # shape: [num_classes]
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12))

        # Entropy for each MC sample
        sample_entropies = -np.sum(mc_probs * np.log(mc_probs + 1e-12), axis=1)  # [n_forward]
        expected_entropy = np.mean(sample_entropies)

        mi = predictive_entropy - expected_entropy
        return float(mi)

    def _sample_probs(self, inputs):
        """
        Perform stochastic forward passes to get class probabilities.

        Args:
            inputs (Dict[str, torch.Tensor]): Tokenized inputs.

        Returns:
            np.ndarray: [n_forward, num_classes]
        """
        self.model.train()  # Enable dropout
        probs_list = []
        with torch.no_grad():
            for _ in range(self.n_forward):
                logits = self.model(**inputs).logits  # [1, num_classes]
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                probs_list.append(probs)
        self.model.eval()
        return np.stack(probs_list, axis=0)
