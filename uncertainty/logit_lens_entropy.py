"""
logit_lens_entropy.py

Uncertainty Method: Logit Lens Entropy

This method computes entropy over softmax-transformed logits from an intermediate hidden layer.
It reflects the uncertainty present in the model’s internal representation (before the final output layer).

Reference:
    LogitLensEntropy^(l) = -∑_i σ(z^(l))_i * log(σ(z^(l))_i)

Usage:
    from uncertainty.methods.logit_lens_entropy import LogitLensEntropy

    lens = LogitLensEntropy(model, tokenizer, target_layer=-2)
    uncertainty_score = lens.quantify("your input code/question")
"""

import torch
import torch.nn.functional as F
import numpy as np


class LogitLensEntropy:
    def __init__(self, model, tokenizer, target_layer: int = -2, device: str = "cuda"):
        """
        Args:
            model: A decoder-only transformer model (e.g., LLaMA, GPT, ChatGLM).
            tokenizer: Corresponding tokenizer.
            target_layer: Which layer's hidden state to extract logits from (default: second to last).
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.target_layer = target_layer
        self.model.to(self.device)

        if not hasattr(model, "transformer") or not hasattr(model.transformer, "layers"):
            raise ValueError("Model must expose transformer.layers to extract intermediate hidden states.")

    def get_hidden_state(self, input_ids, attention_mask):
        """
        Forward pass to extract hidden states from the target layer.
        """
        all_hidden_states = []

        def hook(module, input, output):
            all_hidden_states.append(output)

        # register forward hook
        handle = self.model.transformer.layers[self.target_layer].register_forward_hook(hook)

        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        handle.remove()

        # output shape: (batch_size, seq_len, hidden_dim)
        return all_hidden_states[0]

    def quantify(self, input_str: str):
        """
        Compute the entropy over softmax(logits) from an intermediate layer.

        Args:
            input_str (str): Input prompt or code

        Returns:
            float: Entropy value from intermediate logits
        """
        self.model.eval()
        inputs = self.tokenizer(input_str, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Get hidden state from target layer
        hidden_state = self.get_hidden_state(input_ids, attention_mask)  # [1, seq_len, hidden_dim]
        last_token_state = hidden_state[:, -1, :]  # [1, hidden_dim]

        # Map to logits using model's final LM head
        if hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(last_token_state)  # [1, vocab_size]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "output_layer"):
            logits = self.model.transformer.output_layer(last_token_state)
        else:
            raise ValueError("Cannot find output projection layer to compute logits.")

        probs = F.softmax(logits, dim=-1)  # [1, vocab_size]
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # [1]
        return float(entropy.item())
