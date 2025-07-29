# input_clarification_ensembles.py

import torch
import torch.nn.functional as F
import numpy as np


def input_clarification_ensembles(model, tokenizer, inputs, clarification_pool, num_samples=5, device='cuda'):
    """
    Args:
        model: Transformer-based model (e.g., CodeBERT)
        tokenizer: Tokenizer corresponding to the model
        inputs: List[str], original input texts
        clarification_pool: Dict[int, List[str]], maps index of each input to a list of clarification variants
        num_samples: int, how many clarified variants to sample per input
        device: str, device for model inference

    Returns:
        uncertainty_scores: List[float], entropy-based uncertainty scores
    """
    model.eval()
    model.to(device)

    uncertainty_scores = []

    for idx, original_input in enumerate(inputs):
        variants = clarification_pool.get(idx, [])
        sampled_variants = variants[:num_samples] if len(variants) >= num_samples else variants
        if not sampled_variants:
            # fallback: use original input
            sampled_variants = [original_input]

        logits_list = []

        for variant in sampled_variants:
            encoded = tokenizer(variant, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                output = model(**encoded)
                logits = output.logits
                logits_list.append(logits.cpu())

        stacked_logits = torch.stack(logits_list)  # (N, B=1, C)
        probs = F.softmax(stacked_logits, dim=-1).squeeze(1)  # (N, C)

        avg_probs = probs.mean(dim=0)  # (C,)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-12)).item()
        uncertainty_scores.append(entropy)

    return uncertainty_scores
