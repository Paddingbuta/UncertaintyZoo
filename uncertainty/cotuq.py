"""
avg_prediction_entropy.py

This script calculates the **average predictive entropy** for a given set of model predictions.

Usage:
    1. As a Python script:
        from avg_prediction_entropy import avg_prediction_entropy
        score = avg_prediction_entropy(probabilities)
        # `probabilities` should be a NumPy array or PyTorch tensor of shape (num_samples, num_classes)

    2. With batch data (e.g., from model output):
        - Ensure you pass softmax probabilities, not logits.
        - The function handles PyTorch tensors and NumPy arrays.

Returns:
    A float value representing the average predictive entropy across all input samples.

Requirements:
    - numpy
    - torch (if using PyTorch tensors)

Reference:
    Predictive entropy quantifies the uncertainty in the predicted class probabilities.
    Higher entropy indicates greater uncertainty.

Author: [Your Name]
"""


import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple


def generate_cot_responses(model, tokenizer, question: str, num_samples: int = 5, device: str = "cuda", max_new_tokens: int = 128):
    """
    Step 1: Generate multiple chain-of-thought (CoT) responses from the model.
    """
    prompt = f"{question.strip()}\nLet's think step by step:"
    model.to(device)
    model.eval()
    responses = []

    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
            )
        decoded = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        responses.append(decoded)

    return responses


def extract_keywords_with_scores(model, tokenizer, cot_response: str, device: str = "cuda") -> List[Tuple[str, float]]:
    """
    Step 2: Extract keywords from CoT using self-annotation prompt.

    Returns list of (keyword, importance_score) pairs.
    """
    prompt = (
        f"Please extract key concepts or phrases from the following reasoning with importance scores (1–10):\n"
        f"{cot_response}\n\n"
        f"Return them in format: concept: score"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=64,
            return_dict_in_generate=True
        )
    decoded = tokenizer.decode(output.sequences[0], skip_special_tokens=True)

    # Parse lines like: "buffer overflow: 9"
    keyword_scores = []
    for line in decoded.split("\n"):
        if ":" in line:
            try:
                key, score = line.strip().split(":")
                keyword_scores.append((key.strip(), float(score.strip())))
            except:
                continue

    return keyword_scores


def get_token_probability(model, tokenizer, prompt: str, token_str: str, device: str = "cuda") -> float:
    """
    Step 3: Compute the probability of the model generating `token_str` after `prompt`.
    """
    model.to(device)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]

    next_token_logits = logits[0, -1]  # last token logits
    probs = F.softmax(next_token_logits, dim=-1)

    token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token_str)[0])
    prob = probs[token_id].item()
    return prob


def cotuq_uncertainty(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    num_samples: int = 5,
    device: str = "cuda"
) -> float:
    """
    Main CoT-UQ Uncertainty Estimation Pipeline

    Returns:
        uncertainty score = 1 - confidence (confidence ∈ [0, 1])
    """
    cot_responses = generate_cot_responses(model, tokenizer, question, num_samples=num_samples, device=device)

    total_score = 0.0
    total_weight = 0.0

    for response in cot_responses:
        keywords = extract_keywords_with_scores(model, tokenizer, response, device=device)

        for kw, score in keywords:
            prob = get_token_probability(model, tokenizer, question, kw, device=device)
            weighted = prob * score
            total_score += weighted
            total_weight += score

    confidence = total_score / (total_weight + 1e-8)
    uncertainty = 1.0 - confidence
    return uncertainty
