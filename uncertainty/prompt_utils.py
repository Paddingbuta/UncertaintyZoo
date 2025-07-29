# uncertainty/methods/prompt_utils.py

import torch
from rouge_score import rouge_scorer
import re


def compute_rouge_l(str1: str, str2: str) -> float:
    """
    Compute ROUGE-L score between two strings.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(str1, str2)['rougeL'].fmeasure
    return score


def paraphrase_prompts(prompt: str, model, tokenizer, device="cpu", n: int = 5, max_length: int = 128) -> list:
    """
    Generate paraphrased versions of the input prompt using a generation model.
    """
    model.eval()
    input_prompt = f"Paraphrase: {prompt}"

    inputs = tokenizer([input_prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            max_length=max_length,
            num_return_sequences=n
        )

    paraphrases = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return list(set([p.strip() for p in paraphrases if p.strip().lower() != prompt.strip().lower()]))


def prompt_confidence_score(prompt: str, model, tokenizer, device="cpu", max_length: int = 128) -> float:
    """
    Prompt the model to generate a confidence score between 0 and 1.

    Format expected from model:
    "Confidence: 0.87"

    Returns:
        float: Confidence score (0-1), or None if not found.
    """
    # Format the prompt clearly to elicit a score
    scoring_prompt = (
        f"{prompt.strip()}\n\n"
        f"On a scale from 0 to 1, how confident are you in the above answer? Please reply with:\n"
        f"Confidence: <score between 0 and 1>"
    )

    inputs = tokenizer(scoring_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_length=max_length,
            num_return_sequences=1
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r"Confidence:\s*([0-1]\.\d+)", decoded)
    if match:
        return float(match.group(1))
    else:
        return None
