import torch
import torch.nn.functional as F
import random
import re
from typing import List
from statistics import mean


def icl_sample(
    model,
    prompt: str,
    label_tokens: List[str],
    num_clarifications: int = 6,
    metric: str = "kl",
    intensity: float = 0.7,
):
    """
    Input Clarification Ensembles (ICE) — Generative Model Version
    --------------------------------------------------------------
    This method quantifies uncertainty by generating multiple
    *semantically equivalent but syntactically distinct* prompts
    (clarified versions) of the user's input and measuring how much
    the model's output distribution varies across them.

    Core Idea
    ----------
        Given an input x, create S rewritten variants {x̃_i}.
        Compute p_i = model(x̃_i) over label tokens.
        The ICE score is:
            ICE(x) = (1/S) Σ_i D(p_i || mean_p),
        where mean_p = (1/S) Σ_i p_i.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Generative model such as ChatGLM, GPT, or Llama.
    prompt : str
        The user-supplied prompt. It can describe any task.
    label_tokens : list[str]
        Tokens representing discrete output classes, e.g., ["Yes", "No"].
    num_clarifications : int, optional
        Number of paraphrased variants to generate (default: 6).
    metric : str, optional
        Divergence metric to measure variation: "kl" or "l2".
    intensity : float, optional
        Strength (0–1) controlling how much rewriting to apply (default: 0.7).

    Returns
    -------
    float
        ICE uncertainty score — higher means the model’s answers are
        more sensitive to wording variation.

    Notes
    -----
    - This version is specialized for generative models.
    - Rewriting operations include synonym swap, clause reordering,
      directive injection, and explicit reasoning expansion.
    - The goal is *semantic preservation with lexical diversity*.
    """

    # ----------------------------
    # Paraphrasing Layer
    # ----------------------------
    def clarify_prompt(base: str, n: int, intensity: float) -> List[str]:
        """
        Generate multiple clarified versions of a prompt.
        Applies a mix of linguistic transformations that retain
        meaning but vary phrasing, structure, and framing.
        """
        variants = []

        # Helper transformations
        def synonym_substitution(text):
            subs = {
                "explain": ["clarify", "describe", "elaborate"],
                "analyze": ["inspect", "examine", "evaluate"],
                "determine": ["decide", "assess", "judge"],
                "safe": ["secure", "harmless"],
                "vulnerable": ["unsafe", "exploitable"],
                "why": ["for what reason", "how come"],
                "please": ["kindly", "would you"],
                "what": ["which", "in what way"],
            }
            for k, v in subs.items():
                if random.random() < intensity and re.search(rf"\\b{k}\\b", text, re.IGNORECASE):
                    text = re.sub(rf"\\b{k}\\b", random.choice(v), text, flags=re.IGNORECASE)
            return text

        def sentence_reordering(text):
            # Reorder two clauses joined by comma or 'and'
            parts = re.split(r"(,| and )", text)
            if len(parts) > 3 and random.random() < intensity:
                head, mid, tail = parts[0], parts[1], "".join(parts[2:])
                text = tail.strip().capitalize() + " " + head.strip() + mid.strip() + "."
            return text

        def directive_injection(text):
            injects = [
                "Be precise and reason briefly.",
                "Answer directly without unnecessary elaboration.",
                "Carefully consider before responding.",
                "Think step by step and decide your final answer.",
                "Provide your reasoning concisely."
            ]
            if random.random() < intensity:
                if random.random() < 0.5:
                    text = f"{random.choice(injects)}\n{text}"
                else:
                    text = f"{text}\n{random.choice(injects)}"
            return text

        def question_reframing(text):
            q_forms = [
                "Could you determine {}?",
                "Please identify {}?",
                "In your judgment, {}?",
                "Would you say {}?",
                "Consider the question carefully: {}?"
            ]
            if random.random() < intensity:
                text = re.sub(r"^(.*?)([?.])?$", lambda m: random.choice(q_forms).format(m.group(1).strip()), text)
            return text

        for _ in range(n):
            temp = base
            ops = [synonym_substitution, sentence_reordering, directive_injection, question_reframing]
            random.shuffle(ops)
            for op in ops:
                if random.random() < intensity:
                    temp = op(temp)
            variants.append(temp)
        return variants

    # ----------------------------
    # Divergence metrics
    # ----------------------------
    def kl_divergence(p, q):
        """Kullback-Leibler divergence."""
        return torch.sum(p * torch.log((p + 1e-12) / (q + 1e-12)))

    def l2_distance(p, q):
        """Euclidean (L2) distance."""
        return torch.sqrt(torch.sum((p - q) ** 2))

    # ----------------------------
    # Probability extraction
    # ----------------------------
    def get_probs(text: str) -> torch.Tensor:
        """Return normalized probability distribution over label tokens."""
        inputs = model.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs_vocab = torch.softmax(logits, dim=-1)
            label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
            probs = probs_vocab[label_ids]
            probs = probs / probs.sum()
        return probs.cpu()

    # ----------------------------
    # Main Computation
    # ----------------------------
    clarified_prompts = clarify_prompt(prompt, num_clarifications, intensity)
    prob_list = [get_probs(p) for p in clarified_prompts]
    stacked = torch.stack(prob_list)
    mean_probs = stacked.mean(dim=0)

    divergences = []
    for p in prob_list:
        if metric == "kl":
            d = kl_divergence(p, mean_probs)
        elif metric == "l2":
            d = l2_distance(p, mean_probs)
        else:
            raise ValueError("metric must be 'kl' or 'l2'.")
        divergences.append(d.item())

    return float(sum(divergences) / len(divergences))
