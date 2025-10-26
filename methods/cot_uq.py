import torch
import torch.nn.functional as F
from typing import List, Optional, Dict


def cot_uq(
    model,
    prompt: str,
    num_chains: int = 5,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    delimiter: Optional[str] = "Final Answer:",
    focus_span: str = "before_answer",  # "before_answer" | "full"
    label_tokens: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Chain-of-Thought Uncertainty (CoT-UQ) for Generative LMs
    --------------------------------------------------------
    Generates multiple chain-of-thought (CoT) samples and computes
    stepwise token entropies from per-step logits to quantify the
    stability of the reasoning process.

    For each chain i:
        - Decode with sampling to produce a CoT + (optional) final answer.
        - From the list of per-step logits (scores), compute per-token entropy:
            H_t = -sum_v p_t(v) * log p_t(v),  p_t = softmax(logits_t)
        - Aggregate per-chain metrics:
            * mean entropy over the CoT span
            * delta entropy: H_last - H_first over the CoT span

    If `label_tokens` is provided (classification-like prompting), we also compute
    inter-chain dispersion of the final-step class distribution (variance / MI).

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Causal LM compatible with `generate(return_dict_in_generate=True, output_scores=True)`.
        The caller must set `model.tokenizer` as the paired tokenizer.
    prompt : str
        User-provided prompt that elicits chain-of-thought followed by a final answer.
        If a delimiter is present in the generations, the tokens before its first
        occurrence are treated as the CoT span.
    num_chains : int, optional
        Number of independent CoT samples (default: 5).
    max_new_tokens : int, optional
        Max tokens to generate per chain (default: 128).
    temperature : float, optional
        Sampling temperature (default: 0.7).
    top_p : float, optional
        Nucleus sampling threshold (default: 0.95).
    top_k : int, optional
        Top-k sampling (default: 50).
    repetition_penalty : float, optional
        Repetition penalty for decoding (default: 1.0 → disabled).
    delimiter : str or None, optional
        If provided, the first occurrence in the decoded text splits CoT vs. answer.
        If None, the entire generation is treated as one span.
    focus_span : str, optional
        "before_answer": compute CoT metrics on tokens before delimiter;
        "full": use all generated tokens (CoT + answer).
    label_tokens : list[str], optional
        If provided, we compute final-step class distribution over these tokens and
        report ensemble dispersion (variance and MI).

    Returns
    -------
    dict
        {
          "cot_mean_entropy": float,
          "cot_delta_entropy_mean": float,
          "cot_delta_entropy_std": float,
          "final_prob_variance": float (if label_tokens given),
          "final_expected_entropy": float (if label_tokens given),
          "final_mutual_information": float (if label_tokens given)
        }

    Notes
    -----
    - Requires HF generate with `return_dict_in_generate=True` and `output_scores=True`
      so that `outputs.scores` is a list of logits per generated token.
    - To locate the CoT span, we map the `delimiter` occurrence in the decoded text
      back to a token index by incrementally decoding prefixes of generated ids.
    - If the delimiter is not found, we treat either the whole generation or the
      desired `focus_span` policy.
    - This function does not leak any private chain-of-thought in outputs; it only
      uses logits to compute entropy statistics.
    """

    device = next(model.parameters()).device
    tokenizer = model.tokenizer

    def decode_ids(ids: torch.Tensor) -> str:
        """Decode a tensor of token ids to string."""
        return tokenizer.decode(ids.tolist(), skip_special_tokens=True)

    def map_delim_to_token_index(gen_ids: torch.Tensor, delim: str) -> int:
        """
        Find the token index where the delimiter *starts* in the decoded text.
        We decode prefixes progressively until the delimiter becomes visible.
        Returns an integer in [0, len(gen_ids)] indicating the split index,
        or len(gen_ids) if not found.
        """
        if not delim:
            return len(gen_ids)
        running_text = ""
        for t in range(1, len(gen_ids) + 1):
            running_text = decode_ids(gen_ids[:t])
            if delim in running_text:
                # Backtrack to the earliest token where the substring appears
                # We refine by shrinking until just before the delimiter vanishes
                low, high = 0, t
                pos = running_text.find(delim)
                # Binary-like shrink to approximate the token boundary
                while low < high:
                    mid = (low + high) // 2
                    if delim in decode_ids(gen_ids[:mid]):
                        high = mid
                    else:
                        low = mid + 1
                return low
        return len(gen_ids)

    def entropy_from_logits(logits: torch.Tensor) -> float:
        """Compute Shannon entropy from unnormalized logits (1D)."""
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * torch.log(probs + 1e-12)).sum().item()
        return ent

    def class_probs_from_logits(logits: torch.Tensor, labels: List[str]) -> torch.Tensor:
        """Map final-step logits to a normalized class distribution over label tokens."""
        label_ids = [tokenizer.convert_tokens_to_ids(t) for t in labels]
        probs = torch.softmax(logits, dim=-1)[label_ids]
        probs = probs / probs.sum()
        return probs

    # Containers for per-chain metrics
    per_chain_mean_entropy = []
    per_chain_delta_entropy = []
    final_class_probs = []  # only if label_tokens given

    # Prepare constant input ids
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    for _ in range(num_chains):
        # Decode one CoT sample with scores for each generated token
        gen_out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # scores: List[Tensor[vocab_size]], length == generated_length
        scores: List[torch.Tensor] = gen_out.scores or []
        if len(scores) == 0:
            # If scores are unavailable, skip this chain
            continue

        # Extract generated ids only (exclude the prompt part)
        # gen_out.sequences: [1, prompt_len + gen_len]
        full_seq = gen_out.sequences[0]
        gen_ids = full_seq[len(enc["input_ids"][0]):]  # [gen_len]
        gen_len = gen_ids.size(0)

        # Identify CoT span by delimiter mapping (if requested)
        if delimiter and focus_span == "before_answer":
            split_idx = map_delim_to_token_index(gen_ids, delimiter)
            span_scores = scores[:split_idx]  # up to (but not including) answer
        else:
            span_scores = scores  # use all generated steps

        # Compute per-token entropies over the chosen span
        token_entropies = [entropy_from_logits(s[0].to(device)) for s in span_scores]  # s[0]: [vocab]
        if len(token_entropies) == 0:
            # No CoT tokens (e.g., delimiter appears immediately). Skip.
            continue

        # Chain-level mean entropy and delta entropy
        H_mean = float(sum(token_entropies) / len(token_entropies))
        H_delta = float(token_entropies[-1] - token_entropies[0])

        per_chain_mean_entropy.append(H_mean)
        per_chain_delta_entropy.append(H_delta)

        # Optional: final-step class distribution (for ensemble dispersion)
        if label_tokens is not None and len(scores) > 0:
            last_logits = scores[-1][0].to(device)  # [vocab]
            probs = class_probs_from_logits(last_logits, label_tokens).cpu()
            final_class_probs.append(probs)

    # Aggregate CoT metrics
    if len(per_chain_mean_entropy) == 0:
        # No valid chains produced (e.g., model did not return scores).
        return {
            "cot_mean_entropy": 0.0,
            "cot_delta_entropy_mean": 0.0,
            "cot_delta_entropy_std": 0.0,
        }

    cot_mean_entropy = float(sum(per_chain_mean_entropy) / len(per_chain_mean_entropy))

    # Mean / std of entropy deltas across chains
    mu = sum(per_chain_delta_entropy) / len(per_chain_delta_entropy)
    var = sum((d - mu) ** 2 for d in per_chain_delta_entropy) / max(1, (len(per_chain_delta_entropy) - 1))
    cot_delta_entropy_mean = float(mu)
    cot_delta_entropy_std = float(var ** 0.5)

    results = {
        "cot_mean_entropy": cot_mean_entropy,
        "cot_delta_entropy_mean": cot_delta_entropy_mean,
        "cot_delta_entropy_std": cot_delta_entropy_std,
    }

    # If class labels provided, compute ensemble dispersion of final prediction
    if len(final_class_probs) > 0:
        P = torch.stack(final_class_probs)  # [S, C]
        mean_p = P.mean(dim=0)
        # Class-wise variance averaged over classes
        final_var = torch.var(P, dim=0).mean().item()
        # Expected entropy (average entropy over chains)
        H_exp = float((-(P * (P + 1e-12).log()).sum(dim=1)).mean().item())
        # Mutual Information (BALD-style): H(mean_p) - E[H(p)]
        H_mean = float((-(mean_p * (mean_p + 1e-12).log()).sum()).item())
        MI = H_mean - H_exp

        results.update(
            final_prob_variance=float(final_var),
            final_expected_entropy=float(H_exp),
            final_mutual_information=float(MI),
        )

    return results
