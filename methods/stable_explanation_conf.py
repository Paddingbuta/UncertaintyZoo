import torch
import torch.nn.functional as F
from itertools import combinations
from typing import List, Optional, Dict, Literal


def _pairwise_cosine_mean(vectors: List[torch.Tensor]) -> float:
    """Mean pairwise cosine similarity for a list of 1-D tensors."""
    if len(vectors) < 2:
        return 1.0
    sims = []
    for i, j in combinations(range(len(vectors)), 2):
        vi = vectors[i].unsqueeze(0)
        vj = vectors[j].unsqueeze(0)
        sim = F.cosine_similarity(vi, vj, dim=-1).item()
        sims.append(sim)
    return float(sum(sims) / len(sims))


def _majority_agreement_rate(labels: List[int]) -> float:
    """Proportion of the majority class among categorical labels."""
    if not labels:
        return 0.0
    from collections import Counter
    cnt = Counter(labels)
    return max(cnt.values()) / len(labels)


def _mean_pool_embeddings(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean-pool last hidden states to a sentence embedding.
    hidden_states: [1, seq_len, hidden_dim]
    attention_mask: [1, seq_len]
    """
    if attention_mask is None:
        return hidden_states.mean(dim=1).squeeze(0)
    mask = attention_mask.unsqueeze(-1).float()  # [1, S, 1]
    summed = (hidden_states * mask).sum(dim=1)   # [1, H]
    denom = mask.sum(dim=1).clamp(min=1.0)       # [1, 1]
    return (summed / denom).squeeze(0)           # [H]


def stable_explanation_conf(
    model,
    text: str,
    model_type: Literal["generative", "discriminative"] = "generative",
    # For generative models
    num_samples: int = 6,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    delimiter: Optional[str] = None,  # e.g., "Final Answer:" if your prompt uses one
    label_tokens: Optional[List[str]] = None,  # e.g., ["Yes", "No"] for classification-like prompting
    explanation_max_tokens: int = 1600,  # hard cap when re-embedding explanations
    # For discriminative models
    disc_dropout: bool = True,
    disc_max_length: int = 512,
    # Fusion weights
    gamma_label: float = 0.5,
    gamma_text: float = 0.5,
) -> Dict[str, float]:
    """
    Stable Explanation Confidence (SEC)
    -----------------------------------
    Computes a stability-based confidence score by combining:
      (1) label stability across multiple runs
      (2) explanation similarity across multiple runs

    Generative branch:
      - Sample S generations with sampling (temperature/top-p/top-k).
      - Extract explanation text and conclusion per sample.
      - Encode explanations to embeddings (mean-pooled hidden states).
      - C_text: mean pairwise cosine similarity of explanation embeddings.
      - C_label: majority agreement rate over predicted labels.
      - SEC = gamma_label * C_label + gamma_text * C_text.

    Discriminative branch:
      - Enable dropout, run S stochastic forwards with attentions.
      - Treat attention-based attribution vector as an "explanation":
          * mean over heads of attention weights from CLS-to-tokens (or tokens-to-CLS).
        Compare explanations via cosine; compute label agreement like above.

    Returns
    -------
    dict with:
      {
        "C_label": float in [0,1],
        "C_text":  float in [-1,1] (typically [0,1]),
        "SEC":     float in [0,1] (clipped)
      }
    """
    model.eval()
    device = next(model.parameters()).device

    # ---------------------------
    # Generative models
    # ---------------------------
    if model_type == "generative":
        tokenizer = model.tokenizer
        enc = tokenizer(text, return_tensors="pt").to(device)

        def decode_ids(ids: torch.Tensor) -> str:
            return tokenizer.decode(ids.tolist(), skip_special_tokens=True)

        def extract_explanation_and_label(decoded: str):
            """
            Split explanation vs. conclusion from decoded text.
            If no delimiter is provided, treat the whole generation as explanation,
            and infer label by scanning label tokens (if provided).
            """
            exp_text = decoded
            label_idx = None

            if delimiter and delimiter in decoded:
                parts = decoded.split(delimiter, 1)
                exp_text = parts[0].strip()
                tail = parts[1].strip()
                # Infer label by first matching of label_tokens in the tail
                if label_tokens:
                    low_tail = tail.lower()
                    lt_lower = [lt.lower() for lt in label_tokens]
                    for i, tok in enumerate(lt_lower):
                        if tok in low_tail:
                            label_idx = i
                            break

            # If label not found by text, fall back to None; we will use logits later if possible
            return exp_text, label_idx

        def embed_text(sent: str) -> torch.Tensor:
            """Encode explanation text with the same LM to a sentence embedding."""
            emb_inputs = tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=explanation_max_tokens
            ).to(device)
            with torch.no_grad():
                out = model(**emb_inputs, output_hidden_states=True, use_cache=False)
            last = out.hidden_states[-1]  # [1, S, H]
            return _mean_pool_embeddings(last, emb_inputs.get("attention_mask", None)).detach().cpu()

        exp_embeddings: List[torch.Tensor] = []
        labels: List[int] = []

        # We will also collect logits for the final step to infer label if needed
        for _ in range(num_samples):
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
            full_ids = gen_out.sequences[0]  # [prompt_len + gen_len]
            prompt_len = enc["input_ids"].shape[1]
            gen_ids = full_ids[prompt_len:]  # generated part
            decoded = decode_ids(gen_ids)

            exp_text, label_idx = extract_explanation_and_label(decoded)
            exp_embeddings.append(embed_text(exp_text))

            # If we failed to parse label from text but label_tokens are provided, use last-step logits
            if label_idx is None and label_tokens and gen_out.scores:
                last_logits = gen_out.scores[-1][0].to(device)  # [vocab]
                label_ids = [tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
                probs = torch.softmax(last_logits[label_ids], dim=-1)
                label_idx = int(torch.argmax(probs).item())

            if label_idx is not None:
                labels.append(label_idx)

        # Text similarity
        C_text = _pairwise_cosine_mean(exp_embeddings)
        # Label stability (if no labels could be inferred at all, set 0)
        C_label = _majority_agreement_rate(labels) if len(labels) > 0 else 0.0

        SEC = gamma_label * C_label + gamma_text * max(0.0, C_text)  # clip cosine to [0,1] part
        SEC = float(max(0.0, min(1.0, SEC)))  # clamp to [0,1]

        return {"C_label": float(C_label), "C_text": float(C_text), "SEC": SEC}

    # ---------------------------
    # Discriminative models
    # ---------------------------
    elif model_type == "discriminative":
        tokenizer = model.tokenizer
        if disc_dropout:
            model.train(True)  # enable dropout for stochasticity
        else:
            model.eval()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=disc_max_length).to(device)

        expl_vectors: List[torch.Tensor] = []
        labels: List[int] = []

        for _ in range(num_samples):
            with torch.no_grad():
                out = model(**inputs, output_attentions=True)
            logits = out.logits[0]            # [C]
            probs = torch.softmax(logits, dim=-1)
            labels.append(int(torch.argmax(probs).item()))

            # Build an attention-based explanation vector:
            # mean over heads of attentions from the last encoder layer,
            # aggregate attention on tokens as importance weights
            if out.attentions is None or len(out.attentions) == 0:
                # Fall back to last hidden-state mean-pooled embedding
                with torch.no_grad():
                    out2 = model(**inputs, output_hidden_states=True)
                last = out2.hidden_states[-1]  # [1, S, H]
                emb = _mean_pool_embeddings(last, inputs.get("attention_mask", None)).detach().cpu()
                expl_vectors.append(emb)
            else:
                # attentions is a tuple of length L: each [1, heads, S, S]
                A_last = out.attentions[-1][0]            # [heads, S, S]
                A_mean = A_last.mean(dim=0)               # [S, S]
                # Importance per token = attention-to-CLS or CLS-to-token; choose CLS-to-token row 0
                imp = A_mean[0]                           # [S]
                # Normalize to unit norm
                imp = imp / (imp.norm(p=2) + 1e-12)
                expl_vectors.append(imp.detach().cpu())

        C_text = _pairwise_cosine_mean(expl_vectors)
        C_label = _majority_agreement_rate(labels)

        SEC = gamma_label * C_label + gamma_text * max(0.0, C_text)
        SEC = float(max(0.0, min(1.0, SEC)))

        return {"C_label": float(C_label), "C_text": float(C_text), "SEC": SEC}

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
