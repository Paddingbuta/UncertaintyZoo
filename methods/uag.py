import torch
import torch.nn.functional as F
from typing import List, Optional


def uag(
    model,
    text: str,
    model_type: str = "generative",
    label_tokens: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    aggregate: str = "mean",    # "mean" | "sum"
    normalize: bool = True,     # normalize by heads and sequence size
) -> float:
    """
    Uncertainty-Aware Attention Gradients (UAG)
    -------------------------------------------
    Measures gradient-based sensitivity of the attention mechanism.
    For each layer l, compute the Frobenius norm of the element-wise
    product between attention A^(l) and the gradient d(target)/dA^(l):
        U^(l) = || ( dY/dA^(l) ⊙ A^(l) ) ||_F
    Then aggregate across layers.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Discriminative (e.g., CodeBERT) or Generative (e.g., ChatGLM/LLaMA) model.
        The model must support `output_attentions=True`.
    text : str
        Input text or prompt.
    model_type : str, optional
        "generative" or "discriminative".
    label_tokens : list[str], optional
        For generative classification-style prompting: restrict the target
        to these label tokens at the final step. If None, use the max-logit
        token at the final step.
    layers : list[int], optional
        Subset of layer indices to include. If None, include all layers.
    aggregate : str, optional
        How to combine layer-wise scores: "mean" (default) or "sum".
    normalize : bool, optional
        If True, divide each layer's score by sqrt(H * S * S) to make
        values roughly length/head invariant (H=heads, S=seq_len).

    Returns
    -------
    float
        UAG score (higher ⇒ attention is more sensitivity-prone ⇒ higher uncertainty).

    Notes
    -----
    - Runs one differentiable forward pass with `output_attentions=True`.
    - Attentions returned by HF models are differentiable tensors (no detach).
    - We retain grads on attention tensors and backprop from a scalar target:
        * Generative: final-step logit of the strongest label (or max over vocab).
        * Discriminative: maximum class logit.
    - If your model does not support attentions, this function raises a ValueError.
    """

    # Ensure gradient mode
    model.eval()

    # -------- Forward with attentions --------
    if model_type == "generative":
        inputs = model.tokenizer(text, return_tensors="pt")
        # Return attentions and keep graph for gradients
        outputs = model(**inputs, output_attentions=True, use_cache=False)
        if outputs.attentions is None:
            raise ValueError("Model did not return attentions. Enable `output_attentions=True`.")
        # logits: [1, seq_len, vocab_size]
        logits = outputs.logits[0]
        last_logits = logits[-1]  # [vocab_size]

        # Choose scalar target Y for backprop:
        # If label_tokens provided, focus on those label ids; else use max logit over vocab.
        if label_tokens:
            label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
            label_vec = last_logits[label_ids]  # [C]
            # Use the top label's logit as scalar target (stable and simple)
            target = label_vec.max()
        else:
            target = last_logits.max()

    elif model_type == "discriminative":
        inputs = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs, output_attentions=True)
        if outputs.attentions is None:
            raise ValueError("Model did not return attentions. Enable `output_attentions=True`.")
        # logits: [1, num_labels]
        logits = outputs.logits[0]  # [C]
        target = logits.max()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Collect attentions and make sure grads are retained
    # attentions: tuple of length L with tensors [batch=1, heads, S, S]
    attentions = list(outputs.attentions)
    for a in attentions:
        if a.requires_grad:
            a.retain_grad()

    # -------- Backprop to obtain dY/dA^(l) --------
    model.zero_grad(set_to_none=True)
    target.backward(retain_graph=False)

    # -------- Compute layer-wise U^(l) --------
    layer_scores = []
    L = len(attentions)
    idxs = list(range(L)) if layers is None else layers

    for l in idxs:
        A = attentions[l]                 # [1, H, S, S]
        G = A.grad                        # [1, H, S, S] (may be None if not connected)
        if G is None:
            # If gradient is None (unconnected), treat its contribution as zero
            continue

        # Element-wise product followed by Frobenius norm
        # U^(l) = || (G ⊙ A) ||_F
        GA = (G * A).abs()                # [1, H, S, S]
        # Frobenius norm over all dims
        U_l = torch.norm(GA, p="fro")

        if normalize:
            # Normalize by sqrt(H * S * S) to reduce scale sensitivity
            _, H, S, _ = GA.shape
            denom = (H * S * S) ** 0.5
            U_l = U_l / max(denom, 1.0)

        layer_scores.append(U_l.item())

    if not layer_scores:
        # If nothing contributed, return 0.0 to be safe
        return 0.0

    if aggregate == "sum":
        total = sum(layer_scores)
    else:
        total = sum(layer_scores) / len(layer_scores)

    return float(total)
