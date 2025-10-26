import torch


def min_variance(model, text, model_type="discriminative",
                 label_tokens=None, num_samples=10, top_k=None, dropout=True):
    """
    Minimum Variance (Output-Level, Ensemble-Based)
    -----------------------------------------------
    A conservative consistency-based uncertainty metric. It measures the
    smallest variance among class probabilities across multiple stochastic
    forward passes.

    Definition
    ----------
        MinVar = min_c Var_s[p_c^(s)]

    Interpretation
    --------------
    - Small value → at least one class remains stable across all samples,
      implying the model has a consistent belief in that class.
    - Large value → even the most stable class fluctuates, indicating
      overall uncertainty or model inconsistency.

    Range
    -----
        [0, upper], where "upper" depends on the scale of the probabilities.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Model instance (e.g., CodeBERT or ChatGLM).
    text : str
        Input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".
    label_tokens : list[str], optional
        For generative models, list of label tokens (e.g., ["Yes", "No"]).
    num_samples : int, optional
        Number of stochastic forward passes (default: 10).
    top_k : int, optional
        Only compute the minimum over the top-K most probable classes
        (helps avoid bias from consistently low-probability classes).
    dropout : bool, optional
        Whether to enable dropout for sampling.

    Returns
    -------
    float
        Minimum variance across class probabilities.

    Notes
    -----
    - This metric is sensitive to classes that always have near-zero
      probability (which can trivially produce variance ≈ 0). The top-K
      filtering mitigates this issue.
    - Recommended to use jointly with average or maximum variance metrics.
    """
    if model_type == "discriminative":
        # ----------------------------------------
        # Discriminative models (e.g., CodeBERT)
        # ----------------------------------------
        model.train(dropout)
        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        all_probs = []
        for _ in range(num_samples):
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                all_probs.append(probs)

        all_probs = torch.stack(all_probs)  # [S, C]
        var_per_class = torch.var(all_probs, dim=0)  # [C]

        if top_k is not None and top_k < len(var_per_class):
            top_indices = torch.topk(all_probs.mean(dim=0), k=top_k).indices
            var_per_class = var_per_class[top_indices]

        score = torch.min(var_per_class).item()
        return score

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        if label_tokens is None:
            raise ValueError("label_tokens must be provided for generative models.")

        model.train(dropout)
        inputs = model.tokenizer(text, return_tensors='pt')

        all_probs = []
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]  # [vocab_size]
                probs = torch.softmax(last_logits, dim=-1)

                label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
                class_probs = probs[label_ids]
                class_probs = class_probs / class_probs.sum()
                all_probs.append(class_probs)

        all_probs = torch.stack(all_probs)  # [S, C]
        var_per_class = torch.var(all_probs, dim=0)  # [C]

        if top_k is not None and top_k < len(var_per_class):
            top_indices = torch.topk(all_probs.mean(dim=0), k=top_k).indices
            var_per_class = var_per_class[top_indices]

        score = torch.min(var_per_class).item()
        return score

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
