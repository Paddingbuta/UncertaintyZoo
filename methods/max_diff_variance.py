import torch


def max_diff_variance(model, text, model_type="discriminative",
                      label_tokens=None, num_samples=10, agg="mean", dropout=True):
    """
    Maximum Difference Variance (Output-Level, Ensemble-Based)
    -----------------------------------------------------------
    Measures the amplitude of probability fluctuation across stochastic
    forward passes by computing the *range* (max–min) for each class.
    Compared with variance, this metric is more interpretable and
    sensitive to outlier samples.

    Definition
    ----------
        Δ_c = max_s p_c^(s) - min_s p_c^(s)
        MDV_max  = max_c Δ_c
        MDV_mean = (1 / C) * Σ_c Δ_c

    Range
    -----
        [0, 1]
        - Approaches 0 when all stochastic predictions are identical.
        - Approaches 1 if some forward pass assigns near-1 probability
          to a class while another assigns near-0.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model instance (e.g., CodeBERT or ChatGLM).
    text : str
        Input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".
    label_tokens : list[str], optional
        For generative models, list of label tokens (e.g., ["Yes", "No"]).
    num_samples : int, optional
        Number of stochastic forward passes (default: 10).
    agg : str, optional
        Aggregation method over classes: "mean" or "max".
    dropout : bool, optional
        Whether to enable dropout during sampling.

    Returns
    -------
    float
        Maximum Difference Variance (MDV) score.

    Notes
    -----
    - Discriminative models: sample multiple softmax outputs with dropout.
    - Generative models: map final-step token probabilities to label tokens.
    - The range metric is sensitive to the number of samples; larger S
      increases the chance of observing extreme values.
    """
    sampled_probs = []

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

        for _ in range(num_samples):
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                sampled_probs.append(probs.unsqueeze(0))

        sampled_probs = torch.cat(sampled_probs, dim=0)  # [S, C]
        diffs = sampled_probs.max(dim=0).values - sampled_probs.min(dim=0).values

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        if label_tokens is None:
            raise ValueError("label_tokens must be provided for generative models.")

        model.train(dropout)
        inputs = model.tokenizer(text, return_tensors='pt')

        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]  # [vocab_size]
                probs = torch.softmax(last_logits, dim=-1)

                label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
                class_probs = probs[label_ids]
                class_probs = class_probs / class_probs.sum()
                sampled_probs.append(class_probs.unsqueeze(0))

        sampled_probs = torch.cat(sampled_probs, dim=0)  # [S, C]
        diffs = sampled_probs.max(dim=0).values - sampled_probs.min(dim=0).values

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if agg == "mean":
        mdv = diffs.mean().item()
    elif agg == "max":
        mdv = diffs.max().item()
    else:
        raise ValueError("agg must be 'mean' or 'max'")

    return mdv
