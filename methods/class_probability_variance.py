import torch


def class_probability_variance(model, text, model_type="discriminative",
                               label_tokens=None, num_samples=10, dropout=True):
    """
    Class Probability Variance (Output-Level, Ensemble-Based)
    ---------------------------------------------------------
    Measures how much the predicted probability vector fluctuates
    across multiple stochastic forward passes. Unlike class
    prediction variance (which looks only at the voted label),
    this directly examines the variance of the *entire probability
    distribution* for each class.

    Definition
    ----------
        CPV = (1 / C) * Σ_c Var_s[p_c^(s)]

    where Var_s denotes the variance across S stochastic samples.

    Range
    -----
        [0, 1], depending on how unstable the predicted
        probabilities are across different stochastic passes.
        - 0 means perfectly stable predictions.
        - Larger values indicate higher epistemic uncertainty.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model instance (e.g., CodeBERT or ChatGLM).
    text : str
        Input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".
    label_tokens : list[str], optional
        For generative models, list of label tokens
        (e.g., ["Yes", "No"] for Devign classification).
    num_samples : int, optional
        Number of stochastic forward passes (default: 10).
    dropout : bool, optional
        Whether to enable dropout during sampling.

    Returns
    -------
    float
        Average per-class probability variance (CPV).

    Notes
    -----
    - Discriminative models: run multiple stochastic passes (dropout)
      and compute per-class probability variance.
    - Generative models: use the final-step class probabilities mapped
      to label tokens.
    - This metric operates strictly in the *probability domain*.
      (If you prefer logits-domain variance, see method 3.)
    """
    all_probs = []

    if model_type == "discriminative":
        # ----------------------------------------
        # Discriminative models (e.g., CodeBERT)
        # ----------------------------------------
        model.train(dropout)  # enable dropout layers
        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        for _ in range(num_samples):
            with torch.no_grad():
                logits = model(**inputs).logits  # [1, num_labels]
                probs = torch.softmax(logits, dim=-1)[0]
                all_probs.append(probs.unsqueeze(0))  # [1, C]

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
                all_probs.append(class_probs.unsqueeze(0))  # [1, C]

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ----------------------------------------
    # Compute variance across stochastic passes
    # ----------------------------------------
    all_probs = torch.cat(all_probs, dim=0)  # [S, C]
    var_per_class = torch.var(all_probs, dim=0, unbiased=False)  # [C]
    cpv = torch.mean(var_per_class).item()

    return cpv
