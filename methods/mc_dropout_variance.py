import torch


def mc_dropout_variance(model, text, model_type="discriminative",
                        label_tokens=None, num_samples=10, dropout=True):
    """
    Monte Carlo Dropout Variance (Output-Level, Ensemble-Based)
    ------------------------------------------------------------
    Measures the disagreement between multiple stochastic forward passes
    (e.g., dropout-enabled runs) using the variance of predicted class
    probabilities. Reflects epistemic uncertainty (model uncertainty).

    Definition
    ----------
        Var_MC = (1 / C) * Σ_c Var_s[p_c^(s)]

        or equivalently, the sum over classes without normalization
        (the two differ only by a scaling factor).

    Range
    -----
        [0, upper], where the upper bound depends on the number of samples S
        and the distribution shape. Larger variance indicates stronger
        inconsistency among stochastic predictions.

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
    dropout : bool, optional
        Whether to enable dropout during sampling.

    Returns
    -------
    float
        Monte Carlo dropout variance score.

    Notes
    -----
    - Discriminative models:
        Compute variance across S softmax outputs for each class, then
        average over classes.
    - Generative models (classification-style prompt):
        Compute variance across the label-token probabilities only.
    - Sensitive to scale: larger S stabilizes the estimate.
    """
    probs_list = []

    if model_type == "discriminative":
        # ----------------------------------------
        # Discriminative models (e.g., CodeBERT)
        # ----------------------------------------
        model.train(dropout)  # Enable dropout
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
                probs_list.append(probs.unsqueeze(0))

        stacked = torch.cat(probs_list, dim=0)  # [S, C]
        var_per_class = torch.var(stacked, dim=0)
        score = var_per_class.mean().item()

        return score

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
                probs_list.append(class_probs.unsqueeze(0))

        stacked = torch.cat(probs_list, dim=0)  # [S, C]
        var_per_class = torch.var(stacked, dim=0)
        score = var_per_class.mean().item()

        return score

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
