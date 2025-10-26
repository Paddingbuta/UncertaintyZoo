import torch
import torch.nn.functional as F


def predictive_entropy(model, text, model_type="discriminative", label_tokens=None):
    """
    Predictive Entropy (Output-level)
    ---------------------------------
    Computes the entropy of the model’s output probability distribution:
        H(p) = - Σ_i p_i log(p_i)

    Used to measure the dispersion of probability mass across output classes.
    Higher values indicate greater uncertainty.

    Supports:
        - Discriminative models (e.g., CodeBERT)
        - Generative models (e.g., ChatGLM)

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The loaded model instance.
    text : str
        Input text or prompt (already constructed externally).
    model_type : str, optional
        "discriminative" or "generative".
    label_tokens : list[str], optional
        Label tokens for generative models, e.g. ["Yes", "No"].

    Returns
    -------
    float
        Predictive entropy value in the range [0, ln(C)].

    Notes
    -----
    - For discriminative models: apply softmax over class logits and compute entropy.
    - For generative models: first reduce the output to a class distribution
      (e.g., probabilities over "Yes"/"No") and then compute entropy.
      Do NOT compute over the full vocabulary.
    """
    if model_type == "discriminative":
        # ----------------------------------------
        # Discriminative models (e.g., CodeBERT)
        # ----------------------------------------
        inputs = model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            logits = model(**inputs).logits  # [1, num_labels]
            probs = torch.softmax(logits, dim=-1)[0]
            log_probs = torch.log(probs + 1e-12)
            entropy = -(probs * log_probs).sum().item()

        return entropy

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        if label_tokens is None:
            raise ValueError("label_tokens must be provided for generative models.")

        inputs = model.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the logits of the final token
            last_logits = outputs.logits[0, -1, :]  # [vocab_size]
            probs_full = torch.softmax(last_logits, dim=-1)

            # Reduce to class probabilities (Yes/No or multi-class)
            label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
            probs = probs_full[label_ids]
            probs = probs / probs.sum()

            log_probs = torch.log(probs + 1e-12)
            entropy = -(probs * log_probs).sum().item()

        return entropy

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
