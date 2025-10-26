import torch


def deepgini(model, text, model_type="discriminative", label_tokens=None):
    """
    DeepGini (Output-Level)
    -----------------------
    Gini-based uncertainty score measuring the "uniformity" of the predicted
    probability distribution.

    Definition
    ----------
        DeepGini(p) = 1 - Σ_i p_i^2

    Range
    -----
        [0, 1 - 1/C], where C is the number of classes.
        - Approaches 0 when the model is certain (one class dominates).
        - Increases as probabilities become more uniform.

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

    Returns
    -------
    float
        DeepGini uncertainty score.

    Notes
    -----
    - For discriminative models: computed directly from class probabilities.
    - For generative models: probabilities are taken over label tokens at
      the final decoding step (classification-style tasks such as Devign).
    - This metric is cheaper to compute than entropy and emphasizes
      head-class differences due to the square term.
    """
    if model_type == "discriminative":
        # ----------------------------------------
        # Discriminative models (e.g., CodeBERT)
        # ----------------------------------------
        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            logits = model(**inputs).logits  # [1, num_labels]
            probs = torch.softmax(logits, dim=-1)[0]
            score = 1.0 - torch.sum(probs ** 2).item()

        return score

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        if label_tokens is None:
            raise ValueError("label_tokens must be provided for generative models.")

        inputs = model.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1, :]  # [vocab_size]
            probs = torch.softmax(last_logits, dim=-1)

            label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
            class_probs = probs[label_ids]
            class_probs = class_probs / class_probs.sum()

            score = 1.0 - torch.sum(class_probs ** 2).item()

        return score

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
