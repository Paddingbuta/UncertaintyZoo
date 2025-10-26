import torch
import torch.nn.functional as F


def least_confidence(model, text, model_type="discriminative", label_tokens=None):
    """
    Least Confidence (LC)
    ---------------------
    Definition:
        LC(p) = 1 - max_i p_i

    Measures how far the model's prediction is from complete confidence.
    Higher LC values indicate higher uncertainty.

    Properties
    ----------
    - Range: [0, 1 - 1/C], where C is the number of classes.
    - For binary classification: LC = min(p, 1 - p).

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The loaded model instance (e.g., CodeBERT or ChatGLM).
    text : str
        Input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".
    label_tokens : list[str], optional
        Only required for generative models (e.g., ["Yes", "No"]).

    Returns
    -------
    float
        Least Confidence score. Larger values correspond to higher uncertainty.

    Notes
    -----
    - Discriminative models:
        LC = 1 - max(probs)
    - Generative models:
        LC = 1 - max(p_label_tokens)
      If the task is classification-like (e.g., Devign),
      apply this to the final token distribution over the label tokens.
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
            lc = 1 - torch.max(probs).item()

        return lc

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        if label_tokens is None:
            raise ValueError("label_tokens must be provided for generative models.")

        inputs = model.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the logits of the final token
            last_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)

            # Extract probabilities of label tokens
            label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
            probs = probs[label_ids]
            probs = probs / probs.sum()

            lc = 1 - torch.max(probs).item()

        return lc

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
