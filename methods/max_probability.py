import torch
import torch.nn.functional as F


def max_probability(model, text, model_type="discriminative", label_tokens=None):
    """
    Maximum Probability (Output-level)
    ----------------------------------
    Definition:
        MaxProb(p) = max_i p_i

    Measures how much probability mass the model assigns to its
    most confident class. Closely related to:
        - Least Confidence
        - Predictive Entropy
        - DeepGini

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model instance (e.g., CodeBERT, ChatGLM).
    text : str
        The input text or prompt.
    model_type : str, optional
        Model type: "discriminative" or "generative".
    label_tokens : list[str], optional
        Label tokens for generative models, e.g. ["Yes", "No"].

    Returns
    -------
    float
        Maximum predicted probability.
        Range: [1/C, 1] where C is the number of classes.

    Notes
    -----
    - For discriminative models, directly returns max softmax probability.
    - For generative models (classification-like tasks), computes the
      softmax over the final decision distribution (e.g., last token)
      and takes the maximum among label tokens.
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
            max_prob = torch.max(probs).item()

        return max_prob

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        if label_tokens is None:
            raise ValueError("label_tokens must be provided for generative models.")

        inputs = model.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1, :]  # final token logits
            probs = torch.softmax(last_logits, dim=-1)

            # Restrict to label tokens if specified
            label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
            label_probs = probs[label_ids]
            label_probs = label_probs / label_probs.sum()

            max_prob = torch.max(label_probs).item()

        return max_prob

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
