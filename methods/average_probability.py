import torch
import torch.nn.functional as F


def average_probability(model, text, model_type="discriminative",
                        label_tokens=None, target_label=None):
    """
    Average Probability (AvgProb)
    -----------------------------
    Implementation for the Devign binary classification task.

    Supports both:
        - Discriminative models (e.g., CodeBERT)
        - Generative models (e.g., ChatGLM)

    Definition
    ----------
    AvgProb(x, y₁:ₜ) = (1/T) Σₜ p_θ(yₜ | y_{<t}, x)

    For single-label classification (T = 1):
        AvgProb = p_θ(y* | x)

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model instance.
    text : str
        Input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".
    label_tokens : list[str], optional
        Label tokens for generative models, e.g. ["Yes", "No"].
    target_label : int | str
        - For discriminative models: integer index of the true class (0/1).
        - For generative models: target label token (e.g., "Yes" or "No").

    Returns
    -------
    float
        Average probability assigned to the ground-truth label(s).
        Range: [0, 1]. Higher is better.

    Notes
    -----
    - For discriminative models, this is the probability of the true class.
    - For generative models, it averages the target token probability.
      In classification-like tasks such as Devign, only the final token’s
      probability is used (T = 1).
    - Monotonically related to ANLL; higher AvgProb implies lower ANLL.
    """
    if model_type == "discriminative":
        # ----------------------------------------
        # Discriminative models (e.g., CodeBERT)
        # ----------------------------------------
        if target_label is None:
            raise ValueError("target_label (int) must be provided for discriminative models.")

        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            logits = model(**inputs).logits  # [1, num_labels]
            probs = torch.softmax(logits, dim=-1)
            avg_prob = probs[0, target_label].item()

        return avg_prob

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        if label_tokens is None or target_label is None:
            raise ValueError("label_tokens and target_label must be provided for generative models.")

        inputs = model.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the logits of the final token
            last_logits = outputs.logits[0, -1, :]  # [vocab_size]
            probs = torch.softmax(last_logits, dim=-1)

            # Convert target label token to ID
            target_id = model.tokenizer.convert_tokens_to_ids(target_label)
            avg_prob = probs[target_id].item()

        return avg_prob

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
