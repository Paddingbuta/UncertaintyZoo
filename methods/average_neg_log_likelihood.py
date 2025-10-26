import torch
import torch.nn.functional as F


def average_neg_log_likelihood(model, text, model_type="discriminative",
                               label_tokens=None, target_label=None):
    """
    Average Negative Log-Likelihood (ANLL)
    --------------------------------------
    Implementation for the Devign binary classification task.

    Supports both:
        - Discriminative models (e.g., CodeBERT)
        - Generative models (e.g., ChatGLM)

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The loaded model instance.
    text : str
        The input text or prompt.
    model_type : str, optional
        Type of model: "discriminative" or "generative".
    label_tokens : list[str], optional
        List of label tokens for generative models, e.g. ["Yes", "No"].
    target_label : int | str
        - For discriminative models: integer index of the true class (0/1).
        - For generative models: target class token (e.g., "Yes" or "No").

    Returns
    -------
    float
        Average negative log-likelihood (ANLL) score.
        Lower values indicate higher model confidence.

    Notes
    -----
    - For discriminative models, ANLL degenerates to the standard log-loss.
    - For generative models, if the task is classification-like (e.g., Devign),
      only the final step’s log probability for the target token is used (T=1).
    - Use log_softmax for numerical stability.
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
            log_probs = F.log_softmax(logits, dim=-1)
            # Negative log-probability for the true class
            anll = -log_probs[0, target_label].item()

        return anll

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
            log_probs = F.log_softmax(last_logits, dim=-1)

            # Convert target label token to ID
            target_id = model.tokenizer.convert_tokens_to_ids(target_label)
            anll = -log_probs[target_id].item()

        return anll

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
