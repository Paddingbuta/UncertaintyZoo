import math
import torch
import torch.nn.functional as F


def perplexity(model, text, model_type="discriminative",
               label_tokens=None, target_label=None):
    """
    Perplexity (PPL)
    ----------------
    Token-level uncertainty measure derived from the
    Average Negative Log-Likelihood (ANLL).

    Definition
    ----------
        PPL(x, y₁:T) = exp(ANLL(x, y₁:T))
                     = exp(-1/T * Σ log pθ(y_t | y_<t, x))
    For single-label classification (T = 1):
        PPL = 1 / pθ(y* | x)

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The loaded model instance.
    text : str
        Input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".
    label_tokens : list[str], optional
        Label tokens for generative models, e.g. ["Yes", "No"].
    target_label : int | str
        - For discriminative models: integer label index (0/1).
        - For generative models: target token string ("Yes"/"No").

    Returns
    -------
    float
        Perplexity score. Range [1, ∞).
        Lower is better; 1 indicates a perfectly confident model.

    Notes
    -----
    - PPL is simply exp(ANLL).
    - For discriminative models: PPL = 1 / p(y*|x).
    - For generative models: if classification-like, use
      the final token's probability (T = 1).
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
            logits = model(**inputs).logits
            log_probs = F.log_softmax(logits, dim=-1)
            anll = -log_probs[0, target_label].item()
            ppl = math.exp(anll)

        return ppl

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        if label_tokens is None or target_label is None:
            raise ValueError("label_tokens and target_label must be provided for generative models.")

        inputs = model.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)

            target_id = model.tokenizer.convert_tokens_to_ids(target_label)
            anll = -log_probs[target_id].item()
            ppl = math.exp(anll)

        return ppl

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
