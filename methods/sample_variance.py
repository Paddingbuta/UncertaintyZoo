import torch
import torch.nn.functional as F


def sample_variance(model, text, model_type="discriminative",
                    label_tokens=None, num_samples=10,
                    f_type="max_prob", dropout=True):
    """
    Sample Variance (Output-Level, Ensemble-Based)
    ----------------------------------------------
    Computes the variance of a scalar uncertainty statistic
    f(p^(s)) across multiple stochastic forward passes.

    Definition
    ----------
        SV(f) = Var_s[f(p^(s))]
              = (1 / (S - 1)) Σ_s ( f(p^(s)) - mean(f) )²

    where each p^(s) is the predicted class probability
    distribution from the s-th stochastic forward pass
    (e.g., by enabling dropout).

    Purpose
    --------
        Quantifies how sensitive a chosen scalar metric (f)
        is to stochastic sampling — higher variance indicates
        higher epistemic uncertainty.

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
        (e.g., ["Yes", "No"]).
    num_samples : int, optional
        Number of stochastic forward passes (default: 10).
    f_type : str, optional
        The scalar function applied to each sampled distribution:
          - "max_prob"  → f = max_c p_c
          - "entropy"   → f = H(p)
          - "margin"    → f = p1 - p2  (top-2 difference)
    dropout : bool, optional
        Whether to enable dropout during sampling.

    Returns
    -------
    float
        Sample variance of the chosen scalar uncertainty measure.

    Notes
    -----
    - Discriminative models: sample softmax probabilities directly.
    - Generative models: use class probabilities at the final step
      projected onto label tokens.
    - Common pitfalls:
        * Too few samples → high Monte-Carlo noise.
        * Avoid mixing metrics already covered by dedicated variance methods.
    """
    values = []

    # Helper: compute scalar f(p)
    def compute_scalar(probs):
        if f_type == "max_prob":
            return torch.max(probs).item()
        elif f_type == "entropy":
            return (-torch.sum(probs * torch.log(probs + 1e-12))).item()
        elif f_type == "margin":
            top2 = torch.topk(probs, 2).values
            return (top2[0] - top2[1]).item()
        else:
            raise ValueError(f"Unknown f_type: {f_type}")

    # ------------------------------
    # Discriminative (e.g., CodeBERT)
    # ------------------------------
    if model_type == "discriminative":
        model.train(dropout)  # enable dropout
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
                values.append(compute_scalar(probs))

    # ------------------------------
    # Generative (e.g., ChatGLM)
    # ------------------------------
    elif model_type == "generative":
        if label_tokens is None:
            raise ValueError("label_tokens must be provided for generative models.")

        model.train(dropout)
        inputs = model.tokenizer(text, return_tensors='pt')

        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]
                probs = torch.softmax(last_logits, dim=-1)
                label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
                class_probs = probs[label_ids]
                class_probs = class_probs / class_probs.sum()
                values.append(compute_scalar(class_probs))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Compute unbiased sample variance
    if len(values) < 2:
        return 0.0
    tensor_vals = torch.tensor(values)
    return torch.var(tensor_vals, unbiased=True).item()
