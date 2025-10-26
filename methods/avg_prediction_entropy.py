import torch
import torch.nn.functional as F


def avg_prediction_entropy(model, text, model_type="discriminative", label_tokens=None):
    """
    Average Prediction Entropy (Token-level)
    ----------------------------------------
    Measures the average uncertainty density across tokens or prediction steps.

    Definition
    ----------
        AvgPredEntropy(x) = (1/T) * Σ_t H_t
                         = -(1/T) * Σ_t Σ_i p_i^(t) log p_i^(t)

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The loaded model instance.
    text : str
        Input text or prompt.
    model_type : str, optional
        Type of model: "discriminative" or "generative".
    label_tokens : list[str], optional
        Label tokens for generative models, e.g. ["Yes", "No"].

    Returns
    -------
    float
        Average prediction entropy. Higher values indicate higher average
        uncertainty per token.

    Notes
    -----
    - For discriminative models, entropy is computed once over the class
      probability distribution (T=1).
    - For generative models:
        * If the task is classification-like (e.g., Devign), entropy is taken
          over the final token’s probability distribution (T=1).
        * For sequence generation, the entropy can be averaged across all
          tokens.
    - Entropy range: [0, ln(C)] for C classes (or ln(V) for vocabulary size).
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
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            log_probs = torch.log(probs + 1e-12)
            entropy = -torch.sum(probs * log_probs).item()
        return entropy

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        inputs = model.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]

            # If only final-token classification (e.g., Devign)
            if label_tokens is not None:
                last_logits = logits[-1]
                probs = torch.softmax(last_logits, dim=-1)
                label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
                probs = probs[label_ids]
                probs = probs / probs.sum()
                log_probs = torch.log(probs + 1e-12)
                entropy = -torch.sum(probs * log_probs).item()
                return entropy

            # Otherwise: compute mean entropy across tokens
            probs = torch.softmax(logits, dim=-1)
            token_entropies = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
            avg_entropy = token_entropies.mean().item()
            return avg_entropy

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
