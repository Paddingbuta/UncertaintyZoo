import torch
import torch.nn.functional as F


def max_token_entropy(model, text, model_type="discriminative"):
    """
    Maximum Token Entropy (Token-level)
    -----------------------------------
    Definition
    ----------
    Let the model output distribution for token position t be p^(t),
    with entropy:
        H_t = -Σ_i p_i^(t) log p_i^(t)
    The maximum token entropy is then:
        MaxTokenEntropy(x) = max_t H_t

    Intuition
    ----------
    Measures the most uncertain token (or local decision point) in the
    sequence. It answers: “At which position is the model least confident?”

    Characteristics
    ---------------
    - H_t ∈ [0, ln(C)] for classification or [0, ln(V)] for vocabulary size V.
    - Sensitive to local difficulties in sequence modeling.
    - More meaningful for generative or sequence-labeling tasks.

    Implementation Details
    ----------------------
    - Generative models: compute softmax(logits[t]) for each decoding step
      and take the maximum entropy across all tokens.
    - Discriminative models: since no sequence of logits exists,
      this metric is less informative; typically returns entropy of the
      output layer (single token) or None.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model instance (CodeBERT, ChatGLM, etc.).
    text : str
        The input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".

    Returns
    -------
    float
        Maximum token entropy (higher = more uncertain).
    """
    if model_type == "discriminative":
        # ----------------------------------------
        # Discriminative model (e.g., CodeBERT)
        # ----------------------------------------
        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            logits = model(**inputs).logits  # [1, num_labels]
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
            # No sequence dimension; single entropy value
            return entropy.item()

    elif model_type == "generative":
        # ----------------------------------------
        # Generative model (e.g., ChatGLM)
        # ----------------------------------------
        inputs = model.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

            probs = torch.softmax(logits, dim=-1)
            entropies = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
            # Take the maximum entropy across all token positions
            max_entropy = torch.max(entropies).item()
            return max_entropy

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
