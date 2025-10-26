import torch


def token_impossibility_score(model, text, model_type="generative"):
    """
    Token Impossibility Score (Token-level)
    ---------------------------------------
    Definition:
        Let m_t = max_i p_i^(t) be the maximum class probability at decoding step t.
        Then the Token Impossibility Score is defined as:
            TokenImpossibility(x) = 1 - min_t(m_t) = max_t(1 - m_t)

    Intuition:
        Measures whether there exists any decoding step where the model
        becomes highly uncertain. If even one step has a very low maximum
        probability, the whole sequence is considered "impossible" or unreliable.

    Properties:
        - Range: [0, 1]
        - Lower scores → all steps are confident (m_t ≈ 1)
        - Higher scores → at least one step was very uncertain (small m_t)

    Implementation Notes:
        - Generative models: compute max(prob[t]) for each decoding step t,
          take min(m_t), and return (1 - min_m).
        - Discriminative models: no sequence steps; this metric degenerates
          to the Least Confidence score (1 - max(p)).

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The loaded model instance.
    text : str
        Input prompt or sequence.
    model_type : str, optional
        Either "discriminative" or "generative".

    Returns
    -------
    float
        Token Impossibility Score in [0, 1].
    """
    import torch.nn.functional as F

    if model_type == "discriminative":
        # Degenerates to Least Confidence (1 - max(p))
        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            score = 1.0 - torch.max(probs).item()
        return score

    elif model_type == "generative":
        # Token-level uncertainty across decoding steps
        inputs = model.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits[0], dim=-1)  # [seq_len, vocab_size]

        # Maximum probability at each token position
        max_probs, _ = torch.max(probs, dim=-1)  # [seq_len]
        # Find the least confident step (minimum max prob)
        min_conf = torch.min(max_probs).item()
        score = 1.0 - min_conf
        return score

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
