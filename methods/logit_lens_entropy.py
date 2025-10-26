import torch
import torch.nn.functional as F


def logit_lens_entropy(model, text, model_type="discriminative",
                       layers=None, pool_method="mean"):
    """
    Logit-Lens Entropy (Hidden-State Level)
    ---------------------------------------
    Estimates uncertainty by projecting intermediate hidden states
    into the vocabulary (logit) space and computing the entropy of
    the resulting distributions.  Inspired by the "Logit Lens" idea
    from Elhage et al., this quantifies how information converges
    across model layers.

    Definition
    -----------
        For layer l:
            z^(l) = W_out · h^(l)
            p^(l) = softmax(z^(l))
            H^(l) = -Σ_v p_v^(l) log p_v^(l)

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model instance (e.g., CodeBERT or ChatGLM).
    text : str
        Input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".
    layers : list[int], optional
        Indices of hidden layers to analyze. If None, use all layers.
    pool_method : str, optional
        "mean" to average hidden states across tokens, or
        "cls"/"last" to use a specific token.

    Returns
    -------
    dict
        Mapping {layer_index: entropy_value} for each selected layer.

    Notes
    -----
    - Discriminative: use pooled hidden states or the [CLS] token
      before the classifier head; project with classifier weights.
    - Generative: use decoder hidden states before the output
      projection (e.g., lm_head) for each chosen layer.
    - Entropy decreases as representations become more confident
      or concentrated toward the correct output.
    """
    entropies = {}

    if model_type == "discriminative":
        # ----------------------------------------
        # Discriminative models (e.g., CodeBERT)
        # ----------------------------------------
        inputs = model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # list of [1, seq_len, hidden_dim]
            W_out = model.classifier.weight  # [num_labels, hidden_dim]

            target_layers = layers or range(len(hidden_states))
            for l in target_layers:
                h = hidden_states[l]  # [1, seq_len, hidden_dim]
                if pool_method == "mean":
                    h = h.mean(dim=1)  # [1, hidden_dim]
                else:
                    h = h[:, 0, :]      # CLS token
                z = torch.matmul(h, W_out.T)       # [1, num_labels]
                probs = F.softmax(z, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
                entropies[l] = entropy

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        inputs = model.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # list of [1, seq_len, hidden_dim]
            W_out = model.lm_head.weight           # [vocab_size, hidden_dim]

            target_layers = layers or range(len(hidden_states))
            for l in target_layers:
                h = hidden_states[l]  # [1, seq_len, hidden_dim]
                if pool_method == "mean":
                    h = h.mean(dim=1)  # [1, hidden_dim]
                else:
                    h = h[:, -1, :]     # last token
                z = torch.matmul(h, W_out.T)       # [1, vocab_size]
                probs = F.softmax(z, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
                entropies[l] = entropy

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return entropies
