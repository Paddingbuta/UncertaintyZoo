import torch
import torch.nn.functional as F
from itertools import combinations


def cosine_similarity_embeddings(model, text, model_type="discriminative",
                                 num_samples=10, dropout=True, pool_method="mean"):
    """
    Cosine Similarity of Embeddings (Output/Representation-Level)
    -------------------------------------------------------------
    Measures semantic consistency between multiple stochastic
    forward passes by comparing embedding vectors instead of
    probability distributions.

    Definition
    ----------
        CosSim = (2 / [S * (S - 1)]) *
                  Σ_{1 ≤ i < j ≤ S} ( e^(i) · e^(j) / ||e^(i)|| ||e^(j)|| )

    Uncertainty Mapping
    -------------------
        U = 1 - CosSim

    Range
    -----
        CosSim ∈ [-1, 1], typically [0, 1] in practice (vectors point roughly
        in the same direction). Lower CosSim → higher uncertainty.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model instance (e.g., CodeBERT or ChatGLM).
    text : str
        Input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".
    num_samples : int, optional
        Number of stochastic forward passes (default: 10).
    dropout : bool, optional
        Whether to enable dropout for sampling (default: True).
    pool_method : str, optional
        How to obtain a single embedding vector from hidden states:
        "mean" (average pooling) or "cls" (first token).

    Returns
    -------
    float
        1 - mean cosine similarity, representing embedding-level uncertainty.

    Notes
    -----
    - Discriminative models: use penultimate hidden state or pooled output.
    - Generative models: use hidden states of the last token or the
      mean of all hidden states from each sampled generation.
    - This method is slower than entropy-based ones but captures deeper
      semantic instability between stochastic runs.
    """
    model.train(dropout)
    embeddings = []

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

        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Get penultimate layer hidden state
                hidden_states = outputs.hidden_states[-2]  # [1, seq_len, hidden]
                if pool_method == "mean":
                    emb = hidden_states.mean(dim=1)  # [1, hidden]
                else:
                    emb = hidden_states[:, 0, :]     # CLS token
                embeddings.append(emb.squeeze(0))

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        inputs = model.tokenizer(text, return_tensors="pt")

        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # final layer
                if pool_method == "mean":
                    emb = hidden_states.mean(dim=1)
                else:
                    emb = hidden_states[:, -1, :]  # last token
                embeddings.append(emb.squeeze(0))

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Compute pairwise cosine similarities
    sims = []
    for i, j in combinations(range(len(embeddings)), 2):
        sim = F.cosine_similarity(
            embeddings[i].unsqueeze(0),
            embeddings[j].unsqueeze(0),
            dim=-1
        ).item()
        sims.append(sim)

    cos_sim = sum(sims) / len(sims)
    uncertainty = 1.0 - cos_sim
    return uncertainty
