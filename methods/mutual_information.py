import torch
import torch.nn.functional as F


def mutual_information(model, text, model_type="discriminative",
                       label_tokens=None, num_samples=10, dropout=True):
    """
    Mutual Information (BALD, Output-Level, Ensemble-Based)
    -------------------------------------------------------
    Estimates the *epistemic* (model / parameter) uncertainty by measuring
    the disagreement across multiple stochastic forward passes.
    This is also known as the BALD (Bayesian Active Learning by Disagreement)
    criterion.

    Definition
    ----------
        Let  p^(s)  be the class probability distribution for the s-th sample,
        and  p̄ = (1/S) * Σ_s p^(s)  the mean predictive distribution.

        Predictive entropy:
            H_predictive = -Σ_c p̄_c log p̄_c

        Expected entropy:
            H_expected   = (1/S) * Σ_s [ -Σ_c p_c^(s) log p_c^(s) ]

        Mutual Information (BALD):
            MI = H_predictive - H_expected

    Range
    -----
        [0, ln(C)], where C is the number of classes.
        - MI → 0 if all samples produce identical distributions (no model
          uncertainty).
        - MI is large when the mean looks uncertain but each individual
          sample is confident yet contradictory.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model instance (e.g., CodeBERT or ChatGLM).
    text : str
        Input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".
    label_tokens : list[str], optional
        For generative models, list of label tokens (e.g., ["Yes", "No"]).
    num_samples : int, optional
        Number of stochastic forward passes (default: 10).
    dropout : bool, optional
        Whether to enable dropout during sampling.

    Returns
    -------
    float
        Mutual information (BALD) score.

    Notes
    -----
    - Requires the same sampling procedure as Expected Entropy.
    - Ensuring dropout or other stochasticity is enabled is essential;
      otherwise MI collapses to ~0.
    - For generative models, probability mass must be restricted to label
      tokens; otherwise token-level noise dominates.
    """
    sample_probs = []
    entropies = []

    if model_type == "discriminative":
        # ----------------------------------------
        # Discriminative models (e.g., CodeBERT)
        # ----------------------------------------
        model.train(dropout)
        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        for _ in range(num_samples):
            with torch.no_grad():
                logits = model(**inputs).logits  # [1, num_labels]
                probs = torch.softmax(logits, dim=-1)[0]
                entropy = -torch.sum(probs * torch.log(probs + 1e-12))
                entropies.append(entropy.item())
                sample_probs.append(probs)

        mean_probs = torch.stack(sample_probs, dim=0).mean(dim=0)
        h_predictive = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12))
        h_expected = torch.tensor(entropies).mean()
        mi = (h_predictive - h_expected).item()

        return mi

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
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

                entropy = -torch.sum(class_probs * torch.log(class_probs + 1e-12))
                entropies.append(entropy.item())
                sample_probs.append(class_probs)

        mean_probs = torch.stack(sample_probs, dim=0).mean(dim=0)
        h_predictive = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12))
        h_expected = torch.tensor(entropies).mean()
        mi = (h_predictive - h_expected).item()

        return mi

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
