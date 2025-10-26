import torch
import torch.nn.functional as F


def expected_entropy(model, text, model_type="discriminative",
                     label_tokens=None, num_samples=10, dropout=True):
    """
    Expected Entropy (Output-Level, Ensemble-Based)
    -----------------------------------------------
    Measures the *average intrinsic randomness* of multiple stochastic
    forward passes (e.g., with dropout enabled).  This reflects the
    aleatoric uncertainty – uncertainty inherent to the data.

    Definition
    ----------
        H_expected = (1 / S) * Σ_s H(p^(s))
                   = (1 / S) * Σ_s [ -Σ_c p_c^(s) log p_c^(s) ]

    Range
    -----
        [0, ln(C)], where C is the number of classes.
        Larger values indicate that each sampled prediction is itself diffuse
        (i.e., high data noise or ambiguity).

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
        (e.g., ["Yes", "No"] for Devign classification).
    num_samples : int, optional
        Number of stochastic forward passes (default: 10).
    dropout : bool, optional
        Whether to enable dropout during sampling.

    Returns
    -------
    float
        Expected entropy across stochastic predictions.

    Notes
    -----
    - Discriminative models: sample multiple softmax outputs with dropout.
    - Generative models (classification-style prompt):
        * Use the probability distribution over label tokens at the final step.
    - Do not apply to open-ended generation tasks (token-level entropy would be needed).
    """
    entropies = []

    if model_type == "discriminative":
        # ----------------------------------------
        # Discriminative models (e.g., CodeBERT)
        # ----------------------------------------
        model.train(dropout)  # enable dropout layers
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

        return float(sum(entropies) / len(entropies))

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
                last_logits = outputs.logits[0, -1, :]  # [vocab_size]
                probs = torch.softmax(last_logits, dim=-1)

                label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
                class_probs = probs[label_ids]
                class_probs = class_probs / class_probs.sum()

                entropy = -torch.sum(class_probs * torch.log(class_probs + 1e-12))
                entropies.append(entropy.item())

        return float(sum(entropies) / len(entropies))

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
