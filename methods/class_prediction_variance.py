import torch
import torch.nn.functional as F


def class_prediction_variance(model, text, model_type="discriminative",
                              label_tokens=None, num_samples=10, dropout=True):
    """
    Class Prediction Variance (Output-Level, Ensemble-Based)
    --------------------------------------------------------
    Measures the dispersion of the *final predicted label* across multiple
    stochastic forward passes (e.g., enabling dropout). It captures how
    split the "votes" are among classes.

    Definition
    ----------
        Let y^(s) be the one-hot label of argmax p^(s) for sample s.
        The vote rate for class c is:
            p_hat_c = (1/S) * Σ_s y_c^(s)
        The class prediction variance is:
            Var_pred = 1 - Σ_c (p_hat_c)^2
        This is aligned with Gini/entropy style measures over the vote
        distribution. It reaches its maximum when votes are evenly split,
        and minimum (0) when all samples agree on the same label.

    Range
    -----
        [0, 1 - 1/C], where C is the number of classes.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model instance (e.g., CodeBERT or ChatGLM).
    text : str
        Input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".
    label_tokens : list[str], optional
        For generative models, the list of label tokens (e.g., ["Yes", "No"]).
    num_samples : int, optional
        Number of stochastic forward passes (default: 10).
    dropout : bool, optional
        Whether to keep dropout layers active during sampling.

    Returns
    -------
    float
        Class prediction variance (higher = more disagreement).

    Notes
    -----
    - Ignores per-sample confidence magnitudes; it only looks at the
      *argmax label* each time. Thus it can be low even if each pass is
      uncertain but consistently picks the same class.
    - For tasks framed as classification with a generative model, we map
      the final-step vocabulary distribution to the finite label set and
      take argmax over those label probs.
    """
    if model_type == "discriminative":
        # ----------------------------------------
        # Discriminative models (e.g., CodeBERT)
        # ----------------------------------------
        model.train(dropout)  # enable dropout for stochastic sampling
        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        # Infer number of classes from a single forward
        with torch.no_grad():
            logits = model(**inputs).logits  # [1, C]
            num_classes = logits.shape[-1]

        vote_counts = torch.zeros(num_classes, dtype=torch.long)

        for _ in range(num_samples):
            with torch.no_grad():
                logits = model(**inputs).logits  # [1, C]
                probs = torch.softmax(logits, dim=-1)[0]  # [C]
                pred = torch.argmax(probs).item()
                vote_counts[pred] += 1

        p_hat = vote_counts.float() / float(num_samples)  # normalized vote distribution
        score = 1.0 - torch.sum(p_hat ** 2).item()
        return score

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        if label_tokens is None or len(label_tokens) == 0:
            raise ValueError("label_tokens must be provided for generative models.")

        model.train(dropout)
        inputs = model.tokenizer(text, return_tensors='pt')

        num_classes = len(label_tokens)
        vote_counts = torch.zeros(num_classes, dtype=torch.long)
        label_ids = None  # lazily map to ids once

        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]  # [vocab_size]
                probs = torch.softmax(last_logits, dim=-1)

                if label_ids is None:
                    label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]

                class_probs = probs[label_ids]                 # [C]
                class_probs = class_probs / class_probs.sum()  # normalize over labels
                pred = torch.argmax(class_probs).item()
                vote_counts[pred] += 1

        p_hat = vote_counts.float() / float(num_samples)
        score = 1.0 - torch.sum(p_hat ** 2).item()
        return score

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
