import torch
import torch.nn.functional as F


def margin_score(model, text, model_type="discriminative", label_tokens=None):
    """
    Margin Score (Output-Level)
    ---------------------------
    Implementation for the Devign binary classification task.

    Definition
    ----------
        Given the sorted class probabilities p_(1) >= p_(2) >= ...,
        Margin(p) = p_(1) - p_(2)

    Interpretation
    --------------
        * Range: [0, 1]
        * In binary classification, Margin = 2 * max(p, 1-p) - 1
        * Measures how much the top prediction leads the runner-up.
          A larger margin indicates stronger confidence and robustness.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The loaded model instance.
    text : str
        The input text or prompt.
    model_type : str, optional
        "discriminative" or "generative".
    label_tokens : list[str], optional
        For generative models, the fixed label tokens (e.g., ["Yes", "No"]).

    Returns
    -------
    float
        Margin score between the top-1 and top-2 probabilities.
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
            top2 = torch.topk(probs, 2).values
            margin = (top2[0] - top2[1]).item()

        return margin

    elif model_type == "generative":
        # ----------------------------------------
        # Generative models (e.g., ChatGLM)
        # ----------------------------------------
        if label_tokens is None:
            raise ValueError("label_tokens must be provided for generative models.")

        inputs = model.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)

            # Extract probabilities of label tokens and normalize
            label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
            label_probs = probs[label_ids]
            label_probs = label_probs / label_probs.sum()

            top2 = torch.topk(label_probs, 2).values
            margin = (top2[0] - top2[1]).item()

        return margin

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
