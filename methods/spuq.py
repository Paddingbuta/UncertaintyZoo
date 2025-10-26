import torch
import torch.nn.functional as F
import random
import re


def spuq(model, text, model_type="discriminative", label_tokens=None,
         num_perturb=5, distance="kl", perturb_mode="code"):
    """
    Self-Perturbation Uncertainty Quantification (SPUQ)
    ---------------------------------------------------
    Estimates uncertainty by applying small, semantic-preserving perturbations
    to the input and measuring how much the model's output distribution changes.

    Definition
    ----------
        SPUQ(x) = (1/M) * Σ_m D( p(x) || p(x + δ_m) )

    where:
        - p(x): baseline probability distribution
        - δ_m: the m-th input perturbation
        - D(·||·): divergence measure (KL or L2)

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model instance (e.g., CodeBERT or ChatGLM).
    text : str
        Input text or prompt (for classification-like generative tasks).
    model_type : str, optional
        "discriminative" or "generative".
    label_tokens : list[str], optional
        Label tokens for generative models (e.g., ["Yes", "No"]).
    num_perturb : int, optional
        Number of perturbations to generate (default: 5).
    distance : str, optional
        Divergence type: "kl" (Kullback-Leibler) or "l2" (Euclidean).
    perturb_mode : str, optional
        Type of perturbation:
            - "text": synonym replacements or token masking
            - "code": variable renaming, statement reordering, small edits

    Returns
    -------
    float
        SPUQ score. Higher values → greater sensitivity → higher uncertainty.

    Notes
    -----
    - For discriminative models: compares softmax distributions.
    - For generative models: compares class probabilities over label tokens.
    - Perturbations aim to preserve semantics but alter syntax/form.
    """
    def get_probs(input_text):
        """Return normalized probability distribution for given input."""
        if model_type == "discriminative":
            inputs = model.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
            return probs

        elif model_type == "generative":
            if label_tokens is None:
                raise ValueError("label_tokens must be provided for generative models.")
            inputs = model.tokenizer(input_text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]
                probs = torch.softmax(last_logits, dim=-1)
                label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
                probs = probs[label_ids]
                probs = probs / probs.sum()
            return probs
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def kl_divergence(p, q):
        """KL divergence D_KL(p || q)."""
        return torch.sum(p * torch.log((p + 1e-12) / (q + 1e-12)))

    def l2_distance(p, q):
        """L2 distance between distributions."""
        return torch.sqrt(torch.sum((p - q) ** 2))

    def perturb_input(input_text):
        """Generate semantic-preserving perturbation."""
        if perturb_mode == "text":
            words = input_text.split()
            if len(words) < 3:
                return input_text
            # randomly mask or swap
            if random.random() < 0.5:
                i = random.randint(0, len(words) - 1)
                words[i] = "[MASK]"
            else:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
            return " ".join(words)

        elif perturb_mode == "code":
            # Variable renaming and statement shuffle (simple heuristic)
            code = input_text
            # rename simple variable identifiers like a,b,c
            vars = re.findall(r"\b[a-zA-Z_]\w*\b", code)
            if vars:
                var_to_replace = random.choice(vars)
                new_name = var_to_replace + random.choice(["_tmp", "_x", "_1"])
                code = re.sub(rf"\b{var_to_replace}\b", new_name, code)
            # occasionally reorder two statements (split by ;)
            if ";" in code and random.random() < 0.5:
                stmts = code.split(";")
                if len(stmts) > 2:
                    i, j = random.sample(range(len(stmts) - 1), 2)
                    stmts[i], stmts[j] = stmts[j], stmts[i]
                    code = ";".join(stmts)
            return code

        else:
            return input_text

    # ---- Compute baseline probability ----
    base_probs = get_probs(text)

    # ---- Generate perturbations ----
    divergences = []
    for _ in range(num_perturb):
        perturbed_text = perturb_input(text)
        perturbed_probs = get_probs(perturbed_text)

        if distance == "kl":
            div = kl_divergence(base_probs, perturbed_probs)
        elif distance == "l2":
            div = l2_distance(base_probs, perturbed_probs)
        else:
            raise ValueError("distance must be 'kl' or 'l2'.")

        divergences.append(div.item())

    spuq_score = sum(divergences) / len(divergences)
    return spuq_score
