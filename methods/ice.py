import torch
import torch.nn.functional as F
import random
from statistics import mean
from typing import Callable, List, Tuple, Optional


def ice(
    model,
    prompt: str,
    model_type: str = "generative",
    example_pool: Optional[List[Tuple[str, str]]] = None,
    k_shot: int = 3,
    num_samples: int = 5,
    label_tokens: Optional[List[str]] = None,
    metric: str = "variance",
    insertion_mode: str = "auto",
    placeholder_token: str = "{FEW_SHOTS}",
    render_example: Optional[Callable[[int, Tuple[str, str]], str]] = None,
    max_length: int = 512,
):
    """
    In-Context Learning Sampling (ICL-Sample)
    -----------------------------------------
    Estimate uncertainty by varying the composition of few-shot examples
    injected into the *user-provided* prompt, then measuring variability
    of the model's output distributions across multiple sampled contexts.

    Key difference from naive implementations:
    - The base prompt is supplied by the user.
    - This function internally creates S variant prompts by sampling K examples
      from an example pool and inserting them via a placeholder or by
      pre/append, without overriding the user's instructions.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model instance (e.g., CodeBERT or ChatGLM).
    prompt : str
        The base prompt provided by the user. It may optionally contain
        a placeholder token (e.g., "{FEW_SHOTS}") where sampled examples will be inserted.
    model_type : str, optional
        "generative" or "discriminative".
    example_pool : list[(str, str)], optional
        Pool of (input, label) pairs for ICL. Must contain at least k_shot items.
    k_shot : int, optional
        Number of few-shot examples inserted per sampled prompt (default: 3).
    num_samples : int, optional
        Number of ICL prompt variants to sample (default: 5).
    label_tokens : list[str], optional
        For generative classification-style prompting (e.g., ["Yes", "No"]).
        Required for "generative" to map to class probabilities.
    metric : str, optional
        "variance" | "expected_entropy" | "mi".
    insertion_mode : str, optional
        "auto" (default), "placeholder", "prepend", or "append".
        - "auto": if placeholder exists, replace; otherwise prepend.
    placeholder_token : str, optional
        Placeholder to be replaced by few-shot block when insertion_mode is "auto" or "placeholder".
    render_example : Callable[[int, (str,str)], str], optional
        Custom renderer to format each example pair. If None, a default renderer is used.
    max_length : int, optional
        Max sequence length for tokenization when using discriminative models.

    Returns
    -------
    float
        The uncertainty score induced by ICL variability.

    Notes
    -----
    - Generative models: probabilities are computed over label_tokens at the final decoding step.
    - Discriminative models: treat the entire constructed prompt as input text and read softmax.
    - This method captures *context sensitivity* (ICL dependence), not dropout stochasticity.
    """

    if example_pool is None or len(example_pool) < k_shot:
        raise ValueError("example_pool must be provided and contain at least k_shot items.")

    if render_example is None:
        # Default renderer for each (input, label) pair
        def render_example(i: int, pair: Tuple[str, str]) -> str:
            x, y = pair
            return f"Example {i+1}\nInput:\n{x}\nAnswer:\n{y}"

    def build_few_shot_block(pairs: List[Tuple[str, str]]) -> str:
        """Render a block of few-shot examples."""
        lines = [render_example(i, p) for i, p in enumerate(pairs)]
        return "\n\n".join(lines)

    def inject_examples(base_prompt: str, few_shot_block: str) -> str:
        """Insert few-shot block into the user-provided prompt with minimal intrusion."""
        if insertion_mode == "placeholder" or (
            insertion_mode == "auto" and placeholder_token in base_prompt
        ):
            # Replace a user-declared placeholder
            return base_prompt.replace(placeholder_token, few_shot_block, 1)
        elif insertion_mode == "prepend" or insertion_mode == "auto":
            # Prepend by default when no placeholder found
            return f"{few_shot_block}\n\n{base_prompt}"
        elif insertion_mode == "append":
            return f"{base_prompt}\n\n{few_shot_block}"
        else:
            # Fallback to prepend
            return f"{few_shot_block}\n\n{base_prompt}"

    def entropy(p: torch.Tensor) -> float:
        """Shannon entropy of a probability distribution."""
        return -torch.sum(p * torch.log(p + 1e-12)).item()

    def get_probs(text: str) -> torch.Tensor:
        """
        Return a normalized class probability distribution for the given text.
        - Generative: map final-token distribution to label_tokens.
        - Discriminative: read class softmax directly.
        """
        if model_type == "generative":
            if not label_tokens:
                raise ValueError("label_tokens must be provided for generative models.")
            inputs = model.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]
                probs_vocab = torch.softmax(last_logits, dim=-1)
                label_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in label_tokens]
                probs = probs_vocab[label_ids]
                probs = probs / probs.sum()
            return probs.cpu()

        elif model_type == "discriminative":
            inputs = model.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
            return probs.cpu()

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # ---- Sample S ICL variants and collect class distributions ----
    sampled_probs = []
    for _ in range(num_samples):
        # Sample K examples without replacement
        pairs = random.sample(example_pool, k_shot)
        few_shot_block = build_few_shot_block(pairs)
        variant_prompt = inject_examples(prompt, few_shot_block)
        probs = get_probs(variant_prompt)
        sampled_probs.append(probs)

    stacked = torch.stack(sampled_probs)  # [S, C]
    mean_probs = stacked.mean(dim=0)

    if metric == "expected_entropy":
        # Average entropy across S samples
        H_exp = mean([entropy(p) for p in sampled_probs])
        return H_exp

    elif metric == "mi":
        # Mutual Information (BALD-style): H(mean_p) - mean H(p_i)
        H_mean = entropy(mean_probs)
        H_exp = mean([entropy(p) for p in sampled_probs])
        return H_mean - H_exp

    elif metric == "variance":
        # Average class-wise variance across samples
        var = torch.var(stacked, dim=0).mean().item()
        return var

    else:
        raise ValueError("metric must be one of: 'variance', 'expected_entropy', 'mi'.")
