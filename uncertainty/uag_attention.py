import torch
import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List

def generate_cot_attentions(model, tokenizer, prompt: str, num_samples: int = 5, max_new_tokens: int = 64, device: str = "cuda"):
    """
    Step 1: Generate multiple CoT outputs with attention using sampling.
    Ensure output_attentions=True and extract attention matrices.
    """

    model.to(device)
    model.eval()
    
    all_attentions = []

    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Use model.forward() to manually decode step-by-step (for attention support)
        with torch.no_grad():
            output = model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )
            attentions = output.attentions  # list of layers, shape: (batch, heads, seq, seq)
            all_attentions.append(attentions)

    return all_attentions, inputs["input_ids"][0]


def compute_attention_variance(attentions_list: List[tuple], token_index: int, layer: int = -1, head: int = 0):
    """
    Step 2–3: Extract attention distributions for the same token
    across multiple samples, then compute variance.
    """
    vectors = []
    for attention in attentions_list:
        matrix = attention[layer][0]  # shape: (heads, seq, seq)
        vec = matrix[head, token_index, :]  # vector of attention over all tokens
        vectors.append(vec.cpu().numpy())

    stacked = np.stack(vectors, axis=0)  # (num_samples, seq_len)
    return float(np.var(stacked, axis=0).mean())


def uag_attention_score(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    code_str: str,
    num_samples: int = 5,
    max_new_tokens: int = 64,
    layer: int = -1,
    head: int = 0,
    target_token: str = "vulnerability",
    device: str = "cuda"
):
    """
    Compute UAG uncertainty score based on attention variance.

    Usage:
        >>> from transformers import AutoModel, AutoTokenizer
        >>> model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        >>> tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        >>> score = uag_attention_score(model, tokenizer, "<your code>", device="cuda")

    Notes:
        - Ensure your model supports `output_attentions=True`.
        - ChatGLM3 may require special decoding interface for CoT generation.
    """
    prompt = f"Does the following code contain a vulnerability?\n{code_str}\nLet's think step by step:"

    attentions_list, input_ids = generate_cot_attentions(model, tokenizer, prompt, num_samples, max_new_tokens, device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    try:
        token_index = tokens.index(target_token)
    except ValueError:
        token_index = len(tokens) // 2  # fallback

    score = compute_attention_variance(attentions_list, token_index, layer, head)
    return score
