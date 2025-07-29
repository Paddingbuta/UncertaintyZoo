# uncertainty/methods/icl_sample.py

import random
import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from .max_probability import MaxProbability


class ICLSample:
    """
    ICL-Sample Uncertainty Estimator.
    
    Constructs K different few-shot prompts and computes output uncertainty
    via average (1 - max probability).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        icl_pool: list,
        k: int = 5,
        n_shot: int = 3,
        device: str = "cpu"
    ):
        """
        Args:
            model (PreTrainedModel): A generative model (e.g., GPT-style).
            tokenizer (PreTrainedTokenizer): Corresponding tokenizer.
            icl_pool (List[Tuple[str, str]]): List of (input, output) examples for few-shot context.
            k (int): Number of ICL prompt variants to sample.
            n_shot (int): Number of few-shot examples per prompt.
            device (str): Device string.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.icl_pool = icl_pool
        self.k = k
        self.n_shot = n_shot
        self.device = device
        self.max_prob = MaxProbability(model, tokenizer, device=device)

    def quantify(self, target_input: str) -> float:
        """
        Compute ICL-Sample uncertainty score.

        Args:
            target_input (str): The new sample input (no label).
        
        Returns:
            float: Uncertainty score (higher = more uncertain).
        """
        max_probs = []

        for _ in range(self.k):
            # 1. Sample few-shot examples
            few_shots = random.sample(self.icl_pool, self.n_shot)

            # 2. Format few-shot block
            few_shot_text = ""
            for x, y in few_shots:
                few_shot_text += f"Input: {x.strip()}\nOutput: {y.strip()}\n\n"

            # 3. Construct full prompt
            full_prompt = few_shot_text + f"Input: {target_input.strip()}\nOutput:"

            # 4. Generate completion
            self.model.eval()
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=32,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 5. Use max-prob estimator on generated response
            # Truncate the full prompt to get only new output
            generated_part = decoded[len(full_prompt):].strip()

            if not generated_part:
                continue

            # Feed only generated output to classifier to get softmax score
            max_prob = self.max_prob.quantify(generated_part)
            max_probs.append(max_prob)

        if not max_probs:
            return 1.0  # Max uncertainty if generation fails

        return float(1.0 - np.mean(max_probs))
