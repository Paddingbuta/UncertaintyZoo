# uncertainty/methods/spuq.py
import torch
from .prompt_utils import paraphrase_prompts, compute_rouge_l

class SPUQ:
    """
    Self-Perturbation Uncertainty Quantification (SPUQ)
    """

    def __init__(self, model, tokenizer, device="cpu", num_variants=5, max_length=256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_variants = num_variants
        self.max_length = max_length

    def generate_response(self, prompt: str, temperature=1.0) -> str:
        """
        Generate a response from the model given a prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_length=self.max_length,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def quantify(self, prompt: str) -> float:
        """
        Quantify uncertainty for a given prompt using SPUQ.
        """
        # Step 1: Generate paraphrased prompts
        paraphrased_prompts = paraphrase_prompts(
            prompt, self.model, self.tokenizer, device=self.device, n=self.num_variants
        )
        P = [prompt] + paraphrased_prompts[:self.num_variants]

        # Step 2: Generate responses with temperature=1.0
        Y = [self.generate_response(p, temperature=1.0) for p in P]

        # Step 3: Compute ROUGE-L similarities
        y0 = Y[0]
        p0 = P[0]
        similarities = []
        for i in range(1, len(P)):
            rouge_y = compute_rouge_l(y0, Y[i])          # response similarity
            rouge_p = compute_rouge_l(p0, P[i])          # prompt similarity
            similarities.append(rouge_y * rouge_p)       # weighted match

        # Step 4: Aggregate into SPUQ score
        if not similarities:
            return 1.0  # Max uncertainty if no paraphrases
        spuq_score = 1.0 - sum(similarities) / len(similarities)  # 1 - consistency
        return spuq_score
