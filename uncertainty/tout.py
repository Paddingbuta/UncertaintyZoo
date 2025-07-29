"""
tout.py

Uncertainty Method: Tree-of-Thought Uncertainty (TouT)

This method quantifies uncertainty in reasoning tasks via:
1. Final Answer Variance: Disagreement in final answers from different reasoning paths.
2. Trajectory Disagreement: Semantic dissimilarity between reasoning paths (using Sentence-BERT).

Usage:
    from uncertainty.methods.tout import TreeOfThoughtUncertainty

    model = ...  # Your generation model (e.g., ChatGLM3, LLaMA2, etc.)
    tokenizer = ...  # Corresponding tokenizer
    tout = TreeOfThoughtUncertainty(model, tokenizer, device="cuda")

    code_prompt = "<your code-related question or input>"
    score = tout.quantify(code_prompt)

Dependencies:
    - sentence-transformers
    - torch
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import GenerationConfig


class TreeOfThoughtUncertainty:
    def __init__(self, model, tokenizer, device="cuda", n_paths=10, model_name='all-MiniLM-L6-v2'):
        """
        Args:
            model: Generation model (e.g., ChatGLM, LLaMA, etc.)
            tokenizer: Tokenizer corresponding to model
            device: 'cuda' or 'cpu'
            n_paths: Number of reasoning paths to sample
            model_name: Sentence-BERT model for trajectory similarity
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_paths = n_paths
        self.embedder = SentenceTransformer(model_name)

    def generate_thought_paths(self, prompt):
        """
        Generate multiple reasoning paths using Tree-of-Thought prompting.
        """
        thoughts = []
        input_prompt = (
            f"{prompt}\nLet's think step by step. At each step, consider multiple options before continuing.\n"
        )

        for _ in range(self.n_paths):
            inputs = self.tokenizer(input_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    num_return_sequences=1,
                    output_scores=False,
                    return_dict_in_generate=False
                )
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            thoughts.append(text.strip())

        return thoughts

    def extract_final_answers(self, thought_paths):
        """
        Extract the final answer from each reasoning path.
        Assumes the final answer is the last sentence or line.
        """
        answers = []
        for path in thought_paths:
            lines = [line.strip() for line in path.split('\n') if line.strip()]
            final = lines[-1] if lines else ""
            answers.append(final)
        return answers

    def compute_final_answer_variance(self, final_answers):
        """
        Measure answer disagreement: proportion of unique final answers.
        """
        unique = set(final_answers)
        variance_score = 1 - (len(unique) / len(final_answers))
        return variance_score

    def compute_trajectory_disagreement(self, reasoning_paths):
        """
        Compute average dissimilarity between all pairs of reasoning paths.
        Uses cosine similarity on Sentence-BERT embeddings.
        """
        embeddings = self.embedder.encode(reasoning_paths, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(embeddings, embeddings)
        # We only want upper triangular part, excluding diagonal
        n = similarity_matrix.shape[0]
        dissimilarities = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_matrix[i][j].item()
                dissimilarity = 1 - sim
                dissimilarities.append(dissimilarity)
        if dissimilarities:
            return np.mean(dissimilarities)
        else:
            return 0.0

    def quantify(self, prompt):
        """
        Main entry to compute TouT uncertainty.

        Args:
            prompt (str): The input question/code snippet.

        Returns:
            float: Uncertainty score combining answer variance and trajectory dissimilarity.
        """
        thought_paths = self.generate_thought_paths(prompt)
        final_answers = self.extract_final_answers(thought_paths)

        var_score = self.compute_final_answer_variance(final_answers)
        traj_score = self.compute_trajectory_disagreement(thought_paths)

        # You can tune weights if desired
        uncertainty_score = 0.5 * var_score + 0.5 * traj_score
        return uncertainty_score
