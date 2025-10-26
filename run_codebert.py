import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from quantifier import Quantifier

# =====================================================
# Configuration
# =====================================================
MODEL_PATH = "./model/fine_tuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# List of all 29 method names in methods/
ALL_METHODS = [
    # Token / Output-level
    "average_neg_log_likelihood", "average_probability", "perplexity",
    "max_token_entropy", "avg_prediction_entropy", "token_impossibility_score",
    "margin_score", "max_probability", "least_confidence", "predictive_entropy",
    "deepgini",
    # Ensemble-based
    "expected_entropy", "mutual_information", "mc_dropout_variance",
    "class_prediction_variance", "class_probability_variance",
    "sample_variance", "max_diff_variance", "min_variance",
    "cosine_similarity_embeddings",
]

# =====================================================
# Example input (Devign binary classification)
# =====================================================
TEST_SNIPPETS = [
    ("int main() { char buf[8]; gets(buf); }", 1),   # vulnerable
    ("int main() { char buf[8]; fgets(buf, 8, stdin); }", 0),  # safe
]

PROMPT_TEMPLATE = (
    "Determine whether the following C function contains a vulnerability. "
    "Answer 'Yes' if vulnerable, 'No' if safe.\n\nCode:\n{code}"
)

LABEL_TOKENS = ["Yes", "No"]  # binary classification tokens


# =====================================================
# Initialize model and quantifier
# =====================================================
def load_codebert_model():
    print(f"[Loading fine-tuned CodeBERT model from {MODEL_PATH}]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    model.to(DEVICE).eval()
    model.tokenizer = tokenizer
    return model


def main():
    # Load model
    model = load_codebert_model()

    # Initialize Quantifier with all methods
    uq = Quantifier(model, methods=ALL_METHODS)

    # Iterate through test samples
    for idx, (code_snippet, label) in enumerate(TEST_SNIPPETS):
        print("=" * 80)
        print(f"[Sample {idx+1}]  True Label: {label}  Code:\n{code_snippet}\n")

        # Build prompt (for consistency with generative setup)
        prompt = PROMPT_TEMPLATE.format(code=code_snippet)

        # Run all methods (CodeBERT is discriminative)
        results = uq.quantify(
            input_text=code_snippet,
            prompt=prompt,
            model_type="discriminative",
            task_type="classification",
            num_samples=10,
            label_tokens=LABEL_TOKENS
        )

        # Print results
        for m, score in results.items():
            print(f"{m:35s}: {score}")

    print("\nAll methods executed successfully ✅")


if __name__ == "__main__":
    main()
