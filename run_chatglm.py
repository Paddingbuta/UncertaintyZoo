import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from quantifier import Quantifier

# =====================================================
# Configuration
# =====================================================
MODEL_PATH = "./model/chatglm3-6b"   # 改为你实际保存路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# List of all 29 uncertainty methods
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
    # Input-level
    "spuq", "ice", "icl_sample",
    # Hidden / Reasoning-level
    "logit_lens_entropy", "uag", "cot_uq", "topology_uq", "stable_explanation_conf", "tout",
]

# =====================================================
# Example: Devign-style binary classification prompts
# =====================================================
TEST_SNIPPETS = [
    ("int main() { char buf[8]; gets(buf); }", 1),   # vulnerable
    ("int main() { char buf[8]; fgets(buf, 8, stdin); }", 0),  # safe
]

PROMPT_TEMPLATE = (
    "You are a vulnerability detection assistant.\n"
    "Determine whether the following C code is vulnerable.\n\n"
    "Code:\n{code}\n\n"
    "Answer only with 'Yes' if vulnerable or 'No' if safe.\nReason step by step before the final answer."
)

LABEL_TOKENS = ["Yes", "No"]


# =====================================================
# Model loading
# =====================================================
def load_chatglm_model():
    print(f"[Loading ChatGLM model from {MODEL_PATH}]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
    model.tokenizer = tokenizer
    return model


# =====================================================
# Main execution
# =====================================================
def main():
    model = load_chatglm_model()

    # Initialize Quantifier
    uq = Quantifier(model, methods=ALL_METHODS)

    for idx, (code_snippet, label) in enumerate(TEST_SNIPPETS):
        print("=" * 100)
        print(f"[Sample {idx+1}]  True Label: {label}")
        print(f"Code:\n{code_snippet}\n")

        prompt = PROMPT_TEMPLATE.format(code=code_snippet)

        # Compute all uncertainty scores
        results = uq.quantify(
            input_text=code_snippet,
            prompt=prompt,
            model_type="generative",        # ChatGLM is generative
            task_type="classification",
            num_samples=5,                  # for ensemble-based metrics
            label_tokens=LABEL_TOKENS,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05
        )

        # Display results
        print("🔍 Uncertainty Results:")
        for m, score in results.items():
            print(f"{m:35s}: {score}")
        print()

    print("\nAll methods executed successfully ✅")


if __name__ == "__main__":
    main()
