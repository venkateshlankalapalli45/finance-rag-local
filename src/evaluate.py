import os
import sys
import json
import argparse
from tqdm import tqdm

# Ensure local imports work flawlessly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from generate import LocalGenerator

# Paths
EVAL_DATA_PATH = "data/financebench_open_source.jsonl"
EVAL_RESULTS_PATH = "data/evaluation_results.json"

def compute_simple_overlap(generated, expected):
    """Computes a basic word-level overlap score between generated and ground truth."""
    gen_words = set(generated.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    shared = gen_words.intersection(exp_words)
    return len(shared) / len(exp_words)

def main():
    parser = argparse.ArgumentParser(description="Evaluate the local RAG pipeline.")
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to evaluate (default 10 for quick testing).")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved context chunks.")
    args = parser.parse_args()

    if not os.path.exists(EVAL_DATA_PATH):
        print(f"Error: Evaluation data not found at {EVAL_DATA_PATH}")
        return

    print("Initializing Local RAG Generator for Evaluation...")
    generator = LocalGenerator()
    
    eval_records = []
    print(f"Starting evaluation on the first {args.limit} test cases...")
    
    # Load evaluation cases
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()[:args.limit]

    # Run the evaluation loop
    for line in tqdm(lines, desc="Evaluating RAG"):
        item = json.loads(line)
        question = item.get("question")
        expected_answer = item.get("answer")
        doc_name = item.get("doc_name")
        evidence_text = item.get("evidence_text")
        
        try:
            # Generate answer using our local metadata-routed RAG
            generated_answer, retrieved_contexts = generator.generate_answer(question, k=args.k, doc_name=doc_name)
            
            # Basic match metrics
            score = compute_simple_overlap(generated_answer, expected_answer)
            
            eval_records.append({
                "doc_name": doc_name,
                "question": question,
                "expected_answer": expected_answer,
                "evidence_text": evidence_text,
                "generated_answer": generated_answer,
                "overlap_score": score
            })
        except Exception as e:
            print(f"\nError processing question '{question}': {e}")

    # Compute average metrics
    avg_score = sum(r["overlap_score"] for r in eval_records) / len(eval_records) if eval_records else 0
    print(f"\nEvaluation Complete! Average Overlap Score: {avg_score:.4f}")

    # Save results to disk
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "parameters": {"k": args.k, "limit": args.limit},
            "average_score": avg_score,
            "results": eval_records
        }, f, indent=4, ensure_ascii=False)
        
    print(f"Detailed results saved to {EVAL_RESULTS_PATH}")

if __name__ == "__main__":
    main()