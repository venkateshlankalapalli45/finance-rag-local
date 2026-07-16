import os
import sys
import argparse

# --- PATH BUG FIX ---
# This forces Python to look in the correct 'src' folder first for imports, preventing ImportErrors
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# ---------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from retrieve import LocalRetriever  # This will now import flawlessly!

class LocalGenerator:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"Loading local LLM ({self.model_name}). This might take a moment to download on first run...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Detect GPU if available, otherwise default to CPU
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=device_map
        )
        
        # Initialize the retriever
        self.retriever = LocalRetriever()

    def generate_answer(self, query, k=3, doc_name=None):
        # 1. Retrieve the matching document chunks
        matched_chunks = self.retriever.retrieve(query, k=k, doc_name_filter=doc_name)
        
        if not matched_chunks:
            return "No relevant context could be found in the database for this specific document.", []

        # 2. Format context
        context_str = ""
        for chunk in matched_chunks:
            context_str += f"[Source Document: {chunk['doc_name']}, Page: {chunk['page']}]\nText: {chunk['text']}\n\n"

        # 3. Prompt constraints
        system_prompt = (
            "You are a professional financial analyst. Use ONLY the provided context snippets below "
            "to answer the question. If the context does not contain the answer, say "
            "'I cannot find the answer in the provided documents.'\n"
            "CRITICAL: You must explicitly cite the source document name and page number "
            "from the context where you found your evidence."
        )

        user_content = f"Context:\n{context_str}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 4. Generate response
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response, matched_chunks

def main():
    parser = argparse.ArgumentParser(description="Generate RAG answers using local LLM.")
    parser.add_argument("--query", type=str, required=True, help="The question to ask.")
    parser.add_argument("--doc", type=str, default=None, help="Specific document filter (e.g., WALMART_2023_10K).")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved context chunks.")
    args = parser.parse_args()

    try:
        generator = LocalGenerator()
        answer, contexts = generator.generate_answer(args.query, k=args.k, doc_name=args.doc)
        
        print("\n" + "="*60)
        print(f"QUESTION: {args.query}")
        if args.doc:
            print(f"FILTERED TO: {args.doc}")
        print("="*60)
        print(f"ANSWER:\n{answer}")
        print("="*60)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()