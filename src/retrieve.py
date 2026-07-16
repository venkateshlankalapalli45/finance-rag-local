import os
import json
import faiss
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer

# Paths
INDEX_PATH = "data/faiss_index.index"
METADATA_PATH = "data/chunks_metadata.json"

class LocalRetriever:
    def __init__(self):
        # 1. Load the exact same local model used during ingestion
        print("Loading local embedding model...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Load the FAISS vector index
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Please run ingest.py first!")
        print("Loading local FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)
        
        # 3. Load the textual metadata chunks
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}. Please run ingest.py first!")
        print("Loading document text metadata...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(self, query, k=3, doc_name_filter=None):
        """
        Converts the query to a vector and retrieves the top-k matches.
        Optionally filters results to only include chunks from a specific document.
        """
        # If filtering, we look at more candidates initially to make sure we find matching document chunks
        search_k = k * 15 if doc_name_filter else k
        
        # Convert string query to a vector
        query_vector = self.model.encode([query], convert_to_numpy=True)
        
        # Search the FAISS index
        distances, indices = self.index.search(query_vector, search_k)
        
        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk_data = self.chunks[idx]
            
            # Metadata filter matching
            if doc_name_filter:
                target = doc_name_filter.replace(".pdf", "").lower()
                chunk_doc = chunk_data["doc_name"].replace(".pdf", "").lower()
                if target != chunk_doc:
                    continue
            
            results.append({
                "rank": len(results) + 1,
                "text": chunk_data["text"],
                "doc_name": chunk_data["doc_name"],
                "page": chunk_data["page"],
                "similarity_score": float(dist)
            })
            
            if len(results) >= k:
                break
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Query the local FinanceBench database.")
    parser.add_argument("--query", type=str, required=True, help="The search query or question.")
    parser.add_argument("--doc", type=str, default=None, help="Document name to restrict search (optional).")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved passages.")
    args = parser.parse_args()

    try:
        retriever = LocalRetriever()
        matched_passages = retriever.retrieve(args.query, k=args.k, doc_name_filter=args.doc)
        
        print(f"\n🔍 Top {args.k} matches for: '{args.query}'\n" + "="*50)
        for passage in matched_passages:
            print(f"Rank {passage['rank']} | Source: {passage['doc_name']} (Page {passage['page']})")
            print(f"Distance Score: {passage['similarity_score']:.4f}")
            print(f"Text Content:\n{passage['text']}\n" + "-"*50)
            
    except Exception as e:
        print(f"Error executing retrieval: {e}")

if __name__ == "__main__":
    main()