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
        self.embedding_model = self.model
        
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
            self.chunks_metadata = self.chunks

   # In src/retrieve.py

    def retrieve(self, query, k=3, doc_name_filter=None):
        # 1. If a filter is provided, filter the corpus/metadata FIRST
        if doc_name_filter:
            # Match exact doc_name or loose substring match (e.g. WALMART_2023_10K)
            candidate_indices = [
                i for i, meta in enumerate(self.chunks)
                if meta.get("doc_name") == doc_name_filter 
                or doc_name_filter in meta.get("doc_name", "")
                or doc_name_filter in meta.get("file_name", "")
            ]
            
            if not candidate_indices:
                print(f"Warning: No chunks match the doc_name_filter: {doc_name_filter}")
                return []

            # Convert query to vector
            query_vector = self.embedding_model.encode([query])
            
            # Search FAISS or calculate cosine similarity ONLY on candidate_indices
            # (Alternatively, retrieve top_k * 20 from FAISS and filter down to matching candidates)
            matched_chunks = []
            # Query FAISS with higher depth then filter down
            distances, indices = self.index.search(query_vector, k=min(1000, self.index.ntotal))
            
            for idx in indices[0]:
                if idx in candidate_indices:
                    matched_chunks.append(self.chunks_metadata[idx])
                    if len(matched_chunks) == k:
                        break
            return matched_chunks

        # 2. Standard unfiltered retrieval (when doc_name_filter is None)
        query_vector = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_vector, k=k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

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