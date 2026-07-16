import os
import json
import pypdf
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
RAW_PDFS_DIR = "data/raw_pdfs/"
OPEN_SOURCE_JSONL = "data/financebench_open_source.jsonl"
OUTPUT_INDEX_PATH = "data/faiss_index.index"
OUTPUT_METADATA_PATH = "data/chunks_metadata.json"

# RAG Hyperparameters (Feel free to experiment with these for your report!)
CHUNK_SIZE = 750      # Target character length per chunk
CHUNK_OVERLAP = 75    # Overlap character length between sequential chunks

def chunk_text(text, doc_name, page_num):
    """Splits text into chunks with a sliding window overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_str = text[start:end]
        
        # Metadata keeps track of where this chunk came from
        chunks.append({
            "text": chunk_str.strip(),
            "doc_name": doc_name,
            "page": page_num + 1
        })
        
        # Slide the window forward by chunk_size minus overlap
        start += (CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks

def main():
    print("Initializing local embedding model (all-MiniLM-L6-v2)...")
    # This runs 100% locally on your CPU/GPU without API keys
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    all_chunks = []
    
    # We only want to process the PDFs that are actually mentioned/used in the evaluation set
    print("Scanning evaluation dataset to identify target files...")
    target_docs = set()
    with open(OPEN_SOURCE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "doc_name" in item:
                target_docs.add(item["doc_name"])
    
    print(f"Found {len(target_docs)} unique documents required for evaluation.")

    # Loop through and parse the target PDFs
    for doc_name in tqdm(target_docs, desc="Extracting text from PDFs"):
        filename = doc_name if doc_name.lower().endswith(".pdf") else f"{doc_name}.pdf"
        pdf_path = os.path.join(RAW_PDFS_DIR, filename)
        
        if not os.path.exists(pdf_path):
            print(f"\n[!] Warning: Missing local PDF file for {doc_name}, skipping.")
            continue
            
        try:
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        # Chunk the page text
                        page_chunks = chunk_text(page_text, doc_name, page_num)
                        all_chunks.extend(page_chunks)
        except Exception as e:
            print(f"\n[!] Error parsing {filename}: {e}")

    print(f"\nTotal generated chunks across entire corpus: {len(all_chunks)}")
    
    if not all_chunks:
        print("No text chunks extracted. Ensure your PDFs are downloaded properly in data/raw_pdfs/")
        return

    # Extract text content strings to send to the embedding model
    texts_to_embed = [c["text"] for c in all_chunks]
    
    print("Computing embeddings (Converting text to numerical vectors)...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
    
    # Create local FAISS vector index
    dimension = embeddings.shape[1] # For miniLM, this is 384 dimensions
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print("Saving FAISS index database locally...")
    faiss.write_index(index, OUTPUT_INDEX_PATH)
    
    # Save accompanying textual chunks mapping to disk
    with open(OUTPUT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
        
    print("🎉 Step 2 Ingestion Complete! Vector database created successfully.")

if __name__ == "__main__":
    main()