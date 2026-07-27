# Local Financial RAG Pipeline

A privacy-preserving, context-aware Retrieval-Augmented Generation (RAG) system for answering questions from financial filings using a fully local stack.

## Overview

This project builds a local question-answering pipeline for corporate financial documents such as 10-K and 10-Q reports. Instead of relying on external APIs, it combines:

- PDF ingestion and document parsing
- Text chunking and embedding generation
- FAISS-based vector search
- Metadata-aware retrieval to restrict answers to the requested document
- A local LLM for grounded answer generation

The goal is to provide factual, source-based responses for financial questions while keeping the workflow private and fully local.

## Key Features

- Fully local document processing and inference
- Metadata-based routing for more accurate document-specific retrieval
- Context-constrained generation to reduce hallucinations
- Support for CLI-based querying, API access, and a Streamlit web app
- Evaluation support for measuring answer quality against benchmark questions

## Project Architecture

1. Data ingestion
   - Downloads and parses financial PDF files
   - Extracts text and creates document chunks

2. Embedding and indexing
   - Converts chunk text into dense vector embeddings
   - Stores them in a FAISS index for fast similarity search

3. Retrieval and generation
   - Retrieves relevant chunks using semantic search
   - Uses a local LLM to generate answers grounded in the retrieved context

4. Evaluation
   - Measures answer overlap against benchmark expected answers

## Project Structure

```text
.
├── data/
│   ├── raw_pdfs/
│   ├── financebench_open_source.jsonl
│   ├── financebench_document_information.jsonl
│   ├── faiss_index.index
│   ├── chunks_metadata.json
│   └── evaluation_results.json
├── src/
│   ├── downloads_pdfs.py
│   ├── ingest.py
│   ├── retrieve.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── api.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Installation

Make sure you are using Python 3.10 or newer.

Install the required dependencies:

```bash
pip install torch transformers sentence-transformers faiss-cpu pypdf streamlit fastapi uvicorn requests tqdm
```

## Usage

### 1. Download financial PDFs

```bash
python src/downloads_pdfs.py
```

### 2. Build the vector index

```bash
python src/ingest.py
```

### 3. Run retrieval manually

```bash
python src/retrieve.py --query "What is the total revenue of Walmart?" --doc "WALMART_2020_10K" --k 3
```

### 4. Generate an answer with the local LLM

```bash
python src/generate.py --query "What was the total revenue of Walmart in 2020?" --doc "WALMART_2020_10K" --k 3
```

### 5. Evaluate the pipeline

```bash
python src/evaluate.py --limit 10 --k 3
```

### 6. Start the API server

```bash
uvicorn src.api:app --reload
```

### 7. Launch the web app

```bash
streamlit run src/app.py
```

## Example Workflow

- Load financial documents from the local data folder
- Create text chunks from each PDF page
- Convert chunks into embeddings
- Retrieve the most relevant passages for a question
- Generate a grounded answer with source context

## Evaluation

The evaluation pipeline compares generated answers with expected benchmark answers using a simple overlap-based score. This helps measure the effectiveness of the retrieval strategy and the quality of the generated response.

## Notes

- The first run may download the embedding model and local LLM weights, which can take some time.
- GPU support is optional but can significantly improve performance.
- The metadata filter helps reduce retrieval mistakes across different fiscal years.

## Author

Created by venkatesh lankalapalli and Nishanth singh.

## License

This project is intended for educational and research purposes.
