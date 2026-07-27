from fastapi import FastAPI
from pydantic import BaseModel
from src.generate import LocalGenerator 

app = FastAPI(title="Financial RAG API")

print("Initializing Local RAG Pipeline Components...")
generator = LocalGenerator()
print("RAG System successfully loaded and ready for API requests!")

class QueryRequest(BaseModel):
    question: str
    doc_name: str = None  # <-- Add this parameter line

@app.post("/query")
def answer_question(request: QueryRequest):
    # Pass the doc_name filter into the generation method
    answer, matched_chunks = generator.generate_answer(
        query=request.question, 
        doc_name=request.doc_name
    )
    
    sources = list(set([chunk['doc_name'] for chunk in matched_chunks])) if matched_chunks else []
    return {"answer": answer, "sources": sources}