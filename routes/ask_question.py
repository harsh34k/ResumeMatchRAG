import os
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from pinecone import Pinecone
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.schema import BaseRetriever 
from pydantic import Field
from typing import Optional, List
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain

router = APIRouter()
PINECONE_INDEX_NAME = "resume-match-index2"

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        if not question or not isinstance(question, str):
            return JSONResponse(status_code=400, content={"error": "Question must be a non-empty string"})
        
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(PINECONE_INDEX_NAME)
        
        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        embedded_query = embed_model.embed_query(question)
        
        # Search resumes
        res = index.query(vector=embedded_query, top_k=10, include_metadata=True)
        print(f"[ask_question] Pinecone query returned {len(res['matches'])} matches")
        
        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"] if match["metadata"].get("text", "").strip()
        ]

        if not docs:
            return JSONResponse(status_code=400, content={"error": "No resumes found in the database"})

        # ✅ Extract job description from first matching doc
        job_description = docs[0].metadata.get("job_description", "")
        print(f"[ask_question] Retrieved job description from Pinecone: {job_description[:200]}...")

        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)
            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents
            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs
        
        retriever = SimpleRetriever(docs)
        chain, retriever = get_llm_chain(retriever)
        
        result = query_chain((chain, retriever), question, job_description)
        return result
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/ask/top_candidates/")
async def get_top_candidates(job_description: str = Form(...)):
    try:
        if not job_description or not isinstance(job_description, str):
            return JSONResponse(
                status_code=400,
                content={"error": "Job description must be a non-empty string"}
            )

        # Initialize Pinecone + embeddings
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(PINECONE_INDEX_NAME)

        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        embedded_query = embed_model.embed_query(job_description)

        res = index.query(vector=embedded_query, top_k=5, include_metadata=True)
        matches = res.get("matches", [])

        if not matches:
            return JSONResponse(status_code=404, content={"message": "No candidates found"})

        candidates = []
        for match in matches:
            meta = match.get("metadata", {})
            candidates.append({
                "id": match.get("id"),
                "score": match.get("score"),
                "author": meta.get("author", "Unknown"),
                "source": meta.get("source", ""),
                "page": meta.get("page", ""),
                "text_preview": meta.get("text", "")[:300]  # first 300 chars of chunk
            })

        return {
            "query_job_description": job_description,
            "top_candidates": candidates
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

