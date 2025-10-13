from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from typing import Optional
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from pydantic import Field
from typing import List
from logger import logger
import os

router = APIRouter()
PINECONE_INDEX_NAME = "resume-match-index2"

@router.post("/ask/")
async def ask_question(question: str = Form(...), job_description: str = Form(...)):
    try:
        logger.info(f"User query: {question} with JD: {job_description}")
        print(f"[ask.py] Received question: {question!r}, JD: {job_description!r}")  # Debug
        
        if not question or not isinstance(question, str):
            return JSONResponse(status_code=400, content={"error": "Question must be a non-empty string"})
        if not job_description or not isinstance(job_description, str):
            return JSONResponse(status_code=400, content={"error": "Job description must be a non-empty string"})
        
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(PINECONE_INDEX_NAME)
        
        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        effective_query = f"Based on this job description: {job_description}\n{question}"
        embedded_query = embed_model.embed_query(effective_query)
        
        # Query without filter to ensure all resumes are considered
        res = index.query(vector=embedded_query, top_k=10, include_metadata=True)
        print(f"[ask.py] Pinecone query returned {len(res['matches'])} matches")  # Debug
        for i, match in enumerate(res['matches']):
            print(f"[ask.py] Match {i+1}: file_name={match['metadata'].get('file_name', 'Unknown')}, type={match['metadata'].get('type', 'Unknown')}, score={match['score']:.4f}, content={match['metadata'].get('text', '')[:100]}...")  # Debug
        
        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"] if match["metadata"].get("text", "").strip()
        ]
        print(f"[ask.py] Created {len(docs)} documents")  # Debug
        for i, doc in enumerate(docs):
            print(f"[ask.py] Document {i+1}: file_name={doc.metadata.get('file_name', 'Unknown')}, content_length={len(doc.page_content)}")  # Debug
        
        if not docs:
            logger.warning("No documents retrieved from Pinecone")
            return JSONResponse(status_code=400, content={"error": "No resumes found in the database"})
        
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
        print(f"[ask.py] Passing to query_chain: question={question!r}, jd={job_description!r}")  # Debug
        result = query_chain((chain, retriever), question, job_description)
        
        logger.info("Query successful")
        return result
    except Exception as e:
        logger.exception(f"Error processing question: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})