import os
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from pinecone import Pinecone
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.schema import BaseRetriever 
from pydantic import Field
from typing import Optional, List
from modules.llm import get_llm_chain,get_contextualizer_chain
from modules.query_handlers import query_chain
from logger import logger

router = APIRouter()
PINECONE_INDEX_NAME = "resume-match-index2"

chat_histories: dict[str, List[dict]] = {}  

PINECONE_INDEX_NAME = "resume-match-index2"

@router.post("/ask/")
async def ask_question(
    session_id: str = Form(...),
    question: str = Form(...)
):
    try:
        logger.info(f"[session {session_id}] user query: {question}")

        history = chat_histories.get(session_id, [])
        
        contextualizer = get_contextualizer_chain()
        context_input = {
            "chat_history": history,
            "question": question
        }
        standalone = contextualizer.run(context_input) 
        logger.info(f"[session {session_id}] standalone question: {standalone}")

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(PINECONE_INDEX_NAME)

        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        embedded_query = embed_model.embed_query(standalone)
        res = index.query(vector=embedded_query, top_k=10, include_metadata=True)
        logger.info(f"[session {session_id}] Pinecone query returned {len(res['matches'])} matches")

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"] if match["metadata"].get("text", "").strip()
        ]
        if not docs:
            return JSONResponse(status_code=400, content={"error": "No relevant documents found"})

        retriever = SimpleRetriever(docs)  
        chain = get_llm_chain(retriever)
        result = query_chain(chain, standalone,jd_text=docs[0].metadata.get("job_description", ""),chat_history="\n".join([f"{m['role']}: {m['content']}" for m in history]))


        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.get("response", "")})
        chat_histories[session_id] = history

        return result

    except Exception as e:
        logger.exception(f"[session {session_id}] Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})


class SimpleRetriever(BaseRetriever):
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[dict] = Field(default_factory=dict)

    def __init__(self, documents: List[Document]):
        super().__init__()
        self._docs = documents

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._docs


@router.post("/ask/top_candidates/")
async def get_top_candidates(job_description: str = Form(...)):
    try:
        if not job_description or not isinstance(job_description, str):
            return JSONResponse(
                status_code=400,
                content={"error": "Job description must be a non-empty string"}
            )

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

