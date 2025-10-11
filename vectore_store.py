import io
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings import embedding_model
import pinecone

# Initialize Pinecone
pinecone.init(api_key="YOUR_PINECONE_KEY", environment="gcp-starter")
index_name = "resume-match-rag"
index = pinecone.Index(index_name)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def _extract_text_from_pdf(content: bytes):
    pdf = PyPDFLoader(io.BytesIO(content))
    pages = pdf.load()
    return " ".join([p.page_content for p in pages])
def store_resumes(content, filename):
    text = _extract_text_from_pdf(content)
    chunks = splitter.split_text(text)
    metadatas = [{"type": "resume", "filename": filename} for _ in chunks]
    Pinecone.from_texts(chunks, embedding_model, index_name=index_name, metadatas=metadatas)
    return text

def store_job_description(content, filename):
    text = _extract_text_from_pdf(content)
    chunks = splitter.split_text(text)
    metadatas = [{"type": "job_spec", "filename": filename}]
    Pinecone.from_texts(chunks, embedding_model, index_name=index_name, metadatas=metadatas)
    return text

def find_top_matches(top_k=10):
    # Fetch the latest job spec (assuming one uploaded)
    job_query = index.query(
        vector=embedding_model.embed_query("Find resumes that match this job description"),
        top_k=1,
        include_metadata=True,
        filter={"type": "job_spec"},
    )

    if not job_query or not job_query["matches"]:
        return []

    # Use the job description text for matching
    job_text = job_query["matches"][0]["metadata"]["filename"]

    # Find top resumes
    results = index.query(
        vector=embedding_model.embed_query(job_text),
        top_k=top_k,
        include_metadata=True,
        filter={"type": "resume"},
    )

    return results["matches"]
