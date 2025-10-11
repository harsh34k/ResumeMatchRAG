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
