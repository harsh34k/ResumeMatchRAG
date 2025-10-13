import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "resume-match-index2"
PINECONE_INDEX_HOST = "https://resume-match-index2-2m491df.svc.aped-4627-b74a.pinecone.io"

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-east-1")
existing_indexes = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(host=PINECONE_INDEX_HOST)

MAX_FILES_PER_UPLOAD = 50
BATCH_SIZE = 10

def load_vectorstore(uploaded_files, job_description: str):
    """
    Upload resumes and store their embeddings + job description context.
    """
    if len(uploaded_files) > MAX_FILES_PER_UPLOAD:
        raise ValueError(f"Upload limit exceeded: Max {MAX_FILES_PER_UPLOAD} files allowed per request.")
    
    file_paths = []
    upload_dir_path = Path(UPLOAD_DIR)
    upload_dir_path.mkdir(parents=True, exist_ok=True)
    
    for file in uploaded_files:
        save_path = upload_dir_path / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))
    
    print(f"Processing {len(file_paths)} uploaded resumes...")
    
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    index_stats = index.describe_index_stats()
    print(f"Current index stats: {index_stats}")  

    for i in range(0, len(file_paths), BATCH_SIZE):
        batch_paths = file_paths[i:i + BATCH_SIZE]
        for file_path in batch_paths:
            try:
                pdf_loader = PyPDFLoader(file_path)
                raw_docs = pdf_loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunked_docs = text_splitter.split_documents(raw_docs)
                
                for chunk in chunked_docs:
                    chunk.metadata["file_name"] = Path(file_path).name
                    chunk.metadata["text"] = chunk.page_content
                    chunk.metadata["job_description"] = job_description  # store JD once
                    chunk.metadata["type"] = "resume"
                
                texts = [chunk.page_content for chunk in chunked_docs]
                metadatas = [chunk.metadata for chunk in chunked_docs]
                ids = [f"{Path(file_path).name}-{j}" for j in range(len(chunked_docs))]

                embeddings = embed_model.embed_documents(texts)
                
                index.upsert(vectors=zip(ids, embeddings, metadatas))
                print(f"✅ Uploaded {file_path} with JD context to Pinecone")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")  

    vector_store = PineconeVectorStore(index=index, embedding=embed_model)
    print("Vector store ready:", vector_store)


