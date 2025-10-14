import os
import time
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cloudinary.uploader
from config import cloudinary  # Assumes config.py with Cloudinary setup

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "resume-match-index2"
PINECONE_INDEX_HOST = "https://resume-match-index2-2m491df.svc.aped-4627-b74a.pinecone.io"
CLOUDINARY_UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "resume_matcher_docs")

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
    Upload resumes to Cloudinary and store their embeddings + job description context in Pinecone.
    """
    if len(uploaded_files) > MAX_FILES_PER_UPLOAD:
        raise ValueError(f"Upload limit exceeded: Max {MAX_FILES_PER_UPLOAD} files allowed per request.")

    file_urls = []
    for file in uploaded_files:
        if not file.filename.lower().endswith(".pdf"):
            raise ValueError(f"File {file.filename} is not a PDF")

        try:
            upload_result = cloudinary.uploader.upload(
                file.file,
                resource_type="raw",
                folder=CLOUDINARY_UPLOAD_FOLDER,
                public_id=file.filename.split(".")[0],
                overwrite=True
            )
            file_urls.append({
                "filename": file.filename,
                "public_id": upload_result["public_id"],
                "secure_url": upload_result["secure_url"]
            })
            print(f"✅ Uploaded {file.filename} to Cloudinary")
        except Exception as e:
            print(f"Error uploading {file.filename} to Cloudinary: {str(e)}")
            raise

    print(f"Processing {len(file_urls)} uploaded resumes...")
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    index_stats = index.describe_index_stats()
    print(f"Current index stats: {index_stats}")

    for i in range(0, len(file_urls), BATCH_SIZE):
        batch_urls = file_urls[i:i + BATCH_SIZE]
        for file_info in batch_urls:
            try:
                # Use Cloudinary secure_url directly
                pdf_loader = PyPDFLoader(file_info["secure_url"])
                raw_docs = pdf_loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunked_docs = text_splitter.split_documents(raw_docs)

                for chunk in chunked_docs:
                    chunk.metadata["file_name"] = file_info["filename"]
                    chunk.metadata["text"] = chunk.page_content
                    chunk.metadata["job_description"] = job_description
                    chunk.metadata["type"] = "resume"
                    chunk.metadata["public_id"] = file_info["public_id"]
                    chunk.metadata["secure_url"] = file_info["secure_url"]

                texts = [chunk.page_content for chunk in chunked_docs]
                metadatas = [chunk.metadata for chunk in chunked_docs]
                ids = [f"{file_info['public_id']}-{j}" for j in range(len(chunked_docs))]

                embeddings = embed_model.embed_documents(texts)

                index.upsert(vectors=zip(ids, embeddings, metadatas))
                print(f"✅ Processed {file_info['filename']} with JD context to Pinecone")

            except Exception as e:
                print(f"Error processing {file_info['filename']}: {str(e)}")
                continue

    vector_store = PineconeVectorStore(index=index, embedding=embed_model)
    print("Vector store ready:", vector_store)
    return vector_store

