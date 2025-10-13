import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  

load_dotenv()

# GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
# PINECONE_ENV="us-east-1"
PINECONE_INDEX_NAME="resume-match-index2"
PINECONE_INDEX_HOST='https://resume-match-index2-2m491df.svc.aped-4627-b74a.pinecone.io'

# os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)


# initialize pinecone instance
pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(cloud="aws",region=PINECONE_INDEX_HOST)
existing_indexes=[i["name"] for i in pc.list_indexes()]


if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)


# index=pc.Index(PINECONE_INDEX_NAME)
index = pc.Index(host=PINECONE_INDEX_HOST)

# ... existing imports ...

MAX_FILES_PER_UPLOAD = 50  # CHANGE: Set a hard limit per upload request
BATCH_SIZE = 10  # CHANGE: Process files in batches to avoid memory issues

def load_vectorstore(uploaded_files):
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
    
    # CHANGE: Check index stats before processing (e.g., storage usage)
    index_stats = index.describe_index_stats()
    print(f"Current index stats: {index_stats}")  # Log for monitoring
    # Optional: If storage near limit, raise error (e.g., if index_stats['total_vector_count'] > 900000)
    
    # CHANGE: Batch processing loop
    for i in range(0, len(file_paths), BATCH_SIZE):
        batch_paths = file_paths[i:i + BATCH_SIZE]
        for file_path in batch_paths:
            try:  # CHANGE: Wrap in try/except for error handling
                pdf_loader = PyPDFLoader(file_path)
                raw_docs = pdf_loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunked_docs = text_splitter.split_documents(raw_docs)
                print("chuncked_docsjjjjjjjjjjjjjjjjjjj", chunked_docs)
                
                for chunk in chunked_docs:
                    chunk.metadata["file_name"] = Path(file_path).name
                    chunk.metadata["text"] = chunk.page_content
                
                texts = [chunk.page_content for chunk in chunked_docs]
                metadatas = [chunk.metadata for chunk in chunked_docs]
                ids = [f"{Path(file_path).name}-{j}" for j in range(len(chunked_docs))]
                
                embeddings = embed_model.embed_documents(texts)
                
                index.upsert(vectors=zip(ids, embeddings, metadatas))
                print(f"✅ Uploaded {file_path} to Pinecone")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")  # Log and continue (or raise)
                # e.g., if "quota exceeded", suggest upgrading plan
    
    vector_store = PineconeVectorStore(index=index, embedding=embed_model)
    print("Vector store ready:", vector_store)




# from pathlib import Path
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_pinecone import PineconeVectorStore
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings  
# from langchain.schema import Document  # For JD doc
# import os
# import time
# import uuid
# import hashlib

# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = "resume-match-index2"
# PINECONE_INDEX_HOST = 'https://resume-match-index2-2m491df.svc.aped-4627-b74a.pinecone.io'

# UPLOAD_DIR = "./uploaded_docs"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# pc = Pinecone(api_key=PINECONE_API_KEY)
# spec = ServerlessSpec(cloud="aws", region="us-east-1")  # Adjusted if needed
# existing_indexes = [i["name"] for i in pc.list_indexes()]

# if PINECONE_INDEX_NAME not in existing_indexes:
#     pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="dotproduct", spec=spec)
#     while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
#         time.sleep(1)

# index = pc.Index(host=PINECONE_INDEX_HOST)

# MAX_FILES_PER_UPLOAD = 50
# BATCH_SIZE = 10

# def load_vectorstore(uploaded_files, job_description: str):  # CHANGE: JD mandatory
#     if len(uploaded_files) > MAX_FILES_PER_UPLOAD:
#         raise ValueError(f"Max {MAX_FILES_PER_UPLOAD} files allowed.")
    
#     file_paths = []
#     upload_dir_path = Path(UPLOAD_DIR)
#     upload_dir_path.mkdir(parents=True, exist_ok=True)
    
#     for file in uploaded_files:
#         unique_filename = f"{Path(file.filename).stem}_{uuid.uuid4().hex[:8]}{Path(file.filename).suffix}"
#         save_path = upload_dir_path / unique_filename
#         with open(save_path, "wb") as f:
#             f.write(file.file.read())
#         file_paths.append(str(save_path))
    
#     print(f"Processing {len(file_paths)} resumes with JD...")
    
#     embed_model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True}
#     )
    
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
#     # CHANGE: Process and store JD (overwrite old)
#     print("Storing job description...")
#     index.delete(filter={"type": {"$eq": "jd"}})  # Overwrite
#     jd_doc = Document(page_content=job_description, metadata={"source": "job_description", "type": "jd"})
#     chunked_jd = text_splitter.split_documents([jd_doc])
#     for chunk in chunked_jd:
#         chunk.metadata["file_name"] = "job_description"
#         chunk.metadata["text"] = chunk.page_content
#         chunk.metadata["type"] = "jd"
    
#     jd_texts = [chunk.page_content for chunk in chunked_jd]
#     jd_metadatas = [chunk.metadata for chunk in chunked_jd]
#     jd_ids = [f"jd-{i}" for i in range(len(chunked_jd))]
#     jd_embeddings = embed_model.embed_documents(jd_texts)
#     index.upsert(vectors=zip(jd_ids, jd_embeddings, jd_metadatas))
#     print("JD stored")
    
#     # Process resumes (existing logic with dedup, etc.)
#     for i in range(0, len(file_paths), BATCH_SIZE):
#         batch_paths = file_paths[i:i + BATCH_SIZE]
#         for file_path in batch_paths:
#             try:
#                 with open(file_path, "rb") as f:
#                     file_bytes = f.read()
#                     content_hash = hashlib.md5(file_bytes).hexdigest()
                
#                 existing_check = index.query(vector=[0]*384, filter={"content_hash": {"$eq": content_hash}}, top_k=1, include_metadata=True)
#                 if existing_check.matches:
#                     print(f"Duplicate skipped: {file_path}")
#                     continue
                
#                 pdf_loader = PyPDFLoader(file_path)
#                 raw_docs = pdf_loader.load()
#                 chunked_docs = text_splitter.split_documents(raw_docs)
                
#                 candidate_name = "Unknown"  # Extraction logic if needed
                
#                 for chunk in chunked_docs:
#                     chunk.metadata["file_name"] = Path(file_path).name
#                     chunk.metadata["text"] = chunk.page_content
#                     chunk.metadata["candidate_name"] = candidate_name
#                     chunk.metadata["content_hash"] = content_hash
#                     chunk.metadata["type"] = "resume"  # Explicit type
                
#                 texts = [chunk.page_content for chunk in chunked_docs]
#                 metadatas = [chunk.metadata for chunk in chunked_docs]
#                 ids = [f"{Path(file_path).name}-{j}" for j in range(len(chunked_docs))]
                
#                 embeddings = embed_model.embed_documents(texts)
#                 index.upsert(vectors=zip(ids, embeddings, metadatas))
#                 print(f"Uploaded {file_path}")
#             except Exception as e:
#                 print(f"Error: {str(e)}")
    
#     vector_store = PineconeVectorStore(index=index, embedding=embed_model)
#     return vector_store





# import os
# import time
# from pathlib import Path
# from dotenv import load_dotenv
# from tqdm.auto import tqdm
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_pinecone import PineconeVectorStore
# # from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings  

# load_dotenv()

# # GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
# # PINECONE_ENV="us-east-1"
# PINECONE_INDEX_NAME="resume-match-index2"
# PINECONE_INDEX_HOST='https://resume-match-index2-2m491df.svc.aped-4627-b74a.pinecone.io'

# # os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

# UPLOAD_DIR="./uploaded_docs"
# os.makedirs(UPLOAD_DIR,exist_ok=True)


# # initialize pinecone instance
# pc=Pinecone(api_key=PINECONE_API_KEY)
# spec=ServerlessSpec(cloud="aws",region=PINECONE_INDEX_HOST)
# existing_indexes=[i["name"] for i in pc.list_indexes()]


# if PINECONE_INDEX_NAME not in existing_indexes:
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=384,
#         metric="dotproduct",
#         spec=spec
#     )
#     while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
#         time.sleep(1)


# # index=pc.Index(PINECONE_INDEX_NAME)
# index = pc.Index(host=PINECONE_INDEX_HOST)

# # ... existing imports ...

# MAX_FILES_PER_UPLOAD = 50  # CHANGE: Set a hard limit per upload request
# BATCH_SIZE = 10  # CHANGE: Process files in batches to avoid memory issues

# def load_vectorstore(uploaded_files):
#     if len(uploaded_files) > MAX_FILES_PER_UPLOAD:
#         raise ValueError(f"Upload limit exceeded: Max {MAX_FILES_PER_UPLOAD} files allowed per request.")
    
#     file_paths = []
#     upload_dir_path = Path(UPLOAD_DIR)
#     upload_dir_path.mkdir(parents=True, exist_ok=True)
#     for file in uploaded_files:
#         save_path = upload_dir_path / file.filename
#         with open(save_path, "wb") as f:
#             f.write(file.file.read())
#         file_paths.append(str(save_path))
    
#     print(f"Processing {len(file_paths)} uploaded resumes...")
    
#     embed_model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True}
#     )
    
#     # CHANGE: Check index stats before processing (e.g., storage usage)
#     index_stats = index.describe_index_stats()
#     print(f"Current index stats: {index_stats}")  # Log for monitoring
#     # Optional: If storage near limit, raise error (e.g., if index_stats['total_vector_count'] > 900000)
    
#     # CHANGE: Batch processing loop
#     for i in range(0, len(file_paths), BATCH_SIZE):
#         batch_paths = file_paths[i:i + BATCH_SIZE]
#         for file_path in batch_paths:
#             try:  # CHANGE: Wrap in try/except for error handling
#                 pdf_loader = PyPDFLoader(file_path)
#                 raw_docs = pdf_loader.load()
                
#                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#                 chunked_docs = text_splitter.split_documents(raw_docs)
#                 print("chuncked_docsjjjjjjjjjjjjjjjjjjj", chunked_docs)
                
#                 for chunk in chunked_docs:
#                     chunk.metadata["file_name"] = Path(file_path).name
#                     chunk.metadata["text"] = chunk.page_content
                
#                 texts = [chunk.page_content for chunk in chunked_docs]
#                 metadatas = [chunk.metadata for chunk in chunked_docs]
#                 ids = [f"{Path(file_path).name}-{j}" for j in range(len(chunked_docs))]
                
#                 embeddings = embed_model.embed_documents(texts)
                
#                 index.upsert(vectors=zip(ids, embeddings, metadatas))
#                 print(f"✅ Uploaded {file_path} to Pinecone")
#             except Exception as e:
#                 print(f"Error processing {file_path}: {str(e)}")  # Log and continue (or raise)
#                 # e.g., if "quota exceeded", suggest upgrading plan
    
#     vector_store = PineconeVectorStore(index=index, embedding=embed_model)
#     print("Vector store ready:", vector_store)








    # load,split,embed and upsert pdf docs content

# def load_vectorstore(uploaded_files):
#     embed_model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True}
#     )
#     file_paths = []
#     print("this is file",uploaded_files)

#     for file in uploaded_files:
#         save_path = Path(UPLOAD_DIR) / file.filename
#         with open(save_path, "wb") as f:
#             f.write(file.file.read())
#         file_paths.append(str(save_path))

#     for file_path in file_paths:
#         loader = PyPDFLoader(file_path)
#         print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh, loader here',loader)
#         documents = loader.load()
#         print("this is document",documents)

#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         chunks = splitter.split_documents(documents)

#         texts = [chunk.page_content for chunk in chunks]
#         print("texts",texts)
#         metadatas = [chunk.metadata for chunk in chunks]
#         ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]
#         print("metadata herejjjjjjjjjjjjjjjjjjjjjjjjjj",metadatas)
#         print(f"🔍 Embedding {len(texts)} chunks...")
#         embeddings = embed_model.embed_documents(texts)

#         print("📤 Uploading to Pinecone...")
#         with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
#             index.upsert(vectors=zip(ids, embeddings, metadatas))
#             progress.update(len(embeddings))

#         print(f"✅ Upload complete for {file_path}")


# def load_vectorstore(uploaded_files):
# #    load file
#    file_paths =[] 
#    for file in uploaded_files:
#         save_path = Path(UPLOAD_DIR) / file.filename
#         with open(save_path, "wb") as f:
#             f.write(file.file.read())
#         file_paths.append(str(save_path))

#    PDF_PATH =  "../uploaded_docs/1-Skills.pdf"
#    pdf_loader = PyPDFLoader('../1-Skills.pdf')
#    rawDocs = pdf_loader.load()

# #   split file into chunks

#    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50),
#    chuncked_docs = text_splitter.split_documents(rawDocs)

# #    embeddings
#    embed_model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True}
#     )
#    embeddings = embed_model.embed_documents(chuncked_docs)
   
#    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
#    print("vectore-store",vector_store)