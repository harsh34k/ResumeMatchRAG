from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from vectorstore import store_resumes, store_job_description, find_top_matches
from llm import analyze_match
import uvicorn

app = FastAPI(title="ResumeMatchRAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/upload/resumes/")
async def upload_resumes(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        filename = file.filename
        store_resumes(content, filename)
        results.append({"filename": filename, "status": "uploaded"})
    return {"message": "Resumes uploaded successfully", "uploaded": results}

@app.post("/upload/job/")
async def upload_job(file: UploadFile = File(...)):
    content = await file.read()
    store_job_description(content, file.filename)
    return {"message": "Job description uploaded successfully", "filename": file.filename}

@app.get("/match/")
async def match_resumes():
    matches = find_top_matches(top_k=10)
    analyzed = analyze_match(matches)
    return {"top_matches": analyzed}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
