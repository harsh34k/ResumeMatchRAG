from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from fastapi.responses import JSONResponse
from modules.load_vectorstore import load_vectorstore
from logger import logger

router = APIRouter()

@router.post("/upload_pdfs/")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    job_description: str = Form(...)
):
    try:
        logger.info("📂 Received uploaded files with job description")

        # Pass both arguments here 👇
        load_vectorstore(files, job_description)

        logger.info("✅ Documents added to vectorstore successfully")
        return {"message": "Files processed and vectorstore updated"}

    except ValueError as ve:
        logger.error(f"⚠️ Value error: {ve}")
        return JSONResponse(status_code=400, content={"error": str(ve)})

    except Exception as e:
        logger.exception("💥 Error during PDF upload")
        return JSONResponse(status_code=500, content={"error": str(e)})
