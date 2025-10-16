from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from middlewares.exception_handlers import catch_exception_middleware
from routes.upload_pdfs import router as upload_router
from routes.ask_question import router as ask_router

app = FastAPI(title="Resume Matcher", description="An AI powered resume filtering application for hr teams")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://talentxi.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OIDC Configuration

# Health check does not require authentication
@app.get("/")
async def health_check():
    return {"status": "ok"}

# Middleware for exception handling
app.middleware("http")(catch_exception_middleware)

# Protect routes with OIDC token
app.include_router(upload_router )
app.include_router(ask_router)