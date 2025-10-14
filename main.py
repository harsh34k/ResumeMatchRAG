from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import jwt, JWTError
from middlewares.exception_handlers import catch_exception_middleware
from routes.upload_pdfs import router as upload_router
from routes.ask_question import router as ask_router
import requests

app = FastAPI(title="Medical Assistant API", description="API for AI Medical Assistant Chatbot")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://resume-match-rag-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OIDC Configuration
ISSUER_URL = "https://oidc.vercel.com"
JWKS_URL = f"{ISSUER_URL}/.well-known/jwks"
AUDIENCE = "https://vercel.com"

async def verify_oidc_token(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No token provided")
    try:
        # Fetch JWKS
        response = requests.get(JWKS_URL)
        response.raise_for_status()
        jwks = response.json()
        # Find the correct key based on 'kid'
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        key = next((key for key in jwks["keys"] if key["kid"] == kid), None)
        if not key:
            raise JWTError("No matching key found")
        # Verify token
        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=AUDIENCE,
            issuer=ISSUER_URL,
        )
        return payload
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Token verification failed: {str(e)}")

# Health check does not require authentication
@app.get("/")
async def health_check():
    return {"status": "ok"}

# Middleware for exception handling
app.middleware("http")(catch_exception_middleware)

# Protect routes with OIDC token
app.include_router(upload_router, dependencies=[Depends(verify_oidc_token)])
app.include_router(ask_router, dependencies=[Depends(verify_oidc_token)])