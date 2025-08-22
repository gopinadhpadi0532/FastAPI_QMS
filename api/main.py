from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import logging

from core.config import get_settings
#from core.qa_engine2 import QAEngine
from core.qa_engine import QAEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Conversational QMS Navigator API (LangChain Version)",
    description="API for querying Quality Management System documents.",
    version="2.0.0"
)

# --- Pydantic Models for API ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# --- Global Objects ---
try:
    settings = get_settings()
    qa_engine = QAEngine(settings=settings)
except Exception as e:
    logger.error(f"Failed to initialize QA Engine: {e}", exc_info=True)
    qa_engine = None

# --- API Endpoints ---
@app.post("/ask", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def ask_question(request: QueryRequest):
    """
    Accepts a user question and returns a synthesized answer with source citations.
    """
    if not qa_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA Engine is not available. Please check server logs."
        )

    if not request.question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )
        
    try:
        result = qa_engine.query(request.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        logger.error(f"An error occurred while processing the query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your question."
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok" if qa_engine else "degraded"}