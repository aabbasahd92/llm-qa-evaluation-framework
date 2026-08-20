from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from llm.service import LLMService, get_llm_service

app = FastAPI(title="llm-qa-evaluation-framework", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., description="The user's question to the QA system.")


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str] = Field(default_factory=list)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, llm_service: LLMService = Depends(get_llm_service)) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty or whitespace.")

    try:
        answer = await llm_service.answer_question(question)
    except ValueError:
        raise HTTPException(status_code=400, detail="Question cannot be empty or whitespace.")
    except Exception:
        # Hide provider/internal errors from clients
        raise HTTPException(status_code=502, detail="LLM provider error")

    return AskResponse(
        question=question,
        answer=answer,
        sources=[],
    )
