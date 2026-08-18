from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
def ask_question(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty or whitespace.")

    placeholder_answer = (
        "This is a placeholder response from the Stage 1 foundation. "
        f"No LLM or retrieval system is connected yet for: '{question}'"
    )

    return AskResponse(
        question=question,
        answer=placeholder_answer,
        sources=[],
    )
