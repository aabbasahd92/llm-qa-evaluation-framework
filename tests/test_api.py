from fastapi.testclient import TestClient

from app.main import app

# Override the LLM service dependency with a fake implementation so tests
# never call external APIs.
from llm.service import get_llm_service


class _FakeLLMService:
    async def answer_question(self, question: str) -> str:
        return f"FAKE_ANSWER: {question}"


app.dependency_overrides[get_llm_service] = lambda: _FakeLLMService()

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_ask_request() -> None:
    payload = {"question": "What is the purpose of this application?"}

    response = client.post("/api/v1/ask", json=payload)

    assert response.status_code == 200
    assert response.json()["question"] == payload["question"]
    assert response.json()["answer"] == f"FAKE_ANSWER: {payload['question']}"
    assert response.json()["sources"] == []


def test_empty_question() -> None:
    response = client.post("/api/v1/ask", json={"question": ""})

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty or whitespace."


def test_whitespace_only_question() -> None:
    response = client.post("/api/v1/ask", json={"question": "   \n\t  "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty or whitespace."


def test_missing_question() -> None:
    response = client.post("/api/v1/ask", json={})

    assert response.status_code == 422


def test_response_contract() -> None:
    response = client.post("/api/v1/ask", json={"question": "Describe the project."})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"question", "answer", "sources"}
    assert isinstance(payload["question"], str)
    assert isinstance(payload["answer"], str)
    assert isinstance(payload["sources"], list)
