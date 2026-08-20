"""End-to-end evaluator tests using Fake LLM and Fake Retriever."""
import sys
import asyncio
from pathlib import Path
import json

# Ensure repo root is on sys.path so package imports resolve in CI
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from llm.service import LLMService

from rag.eval import evaluator


class _FakeBuilder:
    def build(self, question: str, retrieved_chunks):
        parts = ["USER QUESTION\n", question.strip(), "\n\n"]
        # include retrieved chunk texts if present
        for item in retrieved_chunks:
            c = item.get("chunk") if isinstance(item, dict) else item
            text = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else str(c))
            parts.insert(0, f"[Source chunk]\n{text}\n\n")
        return "".join(parts)


class _FakeClient:
    def __init__(self):
        self.last_prompt = None

    async def generate(self, prompt: str, model=None) -> str:
        self.last_prompt = prompt
        # respond with a distilled answer that repeats the user's question and includes context marker
        return "ANS:" + prompt.split("USER QUESTION\n")[-1].strip()


class _FakeRetriever:
    def __init__(self, chunks):
        # chunks are simple dicts with 'text' and 'source'
        self.chunks = chunks
        self.last_call = None

    def retrieve(self, q, top_k=5, diversify_sources=True):
        self.last_call = {"q": q, "top_k": top_k, "diversify_sources": diversify_sources}
        return [{"chunk": c, "similarity": 1.0} for c in self.chunks[:top_k]]


def make_chunk(i, source, text):
    return {"chunk_id": f"c{i}", "document_id": f"d{i}", "source": source, "text": text, "chunk_index": 0}


def _load_sample():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "evaluation_dataset_sample.json"
    return json.loads(data_path.read_text(encoding="utf-8"))


def test_evaluator_e2e_happy_path():
    ds = _load_sample()
    ex = ds[0]
    chunks = [make_chunk(1, "a.md", "A helpful sentence about resetting your password."), make_chunk(2, "b.md", "Other text.")]
    retriever = _FakeRetriever(chunks)
    client = _FakeClient()
    svc = LLMService(client)
    builder = _FakeBuilder()

    # run async evaluator
    res = asyncio.run(evaluator.evaluate_example_rag(ex, svc, retriever, builder, top_k=2, diversify_sources=False))
    assert retriever.last_call["q"] == ex["question"]
    # confirm the last prompt used by client contains the user question
    assert client.last_prompt is not None
    assert res["model_answer"].startswith("ANS:")
    assert "reset" in res["model_answer"].lower()


def test_evaluator_e2e_abstain():
    ds = _load_sample()
    ex = ds[1]
    chunks = []
    retriever = _FakeRetriever(chunks)
    client = _FakeClient()
    svc = LLMService(client)
    builder = _FakeBuilder()

    res = asyncio.run(evaluator.evaluate_example_rag(ex, svc, retriever, builder))
    # fake client echoes question; since no context it will include the question only — judge should not mark abstain automatically
    assert isinstance(res["abstained"], bool)
