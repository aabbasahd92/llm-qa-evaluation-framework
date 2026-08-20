"""Minimal evaluator skeleton for Phase B.

Provides evaluate_example and evaluate_dataset helpers that are deterministic and
suitable for unit testing. Integration with Retriever/LLMService will be added
in later phases; for now the evaluator accepts model_answer as input so tests
can exercise metrics and rules_judge behavior deterministically.
"""
from typing import Dict, Any, List
from .schema import validate_example
from .metrics import exact_match, token_f1
from .rules_judge import detect_abstain, match_evidence_by_overlap, unsupported_claims_ratio


def evaluate_example(example: Dict[str, Any], model_answer: str) -> Dict[str, Any]:
    """Evaluate a single example given the model's answer (string).

    Returns a dict with per-example metrics and diagnostic fields.
    """
    validate_example(example)
    expected = example["expected_answers"][0]
    gold_text = expected["expected_text"]
    variants = expected.get("canonical_answer_variants", [])
    # include canonical variants and the main expected text for EM checking
    em_variants = [gold_text] + variants

    em = exact_match(model_answer, em_variants)
    f1 = token_f1(model_answer, gold_text)
    abstained = detect_abstain(model_answer)
    evidence_matches = match_evidence_by_overlap(model_answer, example.get("gold_evidence", []))
    hallucination = unsupported_claims_ratio(model_answer, example.get("gold_evidence", []))

    result = {
        "id": example["id"],
        "em": em,
        "token_f1": f1,
        "abstained": abstained,
        "evidence_matches": evidence_matches,
        "hallucination_ratio": hallucination,
        "model_answer": model_answer,
    }
    return result


def evaluate_dataset(dataset: List[Dict[str, Any]], answers: Dict[str, str]) -> Dict[str, Any]:
    """Evaluate a dataset. `answers` maps example id -> model answer string.

    Returns aggregated report with per-example results and averages.
    """
    results = []
    for ex in dataset:
        aid = ex["id"]
        ma = answers.get(aid, "")
        res = evaluate_example(ex, ma)
        results.append(res)

    # aggregates
    n = len(results)
    avg_f1 = sum(r["token_f1"] for r in results) / n if n else 0.0
    em_rate = sum(1 for r in results if r["em"]) / n if n else 0.0
    abstain_rate = sum(1 for r in results if r["abstained"]) / n if n else 0.0
    avg_hall = sum(r["hallucination_ratio"] for r in results) / n if n else 0.0

    report = {
        "n_examples": n,
        "em_rate": em_rate,
        "avg_token_f1": avg_f1,
        "abstain_rate": abstain_rate,
        "avg_hallucination_ratio": avg_hall,
        "results": results,
    }
    return report


# ------- End-to-end RAG evaluator helpers -------
import asyncio


async def evaluate_example_rag(example: Dict[str, Any], llm_service, retriever, prompt_builder, top_k: int = 5, diversify_sources: bool = True) -> Dict[str, Any]:
    """Evaluate a single example by running the retriever + prompt builder + LLMService.

    llm_service: instance of LLMService
    retriever: object with retrieve(query, top_k, diversify_sources)
    prompt_builder: object with build(question, retrieved_chunks)

    Returns the same dict shape as evaluate_example.
    """
    # Get model answer from LLMService (async)
    question = example.get("question")
    # Prefer LLMService convenience API if present; otherwise perform retrieve->build->client.generate
    if hasattr(llm_service, "answer_question_with_context"):
        model_answer = await llm_service.answer_question_with_context(question, retriever=retriever, prompt_builder=prompt_builder, top_k=top_k, diversify_sources=diversify_sources)
    else:
        # perform retrieval
        retrieved = retriever.retrieve(question, top_k=top_k, diversify_sources=diversify_sources)
        # build prompt
        prompt = prompt_builder.build(question, retrieved)
        # call underlying client
        client = getattr(llm_service, "_client", None)
        if client is None:
            raise RuntimeError("LLM service has no underlying client to call")
        model_answer = await client.generate(prompt, model=getattr(llm_service, "_default_model", None))

    # Delegate to existing evaluator
    return evaluate_example(example, model_answer)


def evaluate_example_rag_sync(example: Dict[str, Any], llm_service, retriever, prompt_builder, top_k: int = 5, diversify_sources: bool = True) -> Dict[str, Any]:
    """Synchronous wrapper for evaluate_example_rag for test convenience."""
    return asyncio.run(evaluate_example_rag(example, llm_service, retriever, prompt_builder, top_k=top_k, diversify_sources=diversify_sources))
