"""Small, explicit agent workflow for planning, retrieval, generation, and validation."""

from dataclasses import dataclass
import re
from typing import Callable

from src.config import get_settings

from .guardrails import enforce_grounding, grounded_context
from .retrieval_agent import RetrievedChunk, retrieve


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    requires_comparison: bool


def plan_query(prompt: str) -> QueryPlan:
    comparison = bool(re.search(r"\b(compare|versus|vs\.?|difference|cheapest|ranking)\b", prompt, re.I))
    intent = "comparison" if comparison else "document_question"
    return QueryPlan(intent=intent, requires_comparison=comparison)


def execute_query(prompt: str, retrieve_fn: Callable[[str], list[RetrievedChunk]] = retrieve) -> tuple[str, list[RetrievedChunk]]:
    plan = plan_query(prompt)
    chunks = retrieve_fn(prompt)
    if not chunks:
        return "Information not found in context", []

    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for query generation")
    from groq import Groq

    format_instruction = (
        "For comparisons, return a concise Markdown table with one row per vendor and cite every factual cell."
        if plan.requires_comparison
        else "Answer directly and cite every factual claim."
    )
    completion = Groq(api_key=settings.groq_api_key).chat.completions.create(
        model=settings.groq_model,
        temperature=0,
        timeout=60,
        messages=[
            {
                "role": "system",
                "content": "You are a procurement analyst. Use only the supplied context. " + format_instruction,
            },
            {"role": "user", "content": f"Question: {prompt}\n\nContext:\n{grounded_context(chunks)}"},
        ],
    )
    answer = completion.choices[0].message.content or ""
    return enforce_grounding(answer, chunks), chunks