from src.agents.execution_agent import plan_query
from src.agents.guardrails import enforce_grounding


def test_planner_routes_comparison_questions() -> None:
    plan = plan_query("Compare annual pricing for each vendor")
    assert plan.intent == "comparison"
    assert plan.requires_comparison is True


def test_guardrail_returns_required_no_context_message() -> None:
    assert enforce_grounding("unsupported answer", []) == "Information not found in context"