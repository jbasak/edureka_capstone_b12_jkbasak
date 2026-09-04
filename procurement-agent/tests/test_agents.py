"""
tests/test_agents.py

Unit tests for all five agents:
  - RetrievalAgent
  - ReasoningAgent
  - ValidationAgent
  - RequirementExtractionAgent
  - AgentController (mode detection + pipeline integration)

All external I/O (Qdrant, Groq) is mocked.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from app.agents.retrieval import RetrievalAgent, RetrievedChunk
from app.agents.reasoning import ReasoningAgent, NOT_FOUND_SENTINEL
from app.agents.validator import ValidationAgent
from app.agents.extractor import RequirementExtractionAgent
from app.agents.controller import AgentController, classify_mode
from app.schemas.chat import ChatRequest

VECTOR_DIM = 384


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _zero_vec():
    return [0.0] * VECTOR_DIM


def _make_chunk(
    text="AWS provides 99.99% uptime.",
    source="AWS_Proposal.pdf",
    vendor="AWS",
    page=3,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source_file=source,
        vendor_name=vendor,
        doc_category="proposal",
        page_number=page,
        row_index=None,
        sheet_name=None,
        chunk_index=0,
        score=0.91,
    )


def _mock_qdrant(hits=None):
    """Return a mock QdrantManager whose search() returns *hits*."""
    if hits is None:
        hits = [
            {
                "score": 0.91,
                "text": "AWS provides 99.99% uptime.",
                "source_file": "AWS_Proposal.pdf",
                "vendor_name": "AWS",
                "doc_category": "proposal",
                "page_number": 3,
                "row_index": None,
                "sheet_name": None,
                "chunk_index": 0,
            }
        ]
    mgr = MagicMock()
    mgr.search.return_value = hits
    return mgr


def _mock_llm_client(content: str):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# ── RetrievalAgent ────────────────────────────────────────────────────────────

class TestRetrievalAgent:
    def test_returns_list_of_retrieved_chunks(self):
        agent = RetrievalAgent(qdrant_manager=_mock_qdrant())
        with patch("app.agents.retrieval.embed_query", return_value=_zero_vec()):
            results = agent.retrieve("What is the uptime SLA?")
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)

    def test_chunk_fields_populated(self):
        agent = RetrievalAgent(qdrant_manager=_mock_qdrant())
        with patch("app.agents.retrieval.embed_query", return_value=_zero_vec()):
            results = agent.retrieve("uptime")
        chunk = results[0]
        assert chunk.source_file == "AWS_Proposal.pdf"
        assert chunk.vendor_name == "AWS"
        assert chunk.score == pytest.approx(0.91)
        assert chunk.page_number == 3

    def test_empty_db_returns_empty_list(self):
        agent = RetrievalAgent(qdrant_manager=_mock_qdrant(hits=[]))
        with patch("app.agents.retrieval.embed_query", return_value=_zero_vec()):
            results = agent.retrieve("anything")
        assert results == []

    def test_vendor_filter_passed_to_qdrant(self):
        mgr = _mock_qdrant()
        agent = RetrievalAgent(qdrant_manager=mgr)
        with patch("app.agents.retrieval.embed_query", return_value=_zero_vec()):
            agent.retrieve("pricing", vendor_filter=["AWS"])
        mgr.search.assert_called_once()
        _, kwargs = mgr.search.call_args
        assert kwargs.get("vendor_filter") == ["AWS"]

    def test_citation_label_page(self):
        chunk = _make_chunk(page=5)
        assert chunk.citation_label() == "[AWS_Proposal.pdf, Page 5]"

    def test_citation_label_row(self):
        chunk = RetrievedChunk(
            text="row text", source_file="pricing.csv", vendor_name="Oracle",
            doc_category="pricing", page_number=None, row_index=12,
            sheet_name=None, chunk_index=3, score=0.8,
        )
        assert chunk.citation_label() == "[pricing.csv, Row 12]"

    def test_short_excerpt_truncates(self):
        long_text = "word " * 200
        chunk = _make_chunk(text=long_text)
        excerpt = chunk.short_excerpt(max_chars=50)
        assert len(excerpt) <= 53  # 50 + "…"
        assert excerpt.endswith("…")


# ── ReasoningAgent ────────────────────────────────────────────────────────────

class TestReasoningAgent:
    def test_returns_string(self):
        canned = "AWS offers 99.99% uptime [AWS_Proposal.pdf, Page 3]."
        agent = ReasoningAgent(llm_client=_mock_llm_client(canned))
        result = agent.reason("What is the uptime?", [_make_chunk()])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_chunks_returns_sentinel(self):
        agent = ReasoningAgent(llm_client=_mock_llm_client("irrelevant"))
        result = agent.reason("What is the uptime?", [])
        assert result == NOT_FOUND_SENTINEL

    def test_llm_called_with_context(self):
        client = _mock_llm_client("answer")
        agent = ReasoningAgent(llm_client=client)
        agent.reason("question", [_make_chunk()])
        client.chat.completions.create.assert_called_once()
        call_kwargs = client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        # System prompt + user message
        assert len(messages) == 2
        assert "CONTEXT CHUNKS" in messages[1]["content"]

    def test_comparative_mode_hint_in_prompt(self):
        client = _mock_llm_client("table answer")
        agent = ReasoningAgent(llm_client=client)
        agent.reason("compare pricing", [_make_chunk()], mode="comparative")
        call_kwargs = client.chat.completions.create.call_args[1]
        user_msg = call_kwargs["messages"][1]["content"]
        assert "table" in user_msg.lower() or "comparison" in user_msg.lower()

    def test_llm_failure_raises_runtime_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API timeout")
        agent = ReasoningAgent(llm_client=client)
        with pytest.raises(RuntimeError, match="LLM call failed"):
            agent.reason("question", [_make_chunk()])

    def test_empty_llm_response_returns_sentinel(self):
        agent = ReasoningAgent(llm_client=_mock_llm_client(""))
        result = agent.reason("question", [_make_chunk()])
        assert result == NOT_FOUND_SENTINEL


# ── ValidationAgent ───────────────────────────────────────────────────────────

class TestValidationAgent:
    def _good_answer(self):
        return (
            "AWS provides 99.99% uptime as stated in their proposal "
            "[AWS_Proposal.pdf, Page 3]. "
            "Oracle charges $520,000 annually [Schedule B Pricing-Oracle.pdf, Page 5]."
        )

    def _uncited_answer(self):
        return (
            "AWS provides 99.99% uptime as stated in their proposal. "
            "Oracle charges a lot of money every year."
        )

    def _chunks(self):
        return [
            _make_chunk(source="AWS_Proposal.pdf"),
            RetrievedChunk(
                text="Oracle annual pricing $520,000",
                source_file="Schedule B Pricing-Oracle.pdf",
                vendor_name="Oracle",
                doc_category="pricing",
                page_number=5,
                row_index=None,
                sheet_name=None,
                chunk_index=1,
                score=0.88,
            ),
        ]

    def test_cited_answer_passes(self):
        agent = ValidationAgent()
        _, count, passed = agent.validate(self._good_answer(), self._chunks())
        assert passed is True
        assert count == 2

    def test_uncited_answer_fails(self):
        agent = ValidationAgent()
        _, count, passed = agent.validate(self._uncited_answer(), self._chunks())
        assert passed is False

    def test_sentinel_always_passes(self):
        agent = ValidationAgent()
        _, count, passed = agent.validate(NOT_FOUND_SENTINEL, [])
        assert passed is True
        assert count == 0

    def test_rewrite_attempted_on_failure(self):
        reasoning = MagicMock()
        reasoning.reason.return_value = self._good_answer()
        agent = ValidationAgent()
        answer, count, passed = agent.validate(
            self._uncited_answer(),
            self._chunks(),
            reasoning_agent=reasoning,
            question="What are the prices?",
        )
        reasoning.reason.assert_called_once()
        # After rewrite the good answer should pass
        assert passed is True

    def test_citations_count_correct(self):
        agent = ValidationAgent()
        answer = (
            "Claim one [doc.pdf, Page 1]. "
            "Claim two [doc.pdf, Page 2]. "
            "Claim three [doc.pdf, Row 5]."
        )
        chunks = [_make_chunk(source="doc.pdf")]
        _, count, _ = agent.validate(answer, chunks)
        assert count == 3


# ── RequirementExtractionAgent ────────────────────────────────────────────────

class TestRequirementExtractionAgent:
    def _xlsx_bytes(self):
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Reqs"
        ws.append(["ID", "Requirement", "Category"])
        ws.append(["REQ-001", "99.9% uptime SLA required", "Availability"])
        ws.append(["REQ-002", "FedRAMP Authorization required", "Compliance"])
        ws.append(["REQ-003", "Data encrypted at rest", "Security"])
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    def _csv_bytes(self):
        return (
            b"requirement_id,description,category\n"
            b"1,Uptime 99.9%,Availability\n"
            b"2,Encryption at rest,Security\n"
        )

    def test_extract_from_xlsx(self):
        agent = RequirementExtractionAgent()
        reqs = agent.extract_requirements(self._xlsx_bytes(), "requirements.xlsx")
        assert len(reqs) == 3
        assert reqs[0].requirement_id == "REQ-001"
        assert "uptime" in reqs[0].description.lower()

    def test_extract_from_csv(self):
        agent = RequirementExtractionAgent()
        reqs = agent.extract_requirements(self._csv_bytes(), "requirements.csv")
        assert len(reqs) == 2

    def test_unsupported_format_raises(self):
        agent = RequirementExtractionAgent()
        with pytest.raises(ValueError, match="supports .csv and .xlsx only"):
            agent.extract_requirements(b"data", "requirements.pdf")

    def test_compliance_table_markdown(self):
        from app.agents.extractor import Requirement, ComplianceVerdict

        reqs = [
            Requirement("R1", "Uptime 99.9%", "Avail", "req.xlsx", 0),
            Requirement("R2", "FedRAMP", "Compliance", "req.xlsx", 1),
        ]
        verdicts = [
            ComplianceVerdict("R1", "AWS", "Pass", "evidence", "[doc, Page 1]"),
            ComplianceVerdict("R1", "Oracle", "Fail", "no evidence", ""),
            ComplianceVerdict("R2", "AWS", "Partially Met", "partial", "[doc, Page 2]"),
            ComplianceVerdict("R2", "Oracle", "Pass", "evidence", "[doc, Page 3]"),
        ]
        agent = RequirementExtractionAgent()
        table = agent.format_compliance_table(reqs, verdicts, ["AWS", "Oracle"])
        assert "| Req ID |" in table
        assert "Pass" in table
        assert "Fail" in table
        assert "Partially Met" in table


# ── AgentController ───────────────────────────────────────────────────────────

class TestAgentController:
    def test_classify_mode_comparative(self):
        for q in [
            "Compare annual pricing",
            "What is the difference between AWS and Oracle?",
            "Rank vendors by cost",
            "AWS vs Google",
        ]:
            assert classify_mode(q) == "comparative", f"Expected comparative for: {q!r}"

    def test_classify_mode_extractive(self):
        for q in [
            "What is the uptime SLA?",
            "Does AWS support FedRAMP?",
            "Summarize the AWS proposal",
        ]:
            assert classify_mode(q) == "extractive", f"Expected extractive for: {q!r}"

    def test_handle_returns_chat_response(self):
        from app.schemas.chat import ChatResponse

        retrieval = MagicMock()
        retrieval.retrieve.return_value = [_make_chunk()]

        reasoning = MagicMock()
        reasoning.reason.return_value = (
            "AWS provides 99.99% uptime [AWS_Proposal.pdf, Page 3]."
        )

        validation = MagicMock()
        validation.validate.return_value = (
            "AWS provides 99.99% uptime [AWS_Proposal.pdf, Page 3].", 1, True
        )

        controller = AgentController(
            retrieval_agent=retrieval,
            reasoning_agent=reasoning,
            validation_agent=validation,
        )
        request = ChatRequest(question="What is the AWS uptime?")
        response = controller.handle(request)

        assert isinstance(response, ChatResponse)
        assert response.validation_passed is True
        assert response.citations_count == 1
        assert len(response.sources) == 1

    def test_handle_empty_db_returns_sentinel(self):
        retrieval = MagicMock()
        retrieval.retrieve.return_value = []

        controller = AgentController(retrieval_agent=retrieval)
        request = ChatRequest(question="Any question")
        response = controller.handle(request)

        assert response.answer == NOT_FOUND_SENTINEL
        assert response.sources == []
        assert response.validation_passed is True

    def test_vendor_filter_forwarded_to_retrieval(self):
        retrieval = MagicMock()
        retrieval.retrieve.return_value = []

        controller = AgentController(retrieval_agent=retrieval)
        request = ChatRequest(
            question="AWS pricing", vendor_filter=["AWS"]
        )
        controller.handle(request)

        retrieval.retrieve.assert_called_once()
        _, kwargs = retrieval.retrieve.call_args
        assert kwargs.get("vendor_filter") == ["AWS"]

    def test_mode_auto_detected_from_keywords(self):
        retrieval = MagicMock()
        retrieval.retrieve.return_value = [_make_chunk()]
        reasoning = MagicMock()
        reasoning.reason.return_value = "table [AWS_Proposal.pdf, Page 3]."
        validation = MagicMock()
        validation.validate.return_value = ("table [AWS_Proposal.pdf, Page 3].", 1, True)

        controller = AgentController(
            retrieval_agent=retrieval,
            reasoning_agent=reasoning,
            validation_agent=validation,
        )
        request = ChatRequest(question="Compare pricing across all vendors")
        response = controller.handle(request)
        assert response.mode == "comparative"

        # Check reasoning was called with mode="comparative"
        reasoning.reason.assert_called_once()
        _, kwargs = reasoning.reason.call_args
        assert kwargs.get("mode") == "comparative"
