"""Unit and contract tests for Applied AI Platform (config, RAG guards, workflow guards, and BFF API)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in (REPO_ROOT, REPO_ROOT / "rag", REPO_ROOT / "platform", REPO_ROOT / "agents"):
    p = str(sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.config import app_env, is_local, is_production, is_stage, state_store_backend
from shared.embeddings import active_embedding_model_id, embedding_dimensions, embedding_provider
from shared.llm import llm_provider
from shared.rag_text import clean_rag_answer, normalize_chunk_text, rag_question_hints
from workflow import (
    AgentResult,
    answer_contradicts_lookup,
    apply_auto_escalation,
    apply_clearance_grounding_guard,
    contains_general_knowledge_leak,
    should_force_refusal,
)


class TestEnvironmentAndProviders:
    def test_app_env_normalization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "localhost")
        assert app_env() == "localhost"
        assert is_local()

        monkeypatch.setenv("APP_ENV", "staging")
        assert app_env() == "stage"
        assert is_stage()
        assert not is_local()

        monkeypatch.setenv("APP_ENV", "prod")
        assert app_env() == "production"
        assert is_production()
        assert not is_local()

    def test_state_store_backend_defaults_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("STATE_STORE_BACKEND", raising=False)
        monkeypatch.setenv("APP_ENV", "localhost")
        assert state_store_backend() == "sqlite"

        monkeypatch.setenv("APP_ENV", "stage")
        assert state_store_backend() == "postgres"

        monkeypatch.setenv("APP_ENV", "production")
        assert state_store_backend() == "postgres"

        monkeypatch.setenv("STATE_STORE_BACKEND", "sqlite")
        assert state_store_backend() == "sqlite"

    def test_cloud_env_defaults_to_openai_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        assert llm_provider() == "openai"
        assert embedding_provider() == "openai"
        assert embedding_dimensions() == 1536
        assert "1536" in active_embedding_model_id()


class TestRagTextNormalizationAndGuards:
    def test_normalize_chunk_text_expands_hdr_and_perpetuity(self) -> None:
        raw = "Deliverable requires HDR pass and Perpetuity (series term)."
        normalized = normalize_chunk_text(raw)
        assert "HDR (High Dynamic Range) pass" in normalized
        assert "not EMEA expiry" in normalized

    def test_rag_question_hints_adds_emea_and_hdr_guidance(self) -> None:
        hints = rag_question_hints("Is EMEA cleared and does it require an HDR grade?")
        assert "2027-12-31" in hints
        assert "HDR pass" in hints

    def test_clean_rag_answer_fixes_local_model_garbles_and_appends_sources(self) -> None:
        raw = 'The memo states that ss is not required and has an etuity (series term) cleared for EMEA.'
        cleaned = clean_rag_answer(raw, source_files=["music-clearance-emea.md"])
        assert "HDR pass is not required" in cleaned
        assert "2027-12-31" in cleaned
        assert "Sources:\n- music-clearance-emea.md" in cleaned


class TestWorkflowGroundingGuards:
    def test_general_knowledge_leak_triggers_forced_refusal(self) -> None:
        assert contains_general_knowledge_leak("Ted Sarandos is the current CEO of Netflix.")
        assert should_force_refusal(
            "I don't have enough information in the provided documents.",
            "Ted Sarandos is the current CEO of Netflix.",
        )

    def test_clearance_grounding_guard_overrides_contradiction(self) -> None:
        lookup = {
            "series": "Harbor Line",
            "territory": "Japan",
            "territory_normalized": "apac",
            "status": "pending",
            "confidence": 0.35,
            "needs_escalation": True,
            "clearance_records": [{"source": "apac-music-clearance-update.md"}],
        }
        assert answer_contradicts_lookup("Harbor Line is cleared for Japan.", lookup)

        res = AgentResult(answer="Harbor Line is cleared for Japan.", lookup_rights=lookup)
        res = apply_clearance_grounding_guard(res)
        assert res.clearance_grounding_override is True
        assert "not cleared" in res.answer.lower()

        res = apply_auto_escalation(res)
        assert res.auto_escalated is True
        assert res.escalation is not None
        assert "Legal escalation (auto-triggered)" in res.answer


class TestBffStateAndContract:
    def test_sqlite_agent_sessions_and_feedback_persistence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STATE_STORE_BACKEND", "sqlite")
        from bff.agent_sessions import get_session, resolve_session, save_pending_session
        from bff.feedback import insert_feedback, list_feedback

        sess_db = tmp_path / "agent_sessions.db"
        tid = save_pending_session(
            thread_id="thread-123",
            prompt="Clearance for EMEA?",
            proposed_output="Draft EMEA summary",
            result_payload={
                "sources": [{"file_name": "music-clearance-emea.md"}],
                "tool_trace": [{"tool": "lookup_rights"}],
            },
            client_platform="web",
            db_path=sess_db,
        )
        assert tid == "thread-123"
        loaded = get_session("thread-123", db_path=sess_db)
        assert loaded is not None
        assert loaded["prompt"] == "Clearance for EMEA?"
        assert loaded["status"] == "pending"

        resolved = resolve_session("thread-123", "approve", db_path=sess_db)
        assert resolved is not None
        assert resolved["status"] == "approved"
        assert resolve_session("thread-123", "approve", db_path=sess_db) is None

        fb_db = tmp_path / "feedback.db"
        saved = insert_feedback(
            session_id="sess-1",
            prompt="Is EMEA cleared?",
            response="Yes, through 2027-12-31.",
            rating="up",
            sources=[{"file_name": "music-clearance-emea.md"}],
            prompt_version="abc123",
            client_platform="web",
            db_path=fb_db,
        )
        assert saved["id"]
        rows = list_feedback(limit=5, db_path=fb_db)
        assert len(rows) == 1
        assert rows[0]["rating"] == "up"

    def test_bff_endpoints_and_optional_api_key_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_ENV", "localhost")
        monkeypatch.setenv("STATE_STORE_BACKEND", "sqlite")
        monkeypatch.delenv("BFF_API_KEY", raising=False)

        from bff.main import app

        client = TestClient(app)

        ver = client.get("/v1/version")
        assert ver.status_code == 200
        body = ver.json()
        assert body["environment"] == "localhost"
        assert body["state_store"] == "sqlite"
        assert "X-Environment" in ver.headers

        spec = client.get("/openapi.yaml")
        assert spec.status_code == 200
        assert "openapi: 3.1.0" in spec.text

        summary = client.get("/v1/evals/summary")
        assert summary.status_code == 200
        assert summary.json()["gold_questions"] >= 20

        # Enforce BFF_API_KEY when set in stage/production
        monkeypatch.setenv("BFF_API_KEY", "test-secret-key")
        unauth = client.get("/v1/evals/summary")
        assert unauth.status_code == 401

        auth = client.get(
            "/v1/evals/summary",
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert auth.status_code == 200
