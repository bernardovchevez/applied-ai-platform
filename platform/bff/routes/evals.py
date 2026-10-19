"""GET /v1/evals/summary."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from deps import verify_bff_api_key
from evals_summary import build_eval_summary, eval_summary_has_data
from models import EvalBackendScore, EvalSummaryResponse

router = APIRouter(dependencies=[Depends(verify_bff_api_key)])


@router.get("/v1/evals/summary", response_model=EvalSummaryResponse)
def evals_summary() -> EvalSummaryResponse:
    data = build_eval_summary()
    if not eval_summary_has_data(data):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "evals_not_found",
                "message": "No eval artifacts found under rag/evals/ or agents/evals/",
            },
        )
    agent_eval = data.get("agent_eval")
    return EvalSummaryResponse(
        gold_questions=data["gold_questions"],
        agent_scenarios=data.get("agent_scenarios") or 0,
        keyword_eval=data["keyword_eval"],
        keyword_eval_timestamp=data.get("keyword_eval_timestamp"),
        ragas_eval=data.get("ragas_eval") or [],
        ragas_eval_timestamp=data.get("ragas_eval_timestamp"),
        ragas_judge_model=data.get("ragas_judge_model"),
        agent_eval=EvalBackendScore(
            backend=agent_eval["backend"],
            passed=agent_eval["passed"],
            total=agent_eval["total"],
            note=agent_eval.get("note"),
        )
        if agent_eval
        else None,
        agent_eval_timestamp=agent_eval.get("timestamp") if agent_eval else None,
        promptfoo=data.get("promptfoo"),
        sources=data.get("sources"),
    )
