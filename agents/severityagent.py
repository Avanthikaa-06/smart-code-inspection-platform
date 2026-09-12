"""
SeverityAgent
-------------
Reusable agent responsible only for severity normalization.

It accepts already-merged findings from the orchestrator, preserves each
tool-assigned severity in ``original_severity``, and optionally asks Gemini
to normalize severity, score, priority, reason, and recommendation in batches.
If Gemini is disabled or fails, the original tool severity is retained.
"""

import json
from typing import List, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import llm_router

VALID_SEVERITIES = {"Critical", "High", "Medium", "Low", "Info"}
FALLBACK_SCORE_PRIORITY = {
    "Critical": (95, 1),
    "High": (80, 2),
    "Medium": (55, 3),
    "Low": (30, 4),
    "Info": (10, 5),
}


class SeverityAnalysisItem(BaseModel):
    index: int = Field(..., description="Finding index from the request batch")
    severity: Literal["Critical", "High", "Medium", "Low", "Informational"] = Field(
        ...,
        description="Normalized severity",
    )
    score: int = Field(..., ge=0, le=100, description="Risk score from 0 to 100")
    priority: int = Field(..., ge=1, le=5, description="Remediation priority from 1 to 5")
    reason: str = Field(..., description="Why this severity was assigned")
    recommendation: str = Field(..., description="Concrete remediation recommendation")


class SeverityAnalysisBatch(BaseModel):
    findings: List[SeverityAnalysisItem] = Field(default_factory=list)


def _normalize_severity(value: object, default: str = "Medium") -> str:
    raw = str(value or default).strip()
    normalized = raw.capitalize()
    if raw.lower() in ("informational", "info"):
        normalized = "Info"
    return normalized if normalized in VALID_SEVERITIES else default


def _clamp_int(value: object, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _fallback_enrich(finding: dict, reason: Optional[str] = None) -> dict:
    enriched = dict(finding)
    original = _normalize_severity(enriched.get("original_severity") or enriched.get("severity"))
    score, priority = FALLBACK_SCORE_PRIORITY.get(original, FALLBACK_SCORE_PRIORITY["Medium"])
    enriched["original_severity"] = original
    enriched["severity"] = original
    enriched["score"] = _clamp_int(enriched.get("score"), 0, 100, score)
    enriched["priority"] = _clamp_int(enriched.get("priority"), 1, 5, priority)
    enriched["reason"] = enriched.get("reason") or reason or "Retained original tool severity."
    enriched["recommendation"] = (
        enriched.get("recommendation")
        or "Review and remediate this finding before release."
    )
    return enriched


def _line_context(code: str, line: int, radius: int = 2) -> Optional[str]:
    try:
        lines = (code or "").splitlines()
        if not lines:
            return None
        start = max(1, int(line or 1) - radius)
        end = min(len(lines), int(line or 1) + radius)
        return "\n".join(f"{idx:>4} {lines[idx - 1]}" for idx in range(start, end + 1))
    except Exception:
        return None


def _finding_payload(finding: dict, index: int, code: str) -> dict:
    line = int(finding.get("line_start") or finding.get("line") or 1)
    return {
        "index": index,
        "tool": finding.get("tool"),
        "detected_by": finding.get("detected_by") or [finding.get("tool")],
        "agent": finding.get("agent") or finding.get("agent_source"),
        "language": finding.get("language"),
        "line": line,
        "title": finding.get("title") or finding.get("category"),
        "description": finding.get("description") or finding.get("message"),
        "cwe": finding.get("cwe_id") or finding.get("cwe"),
        "owasp": finding.get("owasp_category") or finding.get("owasp"),
        "confidence": finding.get("confidence"),
        "original_severity": finding.get("original_severity") or finding.get("severity"),
        "existing_recommendation": finding.get("recommendation"),
        "code_context": finding.get("code_snippet") or _line_context(code, line),
    }


def _chunks(items: list, size: int) -> list:
    return [items[index:index + size] for index in range(0, len(items), size)]


# Prompt only — no LLM bound here. llm_router.run_structured() binds this
# to whichever provider (Gemini primary, Groq backup) ends up handling the
# call, so the prompt/schema contract stays identical across both.
_SEVERITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the Severity Agent for an AI code review and security analysis pipeline. "
        "Your only job is to normalize finding severity. Analyze each finding independently "
        "using CWE, OWASP impact, exploitability, business impact, confidence, and code context. "
        "If a finding's 'detected_by' list contains more than one tool, treat that independent "
        "agreement as a signal of higher confidence and higher priority. "
        "Do not use fixed tool-name rules. Return one result per input finding index. "
        "Severity must be one of: Critical, High, Medium, Low, Informational. "
        "Score is 0-100 where 100 is most severe. Priority is 1-5 where 1 is most urgent."
    )),
    ("human", (
        "Language: {language}\n\n"
        "Analyze these findings as a batch and return normalized severity data for each:\n"
        "{findings_json}"
    )),
])


def _apply_analysis(finding: dict, item: SeverityAnalysisItem) -> dict:
    enriched = dict(finding)
    enriched["original_severity"] = _normalize_severity(
        enriched.get("original_severity") or enriched.get("severity")
    )
    enriched["severity"] = _normalize_severity(item.severity)
    enriched["score"] = _clamp_int(item.score, 0, 100, 50)
    enriched["priority"] = _clamp_int(item.priority, 1, 5, 3)
    enriched["reason"] = (item.reason or "").strip() or "Gemini normalized this finding severity."
    enriched["recommendation"] = (
        (item.recommendation or "").strip()
        or enriched.get("recommendation")
        or "Review and remediate this finding before release."
    )
    return enriched


def analyze(
    findings: list,
    code: str,
    language: str,
    api_key: str = "",
    use_llm: bool = True,
    batch_size: int = 20,
) -> list:
    """
    Normalize severities for merged findings.

    The first three parameters form the stable agent interface requested by
    the orchestrator. ``use_llm`` still gates whether the LLM path is
    attempted at all. ``api_key`` is kept in the signature for backward
    compatibility with existing callers but is no longer required or used
    directly — credentials are now sourced server-side via ``llm_router``,
    which also handles automatic Gemini -> Groq failover, so this agent
    doesn't need to know which provider actually served the request.
    """
    prepared = [_fallback_enrich(dict(item)) for item in (findings or [])]
    if not prepared or not use_llm or not llm_router.any_configured():
        return prepared

    output = list(prepared)
    effective_batch_size = max(1, int(batch_size or 20))

    for batch in _chunks(list(enumerate(prepared)), effective_batch_size):
        payload = [
            _finding_payload(finding, index, code)
            for index, finding in batch
        ]
        try:
            response: SeverityAnalysisBatch = llm_router.run_structured(
                SeverityAnalysisBatch,
                _SEVERITY_PROMPT,
                {
                    "language": (language or "").lower(),
                    "findings_json": json.dumps(payload, ensure_ascii=False, indent=2),
                },
                temperature=0,
            )
            by_index = {item.index: item for item in response.findings}
            for index, finding in batch:
                item = by_index.get(index)
                output[index] = _apply_analysis(finding, item) if item else _fallback_enrich(
                    finding,
                    "Neither LLM provider returned severity data for this finding.",
                )
        except Exception as exc:
            for index, finding in batch:
                output[index] = _fallback_enrich(
                    finding,
                    f"LLM severity analysis failed on both providers; retained original tool severity: {exc}",
                )

    return output