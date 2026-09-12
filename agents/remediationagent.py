"""
RemediationAgent
------------------
Reusable agent responsible only for remediation generation.

It accepts already severity-scored findings from the orchestrator (the
same shape severityagent.analyze() returns) and, for each one, produces a
grounded, minimal code fix:

    1. Understand Detected Issues   -> full finding context is sent as input,
                                        not just the code snippet (severity,
                                        score, priority, reason, existing
                                        recommendation all included)
    2. Find the Root Cause          -> root_cause
    3. Retrieve Secure Guidelines   -> retrieved from rag_engine's knowledge
                                        base when available (see
                                        _retrieve_guidance), not just asked
                                        of Gemini's own training knowledge
    4. Generate Minimal Code Fixes  -> after_snippet only; before_snippet is
                                        taken verbatim from the finding's own
                                        code_snippet, never from the LLM, so
                                        the diff can never drift from ground
                                        truth
    5. Explain Why the Fix Works    -> explanation, citing the retrieved
                                        guideline when one was found
    6. Validate & Recommend Safe    -> syntax-validated locally, before/after
       Application                    compared to catch no-op "fixes", then
                                        tagged into three tiers

Nothing here is a hardcoded fix table (no "if CWE-327: replace md5 with
sha256" style mapping) for the AI-generated fix path. Every after_snippet
that is offered as an "AI Generated Fix" is produced by the LLM reasoning
over the specific finding's own code context and (when available) real
retrieved guideline text; this file's job is to prompt it correctly, ground
it, validate what comes back, and never show an unvalidated, no-op, or
ungrounded fix as if it were safe to trust.

--------------------------------------------------------------------------
Fallback behavior (never show an empty/placeholder remediation card)
--------------------------------------------------------------------------
Every field a remediation card needs (before snippet, after snippet,
explanation, guideline reference, prevention tip) is now guaranteed to be
populated, whether or not the LLM produced it - and whether or not the LLM
ran at all (no API key, invalid key, or a failed call). None of these
fallbacks are presented as an exact automatic fix; each is clearly tagged
with a "fix_tier" so the UI can visually distinguish:

    ai_generated     -> the LLM produced and validated an exact fix
    example_template -> a secure-coding example, not an exact fix, because
                         the LLM could not safely rewrite the code (or the
                         LLM was unavailable)
    manual_review    -> no code context could be found for this finding at
                         all, so only guidance (no before/after diff) can
                         be shown

If no API key is provided (or it's invalid), the LLM path is skipped, but
remediation is still generated locally (fallback guidance + example
template) instead of leaving the finding without remediation - the rest of
the pipeline (severity, findings list, PDF) is unaffected, matching how
severityagent.py degrades when the LLM is off.
"""

import ast
import json
import re
import textwrap
from typing import List, Literal, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import llm_router


# --------------------------------------------------------------------------- #
# Structured output contract (forces the LLM into the 6-responsibility shape
# instead of free-text advice). Note: before_snippet is deliberately NOT a
# field here - see _resolve_before_snippet for why.
# --------------------------------------------------------------------------- #
class RemediationItem(BaseModel):
    index: int = Field(..., description="Finding index from the request batch")
    root_cause: str = Field(..., description="The underlying reason the issue exists, not just the symptom")
    guideline_reference: str = Field(
        ...,
        description=(
            "The specific secure-coding standard this fix follows, e.g. "
            "'OWASP A03:2021 Injection' or 'CWE-89'. If retrieved_guidance was "
            "provided for this finding, ground this in that text and mention "
            "its source. If none was provided, use your own knowledge and say "
            "so explicitly (e.g. 'No local knowledge base match; using general "
            "OWASP guidance')."
        ),
    )
    after_snippet: str = Field(
        ...,
        description=(
            "The minimal corrected replacement for the vulnerable lines only. "
            "Must preserve the original variable names, formatting, indentation, "
            "comments, and surrounding logic - change only what is necessary to "
            "resolve the root cause. Never rewrite the entire function."
        ),
    )
    explanation: str = Field(..., description="Why this specific fix resolves the root cause")
    prevention_tip: str = Field(..., description="A general habit that prevents this class of issue in future code")
    application_confidence: Literal["auto-applicable", "needs-review", "manual-review-required"] = Field(
        ...,
        description=(
            "'auto-applicable': mechanical, behavior-preserving change (e.g. swapping a hash "
            "algorithm, adding an escape call) with no logic change.\n"
            "'needs-review': the fix changes logic or control flow but is straightforward and "
            "low-risk (e.g. switching to a parameterized query).\n"
            "'manual-review-required': the fix is complex, high-risk, or touches authentication, "
            "cryptography, access control, or architecture - a developer must design/verify it, "
            "not just approve it."
        ),
    )


class RemediationBatch(BaseModel):
    remediations: List[RemediationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Local validation helpers - a fix is never shown unless it at least parses.
# This is deliberately separate from "is this fix correct" (which the LLM
# reasoning is responsible for) - it only catches "is this fix not broken".
# --------------------------------------------------------------------------- #
def _validate_python_snippet(snippet: str) -> Tuple[bool, Optional[str]]:
    candidates = [snippet, textwrap.dedent(snippet)]
    for candidate in candidates:
        try:
            ast.parse(candidate)
            return True, None
        except SyntaxError:
            continue
    # Last resort: the snippet may be a valid statement that only fails
    # standalone because it relies on surrounding indentation context
    # (e.g. it's meant to sit inside a function/try block).
    try:
        body = "\n".join(f"    {line}" for line in textwrap.dedent(snippet).splitlines()) or "    pass"
        ast.parse(f"def _wrapper():\n{body}")
        return True, None
    except SyntaxError as exc:
        return False, str(exc)


def _validate_java_snippet(snippet: str) -> Tuple[bool, Optional[str]]:
    try:
        import javalang
    except ImportError:
        return False, "javalang is not installed; could not validate Java syntax."

    # A fix snippet is a fragment (one or two lines), not a full compilation
    # unit, so a full javalang.parse.parse() isn't applicable here. Token-
    # level lexing still catches the common breakage: unterminated strings,
    # mismatched brackets, invalid characters.
    try:
        list(javalang.tokenizer.tokenize(snippet))
        return True, None
    except Exception as exc:
        return False, str(exc)


def _validate_snippet(snippet: str, language: str) -> Tuple[bool, Optional[str]]:
    if not snippet or not snippet.strip():
        return False, "Empty fix snippet."
    lang = (language or "").strip().lower()
    if lang == "python":
        return _validate_python_snippet(snippet)
    if lang == "java":
        return _validate_java_snippet(snippet)
    return False, f"No syntax validator available for language '{language}'."


def _normalize_for_comparison(text: str) -> str:
    """Whitespace-insensitive normalization so a fix that only differs by
    indentation/trailing spaces doesn't count as 'no real change'."""
    return "\n".join(line.strip() for line in (text or "").strip().splitlines() if line.strip())


def _is_noop_fix(before: str, after: str) -> bool:
    return _normalize_for_comparison(before) == _normalize_for_comparison(after)


# --------------------------------------------------------------------------- #
# BEFORE SNIPPET fallback chain
# --------------------------------------------------------------------------- #
def _nearby_lines_from_source(code: str, line: Optional[int], line_start: Optional[int],
                               line_end: Optional[int], radius: int = 3) -> str:
    """Extract a small window of the originally submitted source around the
    finding's reported line(s), used only when the finding itself carries no
    code_snippet / code_context at all."""
    lines = (code or "").splitlines()
    if not lines:
        return ""
    start_line = line_start or line or 1
    end_line = line_end or start_line
    try:
        start_line = int(start_line)
        end_line = int(end_line)
    except (TypeError, ValueError):
        return ""
    start = max(1, start_line - radius)
    end = min(len(lines), end_line + radius)
    if start > len(lines) or end < 1:
        return ""
    return "\n".join(f"{idx:>4} {lines[idx - 1]}" for idx in range(start, end + 1))


def _resolve_before_snippet(finding: dict, code: str) -> Tuple[str, str]:
    """
    Tries, in order:
      1. finding['code_snippet']
      2. finding['code_context']
      3. lines extracted from the originally submitted source using
         line / line_start / line_end
      4. "No code context available for this finding." (never a bare
         placeholder like "(not available)")

    Returns (text, source) where source is one of:
      "code_snippet", "code_context", "extracted", "unavailable"
    """
    code_snippet = finding.get("code_snippet")
    if code_snippet and str(code_snippet).strip():
        return str(code_snippet), "code_snippet"

    code_context = finding.get("code_context")
    if code_context and str(code_context).strip():
        return str(code_context), "code_context"

    extracted = _nearby_lines_from_source(
        code,
        finding.get("line"),
        finding.get("line_start"),
        finding.get("line_end"),
    )
    if extracted.strip():
        return extracted, "extracted"

    return "No code context available for this finding.", "unavailable"


# --------------------------------------------------------------------------- #
# AFTER SNIPPET fallback - a labeled secure-coding example, never a bare
# "(no fix generated)" placeholder, and never presented as an exact fix.
# --------------------------------------------------------------------------- #
_TEMPLATE_GUIDANCE = {
    "sql_injection": "Use parameterized queries / prepared statements instead of concatenating user input.",
    "os_command_injection": "Avoid shell execution. Use ProcessBuilder or dedicated APIs and validate input.",
    "hardcoded_secret": "Move secrets into environment variables or a secure secret manager.",
    "weak_hash": "Use SHA-256 or stronger cryptographic algorithms.",
    "insecure_deserialization": "Avoid unsafe deserialization. Use trusted formats or validate serialized input.",
    "generic": "Apply input validation, output encoding, and least-privilege access appropriate to this finding.",
}


def _match_vuln_category(finding: dict) -> str:
    haystack = " ".join(
        str(finding.get(key) or "")
        for key in ("title", "category", "description", "message", "cwe_id", "cwe")
    ).lower()

    if "sql injection" in haystack or re.search(r"cwe-89\b", haystack):
        return "sql_injection"
    if "command injection" in haystack or "os command" in haystack or re.search(r"cwe-78\b", haystack):
        return "os_command_injection"
    if ("hardcoded" in haystack and ("secret" in haystack or "credential" in haystack or "password" in haystack)) \
            or re.search(r"cwe-798\b", haystack):
        return "hardcoded_secret"
    if "weak hash" in haystack or "md5" in haystack or "sha1" in haystack or "sha-1" in haystack \
            or re.search(r"cwe-32[78]\b", haystack):
        return "weak_hash"
    if "deserializ" in haystack or re.search(r"cwe-502\b", haystack):
        return "insecure_deserialization"
    return "generic"


def _build_example_template(finding: dict) -> str:
    category = _match_vuln_category(finding)
    guidance = _TEMPLATE_GUIDANCE[category]
    return (
        "# Example secure implementation\n"
        f"# {guidance}\n"
        "# This is a secure coding pattern to apply to the vulnerable lines above,\n"
        "# not an exact automatic fix - adapt it to the surrounding code."
    )


# --------------------------------------------------------------------------- #
# WHY THIS FIX WORKS fallback
# --------------------------------------------------------------------------- #
def _fallback_explanation(finding: dict) -> str:
    title = finding.get("title") or finding.get("category") or "this issue"
    return (
        f"This recommendation removes the vulnerable pattern behind '{title}' and follows "
        "secure coding best practices for preventing this class of issue."
    )


# --------------------------------------------------------------------------- #
# SECURE CODING GUIDELINE fallback
# --------------------------------------------------------------------------- #
def _fallback_guideline_reference(finding: dict) -> str:
    cwe = finding.get("cwe_id") or finding.get("cwe")
    owasp = finding.get("owasp_category") or finding.get("owasp")
    refs = [str(ref) for ref in (owasp, cwe) if ref]
    if refs:
        return "Based on industry secure coding guidance: " + " | ".join(refs)
    return "General secure coding best practices."


# --------------------------------------------------------------------------- #
# PREVENTION TIP fallback
# --------------------------------------------------------------------------- #
def _fallback_prevention_tip(finding: dict) -> str:
    title = finding.get("title") or finding.get("category") or "this class of issue"
    ref = finding.get("cwe_id") or finding.get("cwe") or finding.get("owasp_category") or finding.get("owasp")
    if ref:
        return f"Add {ref} ({title}) checks to routine code review and CI static analysis to catch this earlier."
    return f"Add {title} checks to routine code review and CI static analysis to catch this class of issue earlier."


# --------------------------------------------------------------------------- #
# VISUAL INDICATOR badges - three distinct, never-identical tiers
# --------------------------------------------------------------------------- #
_BADGE = {
    "ai_generated": "\u2713 AI Generated Fix",
    "example_template": "\u26a0 Example Secure Implementation",
    "manual_review": "\u26a0 Manual Review Required",
}


# --------------------------------------------------------------------------- #
# RAG grounding - reuses the project's existing rag_engine.RAGPipeline
# instead of relying only on Gemini's own training knowledge.
# --------------------------------------------------------------------------- #
def _retrieve_guidance(rag_pipeline, finding: dict, top_k: int = 2) -> List[dict]:
    """
    Queries the existing Secure Coding Knowledge Base for passages relevant
    to this finding. Returns [] (never raises) if no pipeline was supplied,
    the index isn't built yet, or the query fails for any reason - this is
    an enrichment step, not a required one, so it must never block
    remediation from running.
    """
    if rag_pipeline is None:
        return []
    try:
        if hasattr(rag_pipeline, "store") and not rag_pipeline.store.is_ready():
            return []
        query_parts = [
            finding.get("title") or finding.get("category") or "",
            finding.get("cwe_id") or finding.get("cwe") or "",
            finding.get("owasp_category") or finding.get("owasp") or "",
        ]
        query = " ".join(part for part in query_parts if part).strip()
        if not query:
            return []
        result = rag_pipeline.query(query, top_k=top_k)
        if not result.get("success"):
            return []
        return [
            {"text": item.get("text", ""), "source": item.get("source", "unknown")}
            for item in result.get("results", [])
            if item.get("text")
        ]
    except Exception:
        # Knowledge base being unavailable should never take down
        # remediation generation - fall through to Gemini's own knowledge.
        return []


# --------------------------------------------------------------------------- #
# Batch request payload - includes everything severityagent already derived
# (severity, score, priority, reason, recommendation) plus retrieved
# knowledge-base guidance, so the model reasons with full context instead
# of just the code snippet.
# --------------------------------------------------------------------------- #
def _finding_payload(finding: dict, index: int, retrieved_guidance: List[dict]) -> dict:
    return {
        "index": index,
        "title": finding.get("title") or finding.get("category"),
        "description": finding.get("description") or finding.get("message"),
        "severity": finding.get("severity"),
        "score": finding.get("score"),
        "priority": finding.get("priority"),
        "reason": finding.get("reason"),
        "recommendation": finding.get("recommendation"),
        "cwe": finding.get("cwe_id") or finding.get("cwe"),
        "owasp": finding.get("owasp_category") or finding.get("owasp"),
        "language": finding.get("language"),
        "line": finding.get("line") or finding.get("line_start"),
        "code_context": finding.get("code_snippet"),
        "retrieved_guidance": retrieved_guidance,
    }


def _chunks(items: list, size: int) -> list:
    return [items[index:index + size] for index in range(0, len(items), size)]


# Prompt only — no LLM bound here. llm_router.run_structured() binds this
# to whichever provider (Gemini primary, Groq backup) ends up handling the
# call, so the prompt/schema contract stays identical across both.
_REMEDIATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the Remediation Agent for an AI code review and security analysis pipeline. "
        "For every finding you are given, you must:\n"
        "1. Understand the detected issue using its full context - severity, score, priority, "
        "the reason it was flagged, and any existing tool recommendation.\n"
        "2. Identify the root cause, not just the symptom.\n"
        "3. Ground your fix in a real secure-coding guideline. If the finding includes "
        "'retrieved_guidance' (passages retrieved from the project's own knowledge base), you "
        "MUST base guideline_reference on that retrieved text and cite its source. Only fall "
        "back to your own general OWASP/CWE/CERT knowledge when retrieved_guidance is empty, "
        "and say so explicitly when you do.\n"
        "4. Produce a MINIMAL fix as after_snippet: modify ONLY the vulnerable lines. The fix "
        "must preserve the original variable names, formatting, indentation, comments, and "
        "surrounding logic exactly as given. Never rewrite the entire function or restructure "
        "unrelated code. Do not return a before_snippet - the exact original code is already "
        "known; only generate the corrected replacement.\n"
        "5. Explain why the fix resolves the root cause, referencing the guideline.\n"
        "6. Classify application_confidence into exactly one of three tiers: 'auto-applicable' "
        "for mechanical, behavior-preserving changes only; 'needs-review' for straightforward "
        "logic changes; 'manual-review-required' for anything touching authentication, "
        "cryptography, access control, or architecture. If you cannot produce a code_context-"
        "grounded fix, return an empty after_snippet and mark it 'manual-review-required'.\n"
        "Return exactly one remediation per input finding index."
    )),
    ("human", (
        "Language: {language}\n\n"
        "Generate remediations for these findings:\n{findings_json}"
    )),
])


def _apply_remediation(finding: dict, item: RemediationItem, language: str,
                        retrieved_guidance: List[dict], code: str) -> dict:
    enriched = dict(finding)

    # before_snippet ALWAYS comes from the finding/original source, never
    # from the LLM - this guarantees the diff shown to the user is accurate
    # even if the model's own recollection of the "before" state drifts,
    # and it is never left as a bare "(not available)" placeholder.
    before_snippet, before_source = _resolve_before_snippet(finding, code)

    llm_after_snippet = (item.after_snippet or "").strip()
    confidence = item.application_confidence
    validation_notes = []

    if llm_after_snippet:
        is_valid, validation_error = _validate_snippet(llm_after_snippet, language)
        if not is_valid:
            confidence = "needs-review"
            validation_notes.append(f"Fix failed local syntax validation and was downgraded: {validation_error}")

        if before_source != "unavailable" and _is_noop_fix(before_snippet, llm_after_snippet):
            confidence = "needs-review"
            validation_notes.append("Generated fix is identical to the original code; no actual change was made.")

        after_snippet = llm_after_snippet
        after_source = "ai_generated"
        fix_tier = "ai_generated"
    else:
        is_valid = False
        confidence = "manual-review-required"
        validation_notes.append(
            "The model could not safely generate an exact fix; showing an example secure "
            "implementation instead."
        )
        after_snippet = _build_example_template(finding)
        after_source = "template"
        fix_tier = "example_template"

    # No code context at all overrides everything else - there is nothing
    # concrete to show a diff against, so this is a manual-review case
    # regardless of what the LLM returned.
    if before_source == "unavailable":
        fix_tier = "manual_review"

    explanation = (item.explanation or "").strip() or _fallback_explanation(finding)
    guideline_reference = (item.guideline_reference or "").strip() or _fallback_guideline_reference(finding)
    prevention_tip = (item.prevention_tip or "").strip() or _fallback_prevention_tip(finding)

    enriched["remediation"] = {
        "root_cause": (item.root_cause or "").strip() or "Root cause could not be automatically determined from the available context.",
        "guideline_reference": guideline_reference,
        "retrieved_guidance": retrieved_guidance,
        "before_snippet": before_snippet,
        "before_snippet_source": before_source,
        "after_snippet": after_snippet,
        "after_snippet_source": after_source,
        "explanation": explanation,
        "prevention_tip": prevention_tip,
        "application_confidence": confidence,
        "syntax_valid": is_valid,
        "validation_note": "; ".join(validation_notes) or None,
        "fix_tier": fix_tier,
        "badge": _BADGE.get(fix_tier, _BADGE["manual_review"]),
    }
    enriched["remediation_unavailable_reason"] = None
    return enriched


# --------------------------------------------------------------------------- #
# Fully local fallback - used whenever the LLM path did not produce a
# remediation for this finding at all (no API key, invalid key, chain build
# failure, batch call failure, or the LLM simply omitted this index).
# Still returns a complete, non-empty remediation instead of remediation=None,
# so the remediation page never appears incomplete.
# --------------------------------------------------------------------------- #
def _build_offline_remediation(finding: dict, code: str, reason: str) -> dict:
    enriched = dict(finding)

    before_snippet, before_source = _resolve_before_snippet(finding, code)
    after_snippet = _build_example_template(finding)
    fix_tier = "manual_review" if before_source == "unavailable" else "example_template"

    enriched["remediation"] = {
        "root_cause": finding.get("reason") or "Root cause could not be automatically determined without an active Remediation Agent LLM pass.",
        "guideline_reference": _fallback_guideline_reference(finding),
        "retrieved_guidance": [],
        "before_snippet": before_snippet,
        "before_snippet_source": before_source,
        "after_snippet": after_snippet,
        "after_snippet_source": "template",
        "explanation": _fallback_explanation(finding),
        "prevention_tip": _fallback_prevention_tip(finding),
        "application_confidence": "manual-review-required",
        "syntax_valid": False,
        "validation_note": reason,
        "fix_tier": fix_tier,
        "badge": _BADGE.get(fix_tier, _BADGE["manual_review"]),
    }
    enriched["remediation_unavailable_reason"] = None
    return enriched


# --------------------------------------------------------------------------- #
# Public entry point used by the orchestrator (same interface shape as
# severityagent.analyze: findings in, enriched findings out)
# --------------------------------------------------------------------------- #
def analyze(
    findings: list,
    code: str,
    language: str,
    api_key: str = "",
    use_llm: bool = True,
    rag_pipeline=None,
    batch_size: int = 10,
) -> list:
    """
    Attach a grounded, validated remediation to each finding. Every finding
    always ends up with a complete, non-empty remediation dict - even when
    the LLM is disabled, unavailable, or fails - so the Remediation tab
    never has to render a placeholder.

    ``api_key`` is kept in the signature for backward compatibility with
    existing callers but is no longer required or used directly -
    credentials are sourced server-side via ``llm_router``, which also
    handles automatic Gemini -> Groq failover.

    ``rag_pipeline`` is optional - pass the app's existing
    ``st.session_state.rag_pipeline`` (a rag_engine.RAGPipeline instance) to
    ground guideline_reference in real retrieved text. If omitted or not
    yet built, remediation still runs on the router's LLM (or, if neither
    provider is configured/available, on the local fallback guidance above).
    """
    if not findings:
        return []

    if not use_llm or not llm_router.any_configured():
        return [
            _build_offline_remediation(
                f, code, "No LLM provider configured on the server; showing locally generated guidance instead of an AI-generated fix."
            )
            for f in findings
        ]

    output = [
        _build_offline_remediation(f, code, "Remediation not yet generated.") for f in findings
    ]
    effective_batch_size = max(1, int(batch_size or 10))

    # Cache retrieval by (title, cwe) so findings that repeat the same
    # issue across multiple lines don't re-query the knowledge base.
    guidance_cache: dict = {}

    def _guidance_for(finding: dict) -> List[dict]:
        cache_key = (finding.get("title") or finding.get("category"), finding.get("cwe_id") or finding.get("cwe"))
        if cache_key not in guidance_cache:
            guidance_cache[cache_key] = _retrieve_guidance(rag_pipeline, finding)
        return guidance_cache[cache_key]

    for batch in _chunks(list(enumerate(findings)), effective_batch_size):
        retrieved_by_index = {index: _guidance_for(finding) for index, finding in batch}
        payload = [
            _finding_payload(finding, index, retrieved_by_index[index])
            for index, finding in batch
        ]
        try:
            response: RemediationBatch = llm_router.run_structured(
                RemediationBatch,
                _REMEDIATION_PROMPT,
                {
                    "language": (language or "").lower(),
                    "findings_json": json.dumps(payload, ensure_ascii=False, indent=2),
                },
                temperature=0.1,
            )
            by_index = {item.index: item for item in response.remediations}
            for index, finding in batch:
                item = by_index.get(index)
                output[index] = (
                    _apply_remediation(finding, item, language, retrieved_by_index[index], code)
                    if item
                    else _build_offline_remediation(finding, code, "Neither LLM provider returned a remediation for this finding.")
                )
        except Exception as exc:
            for index, finding in batch:
                output[index] = _build_offline_remediation(
                    finding,
                    code,
                    f"Remediation generation failed on both LLM providers; showing locally generated guidance: {exc}",
                )

    return output
