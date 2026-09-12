import asyncio
import operator
import os
import re
import sys
import time
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from langgraph.graph import END, START, StateGraph
except Exception:
    END = START = StateGraph = None

try:
    from agents import codeanalysis, securityagent, severityagent, remediationagent
except ImportError:
    import codeanalysis, securityagent, severityagent, remediationagent


SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}


class AnalysisState(TypedDict, total=False):
    code: str
    filepath: str
    language: str
    api_key: str
    use_llm: bool
    code_findings: Annotated[list, operator.add]
    security_findings: Annotated[list, operator.add]
    tool_statuses: Annotated[list, operator.add]
    agent_status: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    findings: list
    summary: dict
    success: bool


def _as_dict_findings(items: list) -> list:
    return [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in (items or [])]


def _normalize_finding(finding: dict) -> dict:
    line = int(finding.get("line_start") or finding.get("line") or 1)
    severity = str(finding.get("severity", "Medium")).strip().capitalize()
    if severity not in SEVERITY_ORDER:
        severity = "Medium"
    agent = "security" if "security" in str(finding.get("agent", "")).lower() else "code_analysis"
    title = finding.get("title") or finding.get("category") or "Finding"
    description = finding.get("description") or finding.get("message") or ""
    cwe = finding.get("cwe_id") or finding.get("cwe")
    owasp = finding.get("owasp_category") or finding.get("owasp")
    return {
        **finding,
        "agent": agent,
        "agent_source": "security_vulnerability" if agent == "security" else "code_analysis",
        "line": line,
        "line_start": line,
        "line_end": int(finding.get("line_end") or line),
        "severity": severity,
        "title": title,
        "category": title,
        "description": description,
        "message": description,
        "cwe": cwe,
        "cwe_id": cwe,
        "owasp": owasp,
        "owasp_category": owasp,
        "recommendation": finding.get("recommendation") or "Review and remediate this finding before release.",
        "tool": finding.get("tool") or "unknown",
    }


def _dedupe(findings: list) -> list:
    """
    Groups findings by (file, line, cwe, title) — WITHOUT tool in the key —
    so the same vulnerability flagged by both Bandit and Semgrep collapses
    into ONE entry instead of two, before severity scoring ever runs.
    The most severe finding in each group becomes the representative entry;
    every tool that independently flagged it is recorded in detected_by.
    """
    order: list = []
    grouped: dict = {}

    for raw in findings:
        finding = _normalize_finding(raw)
        cwe_raw = str(finding.get("cwe") or "")
        cwe_match = re.search(r"(CWE-\d+)", cwe_raw, re.IGNORECASE)
        cwe_id = cwe_match.group(1).upper() if cwe_match else ""
        key = (
            finding.get("file"),
            finding.get("line"),
            cwe_id,
            re.sub(r"\s+", " ", finding.get("title", "").lower()).strip(),
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(finding)

    deduped = []
    for key in order:
        group = grouped[key]
        # Worst-case severity wins as the representative entry.
        representative = dict(min(group, key=lambda f: SEVERITY_ORDER.get(f["severity"], 2)))
        tools = sorted({f.get("tool") or "unknown" for f in group})
        representative["detected_by"] = tools
        representative["tool"] = " + ".join(tools) if len(tools) > 1 else tools[0]
        deduped.append(representative)

    return deduped


def _summarize(findings: list) -> dict:
    by_severity = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Info": 0}
    by_agent = {"code_analysis": 0, "security": 0}
    risk_scores = []

    for finding in findings:
        by_severity[finding["severity"]] += 1
        by_agent[finding["agent"]] += 1
        score = finding.get("score")
        if isinstance(score, (int, float)):
            risk_scores.append(score)

    if risk_scores:
        # severityagent.analyze() always attaches a per-finding risk score
        # (LLM-derived, or fallback-enriched when no API key). The overall
        # security score is the inverse of the average risk across every
        # finding, so it reflects what the Severity Agent actually decided
        # instead of a fixed bucket-count formula that never sees it.
        avg_risk = sum(risk_scores) / len(risk_scores)
        security_score = max(0, min(100, round(100 - avg_risk)))
    else:
        penalty = sum(
            by_severity[severity] * weight
            for severity, weight in {
                "Critical": 10, "High": 5, "Medium": 2, "Low": 1, "Info": 0,
            }.items()
        )
        security_score = max(0, min(100, 100 - penalty))

    return {
        "total": len(findings),
        "by_severity": by_severity,
        "by_agent": by_agent,
        "security_score": security_score,
    }


def _code_analysis_node(state: AnalysisState) -> AnalysisState:
    try:
        result = codeanalysis.analyze(
            state["code"],
            state["filepath"],
            state["language"],
            state.get("api_key", ""),
            state.get("use_llm", True),
        )
        return {
            "code_findings": _as_dict_findings(result.findings),
            "tool_statuses": result.tool_statuses,
            "agent_status": [{"code_analysis": result.status}],
            "errors": [result.error] if result.error else [],
        }
    except Exception as exc:
        return {
            "code_findings": [],
            "agent_status": [{"code_analysis": "failed"}],
            "errors": [f"Code analysis crashed: {exc}"],
        }


def _security_node(state: AnalysisState) -> AnalysisState:
    try:
        result = securityagent.analyze(
            state["code"],
            state["filepath"],
            state["language"],
            state.get("api_key", ""),
            state.get("use_llm", True),
        )
        return {
            "security_findings": _as_dict_findings(result.findings),
            "tool_statuses": result.tool_statuses,
            "agent_status": [{"security_vulnerability": result.status}],
            "errors": [result.error] if result.error else [],
        }
    except Exception as exc:
        return {
            "security_findings": [],
            "agent_status": [{"security_vulnerability": "failed"}],
            "errors": [f"Security scan crashed: {exc}"],
        }


def _merge_node(state: AnalysisState) -> AnalysisState:
    findings = _dedupe((state.get("code_findings") or []) + (state.get("security_findings") or []))

    agent_status_updates = []
    tool_status_updates = []
    node_errors = []

    # ---- Severity Normalization Agent ----
    severity_start = time.perf_counter()
    if not findings:
        severity_status = "skipped"
    else:
        try:
            findings = severityagent.analyze(
                findings,
                state.get("code", ""),
                state.get("language", ""),
                state.get("api_key", ""),
                state.get("use_llm", True),
            )
            severity_status = "success"
        except Exception as exc:
            severity_status = "failed"
            node_errors.append(f"Severity Agent crashed: {exc}")
    severity_duration = time.perf_counter() - severity_start

    agent_status_updates.append({"severity_normalization": severity_status})
    tool_status_updates.append({
        "tool": "Severity Agent",
        "agent": "severity_normalization",
        "status": severity_status,
        "findings_count": len(findings),
        "duration_seconds": severity_duration,
    })

    # ---- Remediation Agent ----
    remediation_start = time.perf_counter()
    has_llm_access = bool(state.get("api_key")) and bool(state.get("use_llm", True))
    if not findings:
        remediation_status = "skipped"
    elif not has_llm_access:
        # remediationagent requires a valid Gemini key to generate fixes at all.
        remediation_status = "skipped"
    else:
        try:
            findings = remediationagent.analyze(
                findings,
                state.get("code", ""),
                state.get("language", ""),
                state.get("api_key", ""),
                state.get("use_llm", True),
                rag_pipeline=state.get("rag_pipeline"),
            )
            remediation_status = "success"
        except Exception as exc:
            remediation_status = "failed"
            node_errors.append(f"Remediation Agent crashed: {exc}")
    remediation_duration = time.perf_counter() - remediation_start
    remediated_count = sum(1 for f in findings if f.get("remediation"))

    agent_status_updates.append({"remediation": remediation_status})
    tool_status_updates.append({
        "tool": "Remediation Agent",
        "agent": "remediation",
        "status": remediation_status,
        "findings_count": remediated_count,
        "duration_seconds": remediation_duration,
    })

    findings.sort(key=lambda item: (SEVERITY_ORDER.get(item["severity"], 2), item["line"], item["tool"]))
    errors = [err for err in state.get("errors", []) if err] + node_errors
    return {
        "findings": findings,
        "summary": _summarize(findings),
        "success": not errors,
        "agent_status": agent_status_updates,
        "tool_statuses": tool_status_updates,
        "errors": node_errors,
    }


def _merge_agent_status(state: AnalysisState) -> dict:
    status = {
        "code_analysis": "pending",
        "security_vulnerability": "pending",
        "severity_normalization": "pending",
        "remediation": "pending",
    }
    fragments = state.get("agent_status")
    if isinstance(fragments, list):
        for fragment in fragments:
            status.update(fragment)
    elif isinstance(fragments, dict):
        status.update(fragments)
    return status


def _build_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("code_analysis_agent", _code_analysis_node)
    graph.add_node("security_agent", _security_node)
    graph.add_node("merge_findings", _merge_node)
    graph.add_edge(START, "code_analysis_agent")
    graph.add_edge(START, "security_agent")
    graph.add_edge(["code_analysis_agent", "security_agent"], "merge_findings")
    graph.add_edge("merge_findings", END)
    return graph.compile()


async def _run_langgraph_pipeline(state: AnalysisState) -> AnalysisState:
    if StateGraph is None:
        code_result, sec_result = await asyncio.gather(
            asyncio.to_thread(_code_analysis_node, state),
            asyncio.to_thread(_security_node, state),
        )
        merged_state = {
            **state,
            "code_findings": code_result.get("code_findings", []),
            "security_findings": sec_result.get("security_findings", []),
            "tool_statuses": code_result.get("tool_statuses", []) + sec_result.get("tool_statuses", []),
            "agent_status": code_result.get("agent_status", []) + sec_result.get("agent_status", []),
            "errors": code_result.get("errors", []) + sec_result.get("errors", []),
        }
        merge_result = _merge_node(merged_state)
        merge_result["agent_status"] = merged_state["agent_status"] + merge_result.get("agent_status", [])
        merge_result["tool_statuses"] = merged_state["tool_statuses"] + merge_result.get("tool_statuses", [])
        merge_result["errors"] = merged_state["errors"] + merge_result.get("errors", [])
        return {**merged_state, **merge_result}
    graph = _build_graph()
    return await asyncio.to_thread(graph.invoke, state)


async def run_pipeline_async(code: str, filepath: str, language: str, api_key: str = "", use_llm: bool = True, rag_pipeline=None) -> dict:
    lang = (language or "").strip().lower()
    if lang not in ("python", "java"):
        return {
            "success": False,
            "error": f"Unsupported or undetected language: {language}",
            "file": filepath,
            "language": language,
            "findings": [],
            "tool_statuses": [],
            "agent_status": {
                "code_analysis": "skipped",
                "security_vulnerability": "skipped",
                "severity_normalization": "skipped",
                "remediation": "skipped",
            },
            "summary": _summarize([]),
            "mode": "No analysis run.",
            "llm_remediation": "No analysis run.",
        }

    initial_state: AnalysisState = {
        "code": code,
        "filepath": filepath,
        "language": lang,
        "api_key": api_key or "",
        "use_llm": use_llm,
        "rag_pipeline": rag_pipeline,
        "code_findings": [],
        "security_findings": [],
        "tool_statuses": [],
        "agent_status": [],
        "errors": [],
    }
    final_state = await _run_langgraph_pipeline(initial_state)
    errors = [err for err in final_state.get("errors", []) if err]
    mode_label = (
        "LangGraph parallel local analysis with optional LangChain/Gemini review."
        if api_key and api_key.strip() and use_llm
        else "LangGraph parallel local analysis."
    )
    return {
        "success": not errors,
        "error": "; ".join(errors) if errors else None,
        "file": filepath,
        "language": lang,
        "findings": final_state.get("findings", []),
        "tool_statuses": final_state.get("tool_statuses", []),
        "agent_status": _merge_agent_status(final_state),
        "summary": final_state.get("summary", _summarize([])),
        "mode": mode_label,
        "llm_remediation": mode_label,
    }


def run_analysis(code: str, language: str, api_key: str = None, use_llm: bool = True, filepath: str = None, rag_pipeline=None) -> dict:
    if not code or not code.strip():
        return {"success": False, "error": "No code provided.", "findings": [], "summary": _summarize([])}
    target = filepath or f"submitted{'.py' if (language or '').lower() == 'python' else '.java'}"
    return asyncio.run(run_pipeline_async(code, target, language, api_key or "", use_llm, rag_pipeline))


async def run_pipeline_async_public(code: str, filepath: str, language: str, api_key: str = "", use_llm: bool = True, rag_pipeline=None) -> dict:
    return await run_pipeline_async(code, filepath, language, api_key, use_llm, rag_pipeline)