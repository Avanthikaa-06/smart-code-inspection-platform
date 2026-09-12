"""
agents/prsummaryagent.py
------------------------
Reusable agent responsible only for compiling a structured, PR-review-style
summary out of the other agents' already-produced output.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

GEMINI_MODEL = "gemini-3.5-flash-lite"

# --------------------------------------------------------------------------- #
# Shared constants
# --------------------------------------------------------------------------- #
SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}
SEVERITY_EMOJI = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "🔵",
    "Info": "⚪",
}
FIX_TIER_LABEL = {
    "ai_generated": "AI-Generated Fix",
    "example_template": "Example Pattern",
    "manual_review": "Manual Review Required",
}
MERGE_META = {
    "blocked": ("🛑", "Blocked", "red"),
    "caution": ("⚠️", "Merge with Caution", "orange"),
    "approved": ("✅", "Approved", "brightgreen"),
}
REVIEW_MINUTES_PER_SEVERITY = {"Critical": 3.0, "High": 2.0, "Medium": 1.0, "Low": 0.5, "Info": 0.2}

HEALTH_STATUS_META = {
    "excellent": ("Excellent", "green"),
    "healthy": ("Healthy", "green"),
    "needs_attention": ("Needs Attention", "yellow"),
    "at_risk": ("At Risk", "orange"),
    "critical": ("Critical", "red"),
}

SECURITY_POSTURE_BUCKETS = {
    "Injection Issues": ["injection", "sql injection", "cwe-89", "cwe-79", "xss", "cross-site scripting"],
    "Authentication Issues": ["authentication", "auth bypass", "broken auth", "session fixation", "cwe-287", "cwe-306", "cwe-384"],
    "Secrets": ["hardcoded", "hard-coded", "secret", "api key", "credential", "cwe-798"],
    "Cryptography": ["crypto", "hashing", "md5", "sha1", "weak cipher", "weak encryption", "cwe-327", "cwe-328", "cwe-326"],
    "Command Injection": ["command injection", "os.system", "os command", "shell injection", "cwe-78"],
    "Unsafe APIs": ["eval(", "exec(", "pickle", "unsafe deserialization", "deserialization", "unsafe api", "cwe-502"],
}

PIPELINE_STEPS = [
    ("code_uploaded", "Code Uploaded"),
    ("validation", "Validation"),
    ("code_analysis", "Code Analysis"),
    ("security_analysis", "Security Analysis"),
    ("severity_analysis", "Severity Analysis"),
    ("remediation", "Remediation"),
    ("pr_summary", "PR Summary"),
]


def _severity_rank(severity: Optional[str]) -> int:
    return SEVERITY_ORDER.get(severity, 2)


def _tools_for(finding: dict) -> List[str]:
    tools = finding.get("detected_by")
    if tools:
        return list(tools)
    tool = finding.get("tool")
    return [tool] if tool else ["unknown"]


def _finding_text_blob(finding: dict) -> str:
    return " ".join([
        str(finding.get("title") or ""),
        str(finding.get("description") or ""),
        str(finding.get("category") or ""),
        str(finding.get("cwe") or finding.get("cwe_id") or ""),
        str(finding.get("owasp") or finding.get("owasp_category") or ""),
    ]).lower()


def decide_merge_recommendation(by_severity: dict) -> dict:
    critical = by_severity.get("Critical", 0)
    high = by_severity.get("High", 0)
    medium = by_severity.get("Medium", 0)

    if critical > 0:
        key = "blocked"
        description = f"Blocked — {critical} unresolved Critical finding(s)."
    elif high >= 3:
        key = "blocked"
        description = f"Blocked — {high} unresolved High-severity findings."
    elif high > 0 or medium >= 6:
        key = "caution"
        count = high or medium
        label = "High" if high else "Medium"
        description = f"Merge with Caution — {count} unresolved {label} finding(s)."
    else:
        key = "approved"
        description = "Approved — no blocking issues."

    icon, label, color = MERGE_META[key]
    return {"key": key, "icon": icon, "label": label, "color": color, "description": description}


def compute_agreement_stats(findings: list) -> dict:
    total = len(findings)
    if not total:
        return {"total": 0, "multi_tool": 0, "agreement_pct": 0}
    multi_tool = sum(1 for f in findings if len(_tools_for(f)) > 1)
    return {"total": total, "multi_tool": multi_tool, "agreement_pct": round(100 * multi_tool / total)}


def compute_category_breakdown(findings: list) -> dict:
    breakdown: dict = {}
    for f in findings:
        category = f.get("taxonomy") or ("Security" if f.get("agent") == "security" else "Code Quality")
        breakdown[category] = breakdown.get(category, 0) + 1
    return dict(sorted(breakdown.items(), key=lambda kv: -kv[1]))


def compute_auto_fixable_ratio(findings: list) -> dict:
    tiers = {"auto-applicable": 0, "needs-review": 0, "manual-review-required": 0}
    with_remediation = 0
    for f in findings:
        remediation = f.get("remediation") or {}
        confidence = remediation.get("application_confidence")
        if confidence in tiers:
            tiers[confidence] += 1
            with_remediation += 1

    total = len(findings)
    auto = tiers["auto-applicable"]
    return {
        "auto_applicable": auto,
        "needs_review": tiers["needs-review"],
        "manual_review": tiers["manual-review-required"],
        "total_findings": total,
        "total_with_remediation": with_remediation,
        "ratio_pct": round(100 * auto / total) if total else 0,
    }


def compute_ai_fix_readiness(auto_fixable: dict) -> dict:
    total = auto_fixable["total_findings"] or 0
    def pct(n):
        return round(100 * n / total) if total else 0
    return {
        **auto_fixable,
        "auto_pct": pct(auto_fixable["auto_applicable"]),
        "needs_review_pct": pct(auto_fixable["needs_review"]),
        "manual_pct": pct(auto_fixable["manual_review"]),
    }


def compute_owasp_coverage(findings: list) -> List[str]:
    categories = set()
    for f in findings:
        raw = f.get("owasp") or f.get("owasp_category")
        if not raw:
            continue
        for part in str(raw).split(","):
            cleaned = part.strip()
            if cleaned:
                categories.add(cleaned)
    return sorted(categories)


def compute_cwe_coverage(findings: list) -> List[str]:
    cwes = set()
    for f in findings:
        raw = f.get("cwe") or f.get("cwe_id")
        if not raw:
            continue
        for part in str(raw).split(","):
            cleaned = part.strip()
            if cleaned:
                cwes.add(cleaned)
    return sorted(cwes)


def compute_compliance(findings: list, owasp_coverage: List[str], cwe_coverage: List[str]) -> dict:
    cert_rules = sorted({str(f.get("cert_rule")) for f in findings if f.get("cert_rule")})
    best_practices = set()
    for f in findings:
        raw = f.get("best_practices") or f.get("best_practice")
        if not raw:
            continue
        items = raw if isinstance(raw, (list, tuple, set)) else [raw]
        for item in items:
            cleaned = str(item).strip()
            if cleaned:
                best_practices.add(cleaned)
    return {
        "owasp_categories": owasp_coverage,
        "cwe_ids": cwe_coverage,
        "cert_rules": cert_rules,
        "best_practices": sorted(best_practices),
    }


def compute_security_posture(findings: list, by_severity: dict) -> dict:
    buckets = {name: 0 for name in SECURITY_POSTURE_BUCKETS}
    for f in findings:
        blob = _finding_text_blob(f)
        for name, keywords in SECURITY_POSTURE_BUCKETS.items():
            if any(kw in blob for kw in keywords):
                buckets[name] += 1
    return {
        "critical_vulnerabilities": by_severity.get("Critical", 0),
        "buckets": buckets,
        "trend_available": False,
    }


def compute_code_quality_overview(findings: list) -> dict:
    quality_findings = [
        f for f in findings
        if f.get("agent") != "security" and (f.get("taxonomy") or "Code Quality") != "Security"
    ]
    breakdown: dict = {}
    maintainability_flags = 0
    for f in quality_findings:
        taxonomy = f.get("taxonomy") or "Code Quality"
        breakdown[taxonomy] = breakdown.get(taxonomy, 0) + 1
        if "maintain" in _finding_text_blob(f):
            maintainability_flags += 1
    return {
        "total": len(quality_findings),
        "breakdown": dict(sorted(breakdown.items(), key=lambda kv: -kv[1])),
        "maintainability_flags": maintainability_flags,
    }


def compute_risk_distribution(by_severity: dict, total: int) -> List[dict]:
    rows = []
    for sev in ("Critical", "High", "Medium", "Low", "Info"):
        count = by_severity.get(sev, 0)
        pct = round(100 * count / total) if total else 0
        rows.append({"severity": sev, "count": count, "pct": pct, "emoji": SEVERITY_EMOJI[sev]})
    return rows


def compute_repository_health(mergeability_score: int, by_severity: dict, review_minutes: int) -> dict:
    critical = by_severity.get("Critical", 0)
    high = by_severity.get("High", 0)
    score = mergeability_score if mergeability_score is not None else 100

    if score >= 90 and critical == 0 and high == 0:
        status_key = "excellent"
    elif score >= 75 and critical == 0:
        status_key = "healthy"
    elif score >= 50 and critical == 0:
        status_key = "needs_attention"
    elif score >= 25:
        status_key = "at_risk"
    else:
        status_key = "critical"

    label, color = HEALTH_STATUS_META[status_key]
    stars = max(1, min(5, round(score / 20)))
    return {
        "score": score,
        "stars": stars,
        "status_key": status_key,
        "status_label": label,
        "color": color,
        "review_time_minutes": review_minutes,
    }


def compute_ai_confidence(agreement: dict, auto_fixable: dict, deep_result: dict) -> dict:
    agreement_pct = agreement.get("agreement_pct", 0)
    remediation_pct = auto_fixable.get("ratio_pct", 0)
    syntax_valid = deep_result.get("syntax_valid")

    components = {
        "cross_tool_agreement_pct": agreement_pct,
        "remediation_confidence_pct": remediation_pct,
    }
    values = [agreement_pct, remediation_pct]
    if syntax_valid is not None:
        syntax_pct = 100 if syntax_valid else 0
        components["syntax_validation_pct"] = syntax_pct
        values.append(syntax_pct)
    else:
        components["syntax_validation_pct"] = None

    overall = round(sum(values) / len(values)) if values else 0
    return {"overall_pct": overall, "components": components, "syntax_validated": syntax_valid}


def compute_scan_statistics(deep_result: dict, findings: list, code: str) -> dict:
    tool_statuses = deep_result.get("tool_statuses", []) or []
    agent_status = deep_result.get("agent_status", {}) or {}
    total_tools = len(tool_statuses)
    successful_tools = sum(1 for t in tool_statuses if t.get("status") == "success")
    return {
        "lines_of_code": len((code or "").splitlines()),
        "files_scanned": deep_result.get("files_scanned", 1),
        "agents_executed": len(agent_status),
        "tools_executed": total_tools,
        "execution_time_seconds": deep_result.get("execution_time"),
        "total_findings": len(findings),
        "success_rate_pct": round(100 * successful_tools / total_tools) if total_tools else 100,
    }


def build_scan_timeline(deep_result: dict) -> List[dict]:
    agent_status = deep_result.get("agent_status", {}) or {}
    step_timestamps = deep_result.get("step_timestamps", {}) or {}
    timeline = []
    for key, label in PIPELINE_STEPS:
        status = agent_status.get(key)
        completed = status not in ("failed", "skipped", None) if agent_status else True
        timeline.append({
            "step": label,
            "completed": completed,
            "timestamp": step_timestamps.get(key) or step_timestamps.get(label),
        })
    return timeline


def build_prioritized_fixes(findings: list, top_n: Optional[int] = None) -> List[dict]:
    """
    Ranks findings by severity/priority and returns a flat, PDF/UI-ready
    dict per finding. Each item carries everything a developer needs to
    act on the finding without going back to the raw `findings` list:
    file, line, title, severity, a clear description, root cause (when
    the Remediation Agent supplied one), a concrete recommendation, and
    the relevant code snippet (when available).
    """
    ranked = sorted(
        findings,
        key=lambda f: (
            _severity_rank(f.get("severity")),
            -(f.get("score") or 0),
            f.get("priority") or 3,
        ),
    )
    if top_n:
        ranked = ranked[:top_n]

    items = []
    for rank, finding in enumerate(ranked, start=1):
        remediation = finding.get("remediation") or {}
        severity = finding.get("severity", "Medium")
        title = finding.get("title") or finding.get("category") or "Finding"
        file_ = finding.get("file")
        line = finding.get("line") or finding.get("line_start") or "N/A"

        description = finding.get("description") or finding.get("message") or "No description provided."
        # Root cause is only populated when the Remediation Agent actually
        # produced one — we don't fall back to the description here, since
        # the report needs "description" and "root cause" to be genuinely
        # distinct fields rather than the same text shown twice.
        root_cause = remediation.get("root_cause") or finding.get("root_cause") or ""
        recommendation = remediation.get("explanation") or finding.get("recommendation") or "Review and remediate before release."
        code_snippet = finding.get("code_snippet") or remediation.get("before_snippet") or ""

        items.append({
            "rank": rank,
            "id": finding.get("id"),
            "severity": severity,
            "title": title,
            "file": file_,
            "line": line,
            "priority": finding.get("priority"),
            "tools": _tools_for(finding),
            "cwe": finding.get("cwe") or finding.get("cwe_id"),
            "owasp": finding.get("owasp") or finding.get("owasp_category"),
            "fix_confidence": remediation.get("application_confidence"),
            "fix_tier": remediation.get("fix_tier"),
            "description": description,
            "root_cause": root_cause,
            "recommendation": recommendation,
            "code_snippet": code_snippet,
            # Kept for backward compatibility with existing UI consumers
            # (app.py's Prioritized Fix List cards read `fix_summary`).
            "fix_summary": recommendation,
            "checklist_line": f"- [ ] Fix {title} in {file_}:{line} ({severity})",
        })
    return items


def build_agreement_matrix(findings: list) -> List[dict]:
    matches = [f for f in findings if len(_tools_for(f)) > 1]
    matches.sort(key=lambda f: _severity_rank(f.get("severity")))
    return [
        {
            "title": f.get("title") or f.get("category") or "Finding",
            "severity": f.get("severity", "Medium"),
            "file": f.get("file"),
            "line": f.get("line") or f.get("line_start") or "N/A",
            "tools": _tools_for(f),
        }
        for f in matches
    ]


_NAMED_POSITIVE_CHECKS = [
    ("No Hardcoded Secrets", ["hardcoded", "hard-coded secret", "cwe-798"]),
    ("No SQL Injection in Authentication Module", ["sql injection", "cwe-89"]),
    ("No Dangerous Deserialization", ["unsafe deserialization", "pickle.loads", "cwe-502"]),
    ("No Broken Authentication", ["broken auth", "cwe-287", "cwe-306"]),
]


def build_positive_highlights(findings: list, summary: dict) -> List[str]:
    highlights = []
    by_severity = summary.get("by_severity", {})
    if summary.get("total", 0) == 0:
        return ["No issues were flagged by any tool in the analysis stack."]
    if by_severity.get("Critical", 0) == 0:
        highlights.append("No critical-severity issues detected.")
    if by_severity.get("High", 0) == 0:
        highlights.append("No high-severity vulnerabilities or design flaws found.")

    text = " ".join(_finding_text_blob(f) for f in findings)
    for label, keywords in _NAMED_POSITIVE_CHECKS:
        if not any(kw in text for kw in keywords):
            highlights.append(label)

    uses_hashing = any(kw in text for kw in ("bcrypt", "argon2", "pbkdf2", "scrypt"))
    weak_hashing_found = any(kw in text for kw in ("md5", "sha1", "cwe-327", "cwe-328"))
    if uses_hashing and not weak_hashing_found:
        highlights.append("Secure Password Hashing in use.")

    if not highlights:
        highlights.append("Review findings below closely — no issue-free category stood out.")
    return highlights[:6]


def estimate_review_time(findings: list, code: str = "") -> int:
    lines = len((code or "").splitlines())
    base = 2.0
    per_finding = sum(REVIEW_MINUTES_PER_SEVERITY.get(f.get("severity", "Medium"), 1.0) for f in findings)
    return max(2, round(base + per_finding + lines / 100.0))


def suggest_pr_title(deep_result: dict, summary: dict) -> str:
    file_name = deep_result.get("file") or "code"
    total = summary.get("total", 0)
    by_severity = summary.get("by_severity", {})
    if total == 0:
        return f"chore: {file_name} passes automated review with no findings"
    prefix = "fix" if (by_severity.get("Critical", 0) or by_severity.get("High", 0)) else "chore"
    return f"{prefix}: address {total} finding(s) surfaced in automated review of {file_name}"


def build_pipeline_transparency(deep_result: dict) -> dict:
    agent_status = deep_result.get("agent_status", {}) or {}
    tool_statuses = deep_result.get("tool_statuses", []) or []
    return {
        "agent_status": agent_status,
        "failed_tools": [t for t in tool_statuses if t.get("status") == "failed"],
        "skipped_tools": [t for t in tool_statuses if t.get("status") == "skipped"],
        "overall_success": deep_result.get("success", True),
        "pipeline_error": deep_result.get("error"),
    }


def build_executive_recommendations(findings: list) -> dict:
    buckets = {"immediate_actions": [], "fix_this_week": [], "can_be_deferred": [], "nice_to_have": []}
    sev_to_bucket = {
        "Critical": "immediate_actions",
        "High": "fix_this_week",
        "Medium": "can_be_deferred",
        "Low": "nice_to_have",
        "Info": "nice_to_have",
    }
    ranked = sorted(findings, key=lambda f: _severity_rank(f.get("severity")))
    for f in ranked:
        bucket = sev_to_bucket.get(f.get("severity", "Medium"), "can_be_deferred")
        title = f.get("title") or f.get("category") or "Finding"
        file_ = f.get("file")
        line = f.get("line") or f.get("line_start") or "N/A"
        buckets[bucket].append(f"{title} ({file_}:{line})")
    return buckets


def build_developer_checklist(prioritized_fixes: List[dict]) -> List[str]:
    return [item["checklist_line"] for item in prioritized_fixes]


def export_json_summary(data: dict) -> str:
    exportable = {k: v for k, v in data.items() if k != "markdown"}
    return json.dumps(exportable, ensure_ascii=False, indent=2, default=str)


class ExecutiveNarrativeOutput(BaseModel):
    overview: str = Field(..., description="A 3-5 sentence executive summary for a pull request reviewer.")


def _narrative_grounding(deep_result: dict, merge: dict, summary: dict, auto_fixable: dict, owasp_coverage: List[str], top_fixes: List[dict]) -> dict:
    return {
        "file": deep_result.get("file"),
        "language": deep_result.get("language"),
        "merge_status": merge["label"],
        "total_findings": summary.get("total", 0),
        "by_severity": summary.get("by_severity", {}),
        "mergeability_score": summary.get("security_score"),
        "auto_fixable_pct": auto_fixable["ratio_pct"],
        "owasp_categories_touched": owasp_coverage,
        "top_findings": [
            {"title": f["title"], "severity": f["severity"], "location": f"{f['file']}:{f['line']}"}
            for f in top_fixes[:3]
        ],
    }


def _fallback_narrative(grounding: dict) -> str:
    total = grounding["total_findings"]
    if total == 0:
        return f"This change to `{grounding['file']}` introduced no findings across code quality or security analysis."
    by_sev = grounding["by_severity"]
    sev_bits = ", ".join(f"{count} {sev}" for sev, count in by_sev.items() if count)
    return f"This change to `{grounding['file']}` produced {total} finding(s) ({sev_bits}), with a mergeability score of {grounding['mergeability_score']}/100. Current status: {grounding['merge_status']}."


def _generate_executive_narrative(grounding: dict, api_key: str, use_llm: bool) -> str:
    if not use_llm or not api_key or not api_key.strip():
        return _fallback_narrative(grounding)

    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key.strip(), temperature=0.2)
        structured_llm = llm.with_structured_output(ExecutiveNarrativeOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You write the executive overview paragraph for an automated pull request review summary. You are given exact figures. Do not recount or re-total. Write 3-5 sentences in a senior-reviewer tone."),
            ("human", "Grounding data:\n{grounding_json}"),
        ])
        response: ExecutiveNarrativeOutput = (prompt | structured_llm).invoke({"grounding_json": json.dumps(grounding, ensure_ascii=False, indent=2)})
        return (response.overview or "").strip() or _fallback_narrative(grounding)
    except Exception:
        return _fallback_narrative(grounding)


def render_markdown(deep_result: dict, data: dict) -> str:
    merge = data["merge_recommendation"]
    auto_fixable = data["ai_fix_readiness"]
    health = data["repository_health"]
    score = data.get("mergeability_score", 100)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = [
        "# 🤖 Automated PR Review Summary",
        "",
        f"*Generated {generated_at} · `{deep_result.get('file', 'N/A')}`*",
        "",
        f"## {merge['icon']} {merge['description']}",
        "",
        f"**Repository Health:** {health['score']} / 100 ({health['status_label']})",
        "",
        "## 📋 Executive Overview",
        data["executive_overview"],
        "",
        "## 🚦 Risk Distribution",
    ]
    for row in data["risk_distribution"]:
        lines.append(f"- {row['emoji']} **{row['severity']}**: {row['count']} ({row['pct']}%)")
    
    return "\n".join(lines)


def generate_pr_summary(deep_result: dict, code: str = "", file_name: str = "", api_key: str = "", use_llm: bool = True) -> dict:
    if not deep_result:
        return {"markdown": "_No analysis has been run yet — nothing to summarize._"}

    if file_name:
        deep_result = {**deep_result, "file": file_name}

    findings = deep_result.get("findings", []) or []
    summary = deep_result.get("summary") or {"total": 0, "by_severity": {}, "security_score": 100}
    by_severity = summary.get("by_severity", {})
    total_findings = summary.get("total", 0)
    mergeability_score = summary.get("security_score", 100)

    merge = decide_merge_recommendation(by_severity)
    agreement = compute_agreement_stats(findings)
    category_breakdown = compute_category_breakdown(findings)
    auto_fixable = compute_auto_fixable_ratio(findings)
    ai_fix_readiness = compute_ai_fix_readiness(auto_fixable)
    owasp_coverage = compute_owasp_coverage(findings)
    cwe_coverage = compute_cwe_coverage(findings)
    compliance = compute_compliance(findings, owasp_coverage, cwe_coverage)
    prioritized_fixes = build_prioritized_fixes(findings)
    top_critical_findings = build_prioritized_fixes(findings, top_n=5)
    review_time_minutes = estimate_review_time(findings, code)

    grounding = _narrative_grounding(deep_result, merge, summary, auto_fixable, owasp_coverage, prioritized_fixes)
    executive_overview = _generate_executive_narrative(grounding, api_key, use_llm)

    data = {
        "merge_recommendation": merge,
        "executive_overview": executive_overview,
        "severity_breakdown": by_severity,
        "category_breakdown": category_breakdown,
        "auto_fixable": auto_fixable,
        "ai_fix_readiness": ai_fix_readiness,
        "owasp_coverage": owasp_coverage,
        "compliance": compliance,
        "agreement": agreement,
        "total_findings": total_findings,
        "mergeability_score": mergeability_score,
        "prioritized_fixes": prioritized_fixes,
        "top_critical_findings": top_critical_findings,
        "agreement_matrix": build_agreement_matrix(findings),
        "positive_highlights": build_positive_highlights(findings, summary),
        "review_time_minutes": review_time_minutes,
        "suggested_pr_title": suggest_pr_title(deep_result, summary),
        "pipeline_transparency": build_pipeline_transparency(deep_result),
        "repository_health": compute_repository_health(mergeability_score, by_severity, review_time_minutes),
        "risk_distribution": compute_risk_distribution(by_severity, total_findings),
        "security_posture": compute_security_posture(findings, by_severity),
        "code_quality_overview": compute_code_quality_overview(findings),
        "ai_confidence": compute_ai_confidence(agreement, auto_fixable, deep_result),
        "scan_statistics": compute_scan_statistics(deep_result, findings, code),
        "scan_timeline": build_scan_timeline(deep_result),
        "executive_recommendations": build_executive_recommendations(findings),
        "developer_checklist": build_developer_checklist(prioritized_fixes),
    }
    data["markdown"] = render_markdown(deep_result, data)
    data["export_json"] = export_json_summary(data)
    return data