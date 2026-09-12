"""
Presentation-only module. Renders a GitHub Security / SonarQube / Snyk-style
"Developer Findings Portal" for a single analyzed submission.

This module NEVER runs static analysis tools or calls an LLM. It only reads
the already-computed `deep_result` dict produced by
agents.orchestrator.run_analysis() and, where possible, reuses
agents.prsummaryagent's already-computed aggregates (merge recommendation,
category breakdown, auto-fixable ratio, OWASP coverage, markdown export)
instead of recomputing them. Any grouping/counting done locally in this file
(severity/tool distributions, radar-dimension bucketing) is pure display
aggregation over existing finding dicts — it does not alter, regenerate, or
reinterpret any finding.

Public entry point:
    render_findings_display(deep_result, result, state_key, show_radar=True)

`show_radar` lets callers keep the Security Posture Radar scoped to a single
tab (Developer Portal) instead of every tab that happens to call into this
module — the radar is the only graph in the Dashboard section, so this
parameter controls whether the Dashboard section's one chart renders at all.

SCORING NOTE: all severity-weighted scores in this file (dimension scores,
security-only score, domain-only score) use sqrt-dampened penalties —
`weight[severity] * sqrt(count_of_that_severity)` rather than a flat
`weight * count`. This keeps the score sensitive to finding volume (more
findings of a severity still costs more) without letting volume alone
collapse everything to the same floor, and without letting a large pile of
Low findings outweigh a single Critical.

Developer Portal scope (UI-ONLY — see review notes below):
  - TOP: a single, prominent "Download Report" action. This reuses the
    exact same generate_pr_summary() + generate_pr_pdf() call already used
    by the PR Summary Agent output (agents/prsummaryagent.py +
    pr_summary_pdf.py) with the same deep_result/code/file_name inputs, so
    it is byte-for-byte the same report — never a second/different PDF.
  - Code Health header, metrics, and the Dashboard section (Security
    Posture Radar) are unchanged and kept exactly as before.
  - The old numbered "Findings (N)" list — including per-finding Root
    Cause, Recommendation, Code Snippet, Remediation, Before/After, "Why
    This Fix Works", Secure Coding Guideline, and Prevention Tip — has
    been REMOVED from this portal entirely. That level of detail lives
    only in the PDF report now (it was already there), so nothing is lost
    — it's just no longer duplicated on-screen. This also removes the
    stray raw content that used to render at the very bottom of the page,
    since that was the last thing the (now-removed) per-finding loop drew.

No analysis logic, agents, severity normalization, remediation generation,
chatbot, or backend functionality is touched by this module — it only
decides what already-computed data gets displayed and how.
"""

import html
import math
import os
import tempfile

import streamlit as st
import plotly.graph_objects as go

import llm_router
from agents import prsummaryagent
from pr_summary_pdf import generate_pr_pdf

# --------------------------------------------------------------------------- #
# Local constants (mirrors app.py's severity styling so the portal matches
# the rest of the app's visual language)
# --------------------------------------------------------------------------- #
_SEVERITY_COLORS = {
    "Critical": "#F43F5E",
    "High": "#F97316",
    "Medium": "#F59E0B",
    "Low": "#38BDF8",
    "Info": "#94A3B8",
}
_SEVERITY_CHIP_STYLE = {
    "Critical": "background:rgba(244,63,94,0.16); color:#FB7185; border:1px solid rgba(244,63,94,0.4);",
    "High": "background:rgba(249,115,22,0.14); color:#FB923C; border:1px solid rgba(249,115,22,0.35);",
    "Medium": "background:rgba(245,158,11,0.14); color:#FBBF24; border:1px solid rgba(245,158,11,0.35);",
    "Low": "background:rgba(56,189,248,0.14); color:#38BDF8; border:1px solid rgba(56,189,248,0.35);",
    "Info": "background:rgba(148,163,184,0.12); color:#94A3B8; border:1px solid rgba(148,163,184,0.25);",
}

_GRADE_COLORS = {
    "A+": "#4ADE80", "A": "#4ADE80",
    "B+": "#38BDF8", "B": "#38BDF8",
    "C+": "#FBBF24", "C": "#FBBF24",
    "D": "#FB923C", "F": "#FB7185",
}


# --------------------------------------------------------------------------- #
# Small local helpers (kept private to this module so it has no dependency
# on app.py — app.py depends on this module, not the other way around)
# --------------------------------------------------------------------------- #
def _safe(value) -> str:
    return html.escape(str(value or ""))


def _section_head(label: str) -> None:
    st.markdown(f"<div class='section-head'><div class='bar'></div>{_safe(label)}</div>", unsafe_allow_html=True)


def _section_desc(text: str) -> None:
    st.markdown(f"<div class='section-desc'>{_safe(text)}</div>", unsafe_allow_html=True)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .fd-grade-badge {
            display:inline-flex; align-items:center; justify-content:center;
            width:64px; height:64px; border-radius:50%; font-size:1.5rem; font-weight:800;
            font-family:'JetBrains Mono', monospace; border:2px solid rgba(148,163,184,0.25);
            background: rgba(15,23,42,0.55); margin:0.6rem auto 0;
        }
        .fd-section-title { font-weight:700; color:#F1F5F9; margin-top:0.7rem; margin-bottom:0.3rem; font-size:0.9rem; }
        .fd-radar-legend-swatch {
            display:inline-block; width:10px; height:10px; border-radius:50%;
            margin-right:0.4rem; vertical-align:middle;
        }
        .fd-radar-side-card {
            background: rgba(26,16,48,0.5); border:1px solid rgba(167,139,250,0.16);
            border-radius:12px; padding:1rem 1.1rem; height:100%;
        }
        .fd-radar-side-title { font-weight:700; color:#F1F5F9; font-size:0.85rem; margin-bottom:0.55rem; }
        .fd-insight-good { color:#4ADE80; font-size:0.84rem; margin:0.2rem 0; }
        .fd-insight-bad { color:#FB7185; font-size:0.84rem; margin:0.2rem 0; }
        .fd-radar-stat-row { display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.9rem; }
        .fd-radar-stat-chip {
            flex:1; min-width:120px; background: rgba(26,16,48,0.5);
            border:1px solid rgba(167,139,250,0.14); border-radius:12px;
            padding:0.8rem 1rem; text-align:center;
        }
        /* Download Report bar — top of the Developer Portal. Same visual
           language as the PR Summary download bar in app.py's global CSS
           (.pr-download-bar), duplicated here as a scoped fallback in case
           this module is ever rendered standalone. */
        .fd-download-bar {
            display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.75rem;
            background: rgba(35, 20, 66, 0.55);
            border: 1px solid rgba(167, 139, 250, 0.22);
            border-radius: 14px; padding: 1rem 1.3rem; margin-bottom: 1.3rem;
        }
        .fd-download-bar .fd-download-title { font-weight:700; color:#F8FAFC; font-size:1.05rem; }
        .fd-download-bar .fd-download-sub { color:#C3CEEA; font-size:0.82rem; margin-top:0.15rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _grade_from_score(score: int) -> str:
    if score >= 95:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B+"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C+"
    if score >= 50:
        return "C"
    if score >= 35:
        return "D"
    return "F"


def _grade_color(grade: str) -> str:
    return _GRADE_COLORS.get(grade, "#94A3B8")


def _sqrt_dampened_penalty(findings: list, weights: dict) -> float:
    """
    Shared normalization primitive: groups findings by severity, then
    charges weight[severity] * sqrt(count) per severity bucket instead of
    weight[severity] * count. This keeps volume from either (a) flooring
    every "bad" case to the same 0 score, or (b) letting many low-severity
    findings outscore a single high-severity one.
    """
    counts: dict = {}
    for f in findings:
        sev = f.get("severity", "Medium")
        counts[sev] = counts.get(sev, 0) + 1
    return sum(weights.get(sev, weights.get("Medium", 2)) * math.sqrt(n) for sev, n in counts.items())


def _compute_security_only_score(findings: list) -> int:
    """
    A presentation-only variant of the same severity-weighted penalty
    already used elsewhere in the pipeline (orchestrator._summarize),
    scoped to security findings only, so the portal can show a distinct
    'Security Score' next to the overall 'Code Health Score'.

    Uses sqrt-dampened penalties (see _sqrt_dampened_penalty) so this is a
    normalized 0-100 score, not a flat linear subtraction.
    """
    security_findings = [f for f in findings if f.get("agent") == "security"]
    if not security_findings:
        return 100
    weights = {"Critical": 10, "High": 5, "Medium": 2, "Low": 1, "Info": 0}
    penalty = _sqrt_dampened_penalty(security_findings, weights)
    return max(0, min(100, round(100 - penalty)))


def _compute_domain_only_score(findings: list, agent_key: str) -> int:
    """
    Same normalized, sqrt-dampened penalty formula as
    _compute_security_only_score, generalized to any agent domain, so Code
    Quality gets a comparable headline score.
    """
    domain_findings = [f for f in findings if f.get("agent") == agent_key]
    if not domain_findings:
        return 100
    weights = {"Critical": 10, "High": 5, "Medium": 2, "Low": 1, "Info": 0}
    penalty = _sqrt_dampened_penalty(domain_findings, weights)
    return max(0, min(100, round(100 - penalty)))


def _tool_counts(findings: list) -> dict:
    counts: dict = {}
    for f in findings:
        for tool in (f.get("detected_by") or [f.get("tool") or "unknown"]):
            counts[tool] = counts.get(tool, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- #
# Security Posture Radar — dimension bucketing
# --------------------------------------------------------------------------- #
# Ordered so the first keyword match wins. Purely a presentation-layer
# reclassification of fields the findings already carry (category, title,
# description/message, owasp/owasp_category, cwe/cwe_id) — no new analysis,
# no new fields required from run_analysis().
_DIMENSION_KEYWORDS = [
    ("Injection & Input Validation", (
        "injection", "sql", "xss", "cross-site scripting", "csrf",
        "command injection", "input validation", "sanitiz", "untrusted input",
    )),
    ("Auth & Access Control", (
        "auth", "password", "session", "token", "access control",
        "privilege", "credential", "login", "permission",
    )),
    ("Data Protection", (
        "encrypt", "sensitive data", "data exposure", "pii", "hardcoded secret",
        "hardcoded key", "plaintext", "cleartext", "secret", "cryptograph",
    )),
    ("Secure Configuration", (
        "config", "deserializ", "xxe", "ssrf", "misconfigur", "insecure default",
        "debug mode", "cors",
    )),
    ("Error Handling", (
        "exception", "error handling", "try/except", "catch", "unhandled",
        "stack trace",
    )),
    ("Code Complexity", (
        "complexity", "cyclomatic", "nested", "long method", "long function",
    )),
    ("Maintainability", (
        "duplicate", "dead code", "naming", "unused", "style", "convention",
        "readab", "maintainab",
    )),
]

_DIMENSION_ORDER = [name for name, _ in _DIMENSION_KEYWORDS] + ["Best Practices"]

_SEVERITY_PENALTY = {"Critical": 25, "High": 15, "Medium": 8, "Low": 4, "Info": 1}


def _classify_finding_dimension(f: dict) -> str:
    haystack = " ".join(
        str(f.get(field) or "")
        for field in ("category", "title", "description", "message", "owasp", "owasp_category", "cwe", "cwe_id")
    ).lower()

    for dimension, keywords in _DIMENSION_KEYWORDS:
        if any(kw in haystack for kw in keywords):
            return dimension

    # Fallback: unmatched security findings read as configuration/hardening
    # gaps, unmatched code-quality findings read as general best practices.
    if f.get("agent") == "security":
        return "Secure Configuration"
    return "Best Practices"


def _compute_dimension_scores(findings: list) -> dict:
    """
    One severity-weighted, NORMALIZED health score (0-100) per posture
    dimension, covering every finding exactly once.

    Normalization: for each dimension, findings are grouped by severity
    and charged weight[severity] * sqrt(count) per severity bucket (see
    _sqrt_dampened_penalty), instead of a flat weight * count sum. This
    fixes two problems the flat model had:
      1. Two dimensions with very different finding counts (e.g. 4 vs 10
         Critical findings) no longer both collapse to the same 0 floor —
         they now score distinctly worse as volume increases.
      2. A dimension with many Low-severity findings no longer outscores
         (i.e. shows as "worse than") a dimension with a single Critical
         finding — severity still dominates the ranking, volume only
         adjusts it within that ranking.
    """
    findings_by_dim: dict = {dim: [] for dim in _DIMENSION_ORDER}
    for f in findings:
        findings_by_dim[_classify_finding_dimension(f)].append(f)

    scores = {}
    for dim, dim_findings in findings_by_dim.items():
        if not dim_findings:
            scores[dim] = 100
            continue
        penalty = _sqrt_dampened_penalty(dim_findings, _SEVERITY_PENALTY)
        scores[dim] = max(0, round(100 - penalty))

    return scores


def _dimension_finding_counts(findings: list) -> dict:
    counts = {dim: 0 for dim in _DIMENSION_ORDER}
    for f in findings:
        counts[_classify_finding_dimension(f)] += 1
    return counts


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _render_gauge(score: int, chart_key: str) -> None:
    if score >= 80:
        bar_color = "#4ADE80"
    elif score >= 50:
        bar_color = "#F59E0B"
    else:
        bar_color = "#F43F5E"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 30, "color": "#F8FAFC", "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#64748B", "size": 10}},
            "bar": {"color": bar_color, "thickness": 0.32},
            "bgcolor": "rgba(15,23,42,0.4)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(244,63,94,0.10)"},
                {"range": [50, 80], "color": "rgba(245,158,11,0.10)"},
                {"range": [80, 100], "color": "rgba(74,222,128,0.10)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif", color="#CBD5E1"),
        margin=dict(l=0, r=10, t=10, b=0),
        height=180,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=chart_key)


def _render_security_posture_radar(findings: list, by_severity: dict) -> None:
    """
    The single graph in the Dashboard section. Presentation-only
    aggregation over the already-computed findings list (see
    _classify_finding_dimension / _compute_dimension_scores) — no new
    analysis, no new agent calls, no new fields required upstream.

    Labels shown:
      - Title + caption: what the graph measures
      - "My Code" vs "Target Baseline": the two plotted series
      - Overall Security Posture Score: average of the 8 (normalized)
        dimension scores
      - Key Insights: strongest / weakest dimensions
      - Recommendation: points at the weakest dimension
      - Total Findings + full severity breakdown (Critical / High /
        Medium / Low+Info) — each defaults to 0 via .get() if that
        severity never occurred, it does not error or get skipped.
    """
    dimension_scores = _compute_dimension_scores(findings)
    dimension_counts = _dimension_finding_counts(findings)
    total = sum(dimension_counts.values())
    overall_score = round(sum(dimension_scores.values()) / len(dimension_scores)) if dimension_scores else 100

    axes = _DIMENSION_ORDER
    theta = axes + [axes[0]]
    profile_values = [dimension_scores[a] for a in axes] + [dimension_scores[axes[0]]]
    baseline_values = [90] * len(axes) + [90]  # industry-style reference line

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=baseline_values, theta=theta,
        name="Target Baseline",
        line=dict(color="#F472B6", width=2, dash="dash"),
        fill="none",
        marker=dict(size=4, color="#F472B6"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=profile_values, theta=theta,
        name="My Code",
        line=dict(color="#A78BFA", width=2.5),
        fillcolor="rgba(167,139,250,0.24)", fill="toself",
        marker=dict(size=6, color="#C026D3"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif", color="#CBD5E1"),
        polar=dict(
            bgcolor="rgba(26,16,48,0.4)",
            radialaxis=dict(
                visible=True, range=[0, 100], showticklabels=True,
                tickfont=dict(color="#7C7195", size=9),
                gridcolor="rgba(167,139,250,0.16)",
            ),
            angularaxis=dict(
                tickfont=dict(color="#E2E8F0", size=11.5),
                gridcolor="rgba(167,139,250,0.16)",
            ),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.1, x=0.5, xanchor="center",
            font=dict(color="#E2E8F0", size=11),
        ),
        margin=dict(l=50, r=50, t=55, b=20),
        height=460,
    )

    radar_col, side_col = st.columns([2, 1])

    with radar_col:
        st.markdown(
            "<div class='premium-card'><div class='metric-label'>"
            "Security Posture Radar — 8-Dimension Profile</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="fd_security_posture_radar")
        st.markdown(
            f"<div style='color:#94A3B8; font-size:0.8rem; padding-top:0.3rem;'>"
            f"Closer to the outer edge means fewer/less-severe findings in that "
            f"dimension. The dashed line marks a 90/100 reference target. "
            f"<b style='color:#E2E8F0;'>Overall Security Posture Score: {overall_score}/100</b>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    ranked = sorted(dimension_scores.items(), key=lambda kv: -kv[1])
    strongest = [d for d, s in ranked[:2]]
    weakest = [d for d, s in ranked[-2:]][::-1]

    with side_col:
        st.markdown(
            "<div class='fd-radar-side-card'>"
            "<div class='fd-radar-side-title'>Key Insights</div>",
            unsafe_allow_html=True,
        )
        for dim in strongest:
            st.markdown(
                f"<div class='fd-insight-good'>&#9679; Strong: {_safe(dim)} "
                f"({dimension_scores[dim]}/100)</div>",
                unsafe_allow_html=True,
            )
        for dim in weakest:
            st.markdown(
                f"<div class='fd-insight-bad'>&#9679; Needs attention: {_safe(dim)} "
                f"({dimension_scores[dim]}/100)</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div class='fd-radar-side-title' style='margin-top:0.9rem;'>Recommendation</div>", unsafe_allow_html=True)
        if dimension_scores[weakest[0]] < 90:
            st.markdown(
                f"<div style='color:#CBD5E1; font-size:0.84rem;'>"
                f"Prioritize findings under <b>{_safe(weakest[0])}</b> "
                f"({dimension_counts.get(weakest[0], 0)} finding(s)) to close the "
                f"biggest gap to the target baseline.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='color:#CBD5E1; font-size:0.84rem;'>"
                "All dimensions are at or above the target baseline.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Full severity breakdown — Critical shown on its own, defaulting to 0
    # when absent (never omitted or errored on).
    st.markdown(
        f"""
        <div class="fd-radar-stat-row">
            <div class="fd-radar-stat-chip"><div class="metric-icon">Total Findings</div><div class="metric-value">{total}</div></div>
            <div class="fd-radar-stat-chip"><div class="metric-icon">Critical</div><div class="metric-value" style="color:#FB7185">{by_severity.get('Critical', 0)}</div></div>
            <div class="fd-radar-stat-chip"><div class="metric-icon">High</div><div class="metric-value" style="color:#FB923C">{by_severity.get('High', 0)}</div></div>
            <div class="fd-radar-stat-chip"><div class="metric-icon">Medium</div><div class="metric-value" style="color:#FBBF24">{by_severity.get('Medium', 0)}</div></div>
            <div class="fd-radar-stat-chip"><div class="metric-icon">Low</div><div class="metric-value" style="color:#38BDF8">{by_severity.get('Low', 0) + by_severity.get('Info', 0)}</div></div>
            <div class="fd-radar-stat-chip"><div class="metric-icon">Overall Score</div><div class="metric-value">{overall_score}/100</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# 0. Download Report — TOP of the Developer Portal
# --------------------------------------------------------------------------- #
def _render_download_report_bar(deep_result: dict, result: dict, state_key: str) -> None:
    """
    Prominent "Download Report" action at the very top of the Developer
    Portal.

    IMPORTANT: this does not generate a new/second PDF. It calls the exact
    same two functions, with the exact same inputs, that the PR Summary tab
    already uses to build its report:
        agents.prsummaryagent.generate_pr_summary(...)
        pr_summary_pdf.generate_pr_pdf(...)
    Given the same deep_result/code/file_name, this produces the same
    findings + remediation PDF — it is simply surfaced here as well so a
    developer doesn't have to switch tabs to get it.
    """
    code = (result or {}).get("code", "")
    file_name = (result or {}).get("file_name") or deep_result.get("file") or "analysis"

    st.markdown(
        "<div class='fd-download-bar'>"
        "<div><div class='fd-download-title'>📄 Full Findings &amp; Remediation Report</div>"
        "<div class='fd-download-sub'>Complete PDF report — severity breakdown, every finding, "
        "root cause, and remediation guidance.</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        api_key = (st.session_state.get("gemini_api_key") or "").strip()
        use_llm = llm_router.any_configured()

        pr_data = prsummaryagent.generate_pr_summary(
            deep_result,
            code=code,
            file_name=file_name,
            api_key=api_key,
            use_llm=use_llm,
        )

        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        pdf_filename = f"{file_name}_pr_summary_report.pdf"
        pdf_out_path = os.path.join(temp_dir, pdf_filename)

        pdf_path = generate_pr_pdf(
            deep_result,
            pr_data,
            pdf_out_path,
            project_name=file_name or "Code Analysis Project",
        )

        if pdf_path and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            with open(pdf_path, "rb") as fh:
                st.download_button(
                    "⬇️ Download Report",
                    fh.read(),
                    file_name=pdf_filename,
                    mime="application/pdf",
                    key=f"download_report_devportal_{state_key}",
                    type="primary",
                )
        else:
            st.error("Failed to generate the report PDF. File was not created.")
    except Exception as exc:
        st.error(f"Unable to generate report PDF: {exc}")


# --------------------------------------------------------------------------- #
# 1. Code Health Header
# --------------------------------------------------------------------------- #
def _render_code_health_header(deep_result: dict, findings: list, summary: dict, by_severity: dict) -> None:
    total = summary.get("total", 0)
    code_health_score = summary.get("security_score", 100)
    security_score = _compute_security_only_score(findings)
    grade = _grade_from_score(code_health_score)
    merge = prsummaryagent.decide_merge_recommendation(by_severity)

    _section_head("Code Health")
    header_col, gauge_col = st.columns([2.3, 1])

    with header_col:
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-chip"><div class="metric-icon">Code Health</div><div class="metric-value">{code_health_score}/100</div></div>
            <div class="metric-chip"><div class="metric-icon">Security Score</div><div class="metric-value">{security_score}/100</div></div>
            <div class="metric-chip"><div class="metric-icon">Grade</div><div class="metric-value">{grade}</div></div>
            <div class="metric-chip"><div class="metric-icon">Total Findings</div><div class="metric-value">{total}</div></div>
            <div class="metric-chip"><div class="metric-icon">Critical</div><div class="metric-value" style="color:#FB7185">{by_severity.get('Critical', 0)}</div></div>
            <div class="metric-chip"><div class="metric-icon">High</div><div class="metric-value" style="color:#FB923C">{by_severity.get('High', 0)}</div></div>
            <div class="metric-chip"><div class="metric-icon">Medium</div><div class="metric-value" style="color:#FBBF24">{by_severity.get('Medium', 0)}</div></div>
            <div class="metric-chip"><div class="metric-icon">Low</div><div class="metric-value" style="color:#38BDF8">{by_severity.get('Low', 0)}</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown(
            f"<div class='premium-card' style='padding:0.9rem 1.2rem;'>{merge['icon']} "
            f"<b>{_safe(merge['label'])}</b> — {_safe(merge['description'])}</div>",
            unsafe_allow_html=True,
        )

    with gauge_col:
        st.markdown("<div class='premium-card' style='text-align:center;'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Overall Code Health</div>", unsafe_allow_html=True)
        _render_gauge(code_health_score, chart_key="fd_health_gauge")
        st.markdown(
            f"<div class='fd-grade-badge' style='color:{_grade_color(grade)}; "
            f"border-color:{_grade_color(grade)}55;'>{grade}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 2. Dashboard Chart
# --------------------------------------------------------------------------- #
def _render_dashboard_charts(findings: list, by_severity: dict, show_radar: bool = True) -> None:
    """
    Exactly one graph renders here: the Security Posture Radar. The
    separate Severity Distribution bar chart has been removed — its
    numbers are already surfaced as stat chips under the radar, so a
    second chart was redundant.

    `show_radar` scopes this single chart to whichever caller wants it
    (e.g. only the Developer Portal tab).
    """
    _section_head("Dashboard")

    if show_radar:
        _render_security_posture_radar(findings, by_severity)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def render_findings_display(deep_result: dict, result: dict, state_key: str, show_radar: bool = True) -> None:
    _inject_styles()

    if not deep_result:
        st.markdown(
            "<div class='premium-card' style='text-align:center; color:#94A3B8;'>"
            "Run Deep Analysis to populate the Developer Portal.</div>",
            unsafe_allow_html=True,
        )
        return

    findings = deep_result.get("findings", []) or []
    summary = deep_result.get("summary") or {"total": 0, "by_severity": {}, "security_score": 100}
    by_severity = summary.get("by_severity", {})

    # ---- 1. Download Report — top of the Developer Portal. Reuses the
    # exact same PR Summary PDF generation call (see docstring above). ----
    _render_download_report_bar(deep_result, result, state_key)

    # ---- 2. Code Health (unchanged) ----
    _render_code_health_header(deep_result, findings, summary, by_severity)

    # ---- 3. Dashboard / Security Posture Radar (unchanged) ----
    _render_dashboard_charts(findings, by_severity, show_radar=show_radar)

    # ---- The numbered "Findings (N)" list and all of its per-finding
    # detail (Root Cause, Recommendation, Code Snippet, Remediation,
    # Before/After, Why This Fix Works, Secure Coding Guideline,
    # Prevention Tip) has been intentionally removed from this portal.
    # That detail is not duplicated here — it lives in the PDF report
    # above, which already contains it in full. ----
    