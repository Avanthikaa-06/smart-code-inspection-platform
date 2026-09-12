"""
pr_summary_pdf.py
------------------
Renders the PR Summary `data` dict (the output of
agents/prsummaryagent.py::generate_pr_summary) as a professional,
developer-focused PDF — modeled on a GitHub/PR security review report
rather than an analytics dashboard.

Structure:
  Page 1  — Header, merge recommendation, severity KPI counts, ONE simple
            severity-distribution chart, code quality assessment,
            executive overview.
  Page 2+ — Prioritized findings, one card per finding: severity, file,
            line, title, description, root cause (if available),
            recommendation, and code snippet (if available).
  Final   — Remediation roadmap, bucketed by urgency (falls back to a
            flat developer checklist if bucketed data isn't available).

Nothing here recomputes backend numbers — everything is read straight out
of `data` / `deep_result`.

pip install reportlab
"""

from datetime import datetime, timezone
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    HRFlowable, Preformatted, KeepTogether,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import HorizontalBarChart

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
C_CRITICAL = colors.HexColor("#B91C1C")
C_HIGH = colors.HexColor("#F97316")
C_MEDIUM = colors.HexColor("#EAB308")
C_LOW = colors.HexColor("#3B82F6")
C_INFO = colors.HexColor("#22C55E")
C_GREEN = colors.HexColor("#22C55E")
C_RED = colors.HexColor("#EF4444")
C_ORANGE = colors.HexColor("#F97316")
C_BLUE = colors.HexColor("#2563EB")
C_GREY = colors.HexColor("#6B7280")
C_DARK = colors.HexColor("#111827")
C_CARD_BG = colors.HexColor("#F3F4F6")
C_CODE_BG = colors.HexColor("#F8FAFC")
C_ROW_ALT = colors.HexColor("#F9FAFB")
C_BORDER = colors.HexColor("#E5E7EB")

SEVERITY_COLOR = {"Critical": C_CRITICAL, "High": C_HIGH, "Medium": C_MEDIUM, "Low": C_LOW, "Info": C_INFO}

_HEALTH_STATUS_COLOR = {
    "green": C_GREEN,
    "yellow": C_MEDIUM,
    "orange": C_ORANGE,
    "red": C_RED,
}

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ReportTitle", fontSize=24, leading=28, textColor=C_DARK,
                           fontName="Helvetica-Bold", spaceAfter=4))
styles.add(ParagraphStyle("ReportSubtitle", fontSize=11, leading=15, textColor=C_GREY,
                           fontName="Helvetica"))
styles.add(ParagraphStyle("SectionHeading", fontSize=15, leading=19, textColor=colors.white,
                           fontName="Helvetica-Bold", spaceAfter=8))
styles.add(ParagraphStyle("SubHeading", fontSize=11.5, leading=14, textColor=C_DARK,
                           fontName="Helvetica-Bold", spaceBefore=2, spaceAfter=4))
styles.add(ParagraphStyle("Body", fontSize=9.5, leading=14, textColor=C_DARK, fontName="Helvetica"))
styles.add(ParagraphStyle("BodyMuted", fontSize=8.5, leading=12, textColor=C_GREY, fontName="Helvetica"))
styles.add(ParagraphStyle("CardValue", fontSize=20, leading=22, textColor=colors.white,
                           fontName="Helvetica-Bold", alignment=TA_CENTER))
styles.add(ParagraphStyle("CardLabel", fontSize=8, leading=10, textColor=colors.white,
                           fontName="Helvetica", alignment=TA_CENTER))
styles.add(ParagraphStyle("CodeLabel", fontSize=8, leading=10, textColor=C_GREY,
                           fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=2))
styles.add(ParagraphStyle("SnippetCode", fontName="Courier", fontSize=7.5, leading=9.5, textColor=C_DARK))
styles.add(ParagraphStyle("SevBadge", fontSize=9, leading=11, textColor=colors.white,
                           fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SevLoc", fontSize=8.5, leading=11, textColor=colors.white,
                           fontName="Helvetica", alignment=TA_LEFT))
styles.add(ParagraphStyle("ChecklistItem", fontSize=9, leading=13, textColor=C_DARK, fontName="Helvetica"))
styles.add(ParagraphStyle("HealthLabel", fontSize=9, leading=12, textColor=colors.white,
                           fontName="Helvetica-Bold", alignment=TA_CENTER))
styles.add(ParagraphStyle("HealthSub", fontSize=8, leading=10, textColor=colors.white,
                           fontName="Helvetica", alignment=TA_CENTER))
styles.add(ParagraphStyle("RoadmapHeading", fontSize=10.5, leading=13, textColor=colors.white,
                           fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("RoadmapItem", fontSize=9, leading=13, textColor=C_DARK, fontName="Helvetica"))


def _p(text) -> str:
    """Escape arbitrary finding/description text before it goes into a Paragraph."""
    return _xml_escape(str(text if text not in (None, "") else "N/A"))


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #
def _section_header(title: str, bg=C_DARK) -> Table:
    t = Table([[Paragraph(title, styles["SectionHeading"])]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    return t


def _kpi_card(label: str, value: str, color) -> Table:
    inner = Table([[Paragraph(str(value), styles["CardValue"])],
                    [Paragraph(label.upper(), styles["CardLabel"])]])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
    ]))
    return inner


def _kpi_row(cards: list) -> Table:
    col_w = CONTENT_W / len(cards)
    row = [_kpi_card(label, value, color) for label, value, color in cards]
    t = Table([row], colWidths=[col_w] * len(cards))
    t.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _code_quality_block(data: dict, width=CONTENT_W) -> Table:
    """
    Standalone "Code Quality Assessment" panel — pulls straight from
    prsummaryagent's already-computed `repository_health` (score, stars,
    status) and `code_quality_overview` (non-security finding breakdown,
    maintainability flags). Nothing here is recomputed.
    """
    health = data.get("repository_health") or {}
    quality = data.get("code_quality_overview") or {}

    score = health.get("score", 100)
    stars = health.get("stars", 5)
    status_label = health.get("status_label", "N/A")
    status_color = _HEALTH_STATUS_COLOR.get(health.get("color"), C_GREY)
    star_str = "★" * stars + "☆" * (5 - stars)

    # Left cell: score + stars + status, on a colored panel
    left = Table(
        [
            [Paragraph(f"{score}/100", styles["CardValue"])],
            [Paragraph(star_str, styles["HealthSub"])],
            [Paragraph(_p(status_label).upper(), styles["HealthLabel"])],
        ],
        colWidths=[width * 0.32],
    )
    left.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), status_color),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("TOPPADDING", (0, 2), (-1, 2), 4),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
    ]))

    # Right cell: non-security quality breakdown + maintainability flags
    breakdown = quality.get("breakdown") or {}
    maint_flags = quality.get("maintainability_flags", 0)
    total_quality = quality.get("total", 0)

    right_lines = [Paragraph(
        f"<b>Code Quality Findings:</b> {total_quality} "
        f"(excludes security-agent findings)", styles["Body"]
    )]
    if breakdown:
        breakdown_str = "  ·  ".join(f"{cat}: {count}" for cat, count in breakdown.items())
        right_lines.append(Paragraph(_p(breakdown_str), styles["BodyMuted"]))
    if maint_flags:
        right_lines.append(Paragraph(
            f"<b>{maint_flags}</b> finding(s) flag maintainability concerns.", styles["Body"]
        ))
    else:
        right_lines.append(Paragraph("No maintainability concerns flagged.", styles["Body"]))

    right = Table([[r] for r in right_lines], colWidths=[width * 0.64])
    right.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    outer = Table([[left, right]], colWidths=[width * 0.34, width * 0.66])
    outer.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return outer


def _code_block(snippet: str, width=CONTENT_W) -> Table:
    """Monospace, greyed-out code box — the only place raw text bypasses
    Paragraph's XML escaping, since Preformatted renders text literally."""
    pre = Preformatted(snippet.strip("\n") or "(snippet unavailable)", styles["SnippetCode"])
    t = Table([[pre]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, C_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _severity_distribution_chart(risk_distribution: list, width=CONTENT_W, height=110) -> Drawing:
    """The single retained chart: a simple horizontal bar of finding
    counts per severity — enough for a reviewer to gauge risk at a
    glance without a page full of decorative analytics."""
    rows = [r for r in risk_distribution if r["count"] > 0]
    d = Drawing(width, height)
    if not rows:
        return d
    chart = HorizontalBarChart()
    chart.x, chart.y = 60, 10
    chart.width, chart.height = width - 90, height - 20
    chart.data = [[r["count"] for r in rows]]
    chart.categoryAxis.categoryNames = [r["severity"] for r in rows]
    chart.categoryAxis.labels.fontSize = 8.5
    chart.valueAxis.valueMin = 0
    chart.barWidth = 12
    chart.bars[0].fillColor = C_BLUE
    for i, r in enumerate(rows):
        chart.bars[(0, i)].fillColor = SEVERITY_COLOR[r["severity"]]
    d.add(chart)
    return d


# --------------------------------------------------------------------------- #
# Page 1 — Summary
# --------------------------------------------------------------------------- #
def _page_summary(deep_result: dict, data: dict, project_name: str) -> list:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    by_sev = data["severity_breakdown"]

    flow = [
        Paragraph("Pull Request Security &amp; Code Review Report", styles["ReportTitle"]),
        Paragraph(_p(f"{project_name} · {deep_result.get('file', 'N/A')}"), styles["ReportSubtitle"]),
        Spacer(1, 3),
        Paragraph(
            _p(f"Scanned: {generated_at}  ·  Language: {(deep_result.get('language') or 'N/A').capitalize()}"),
            styles["ReportSubtitle"],
        ),
        Spacer(1, 12),
        HRFlowable(width=CONTENT_W, color=C_BORDER, thickness=1),
        Spacer(1, 12),
    ]

    # Merge recommendation banner
    merge = data["merge_recommendation"]
    merge_color = {"red": C_RED, "orange": C_ORANGE, "brightgreen": C_GREEN}.get(merge["color"], C_GREY)
    banner = Table([[Paragraph(f"<b>{_p(merge['label'])}</b> — {_p(merge['description'])}", styles["Body"])]],
                    colWidths=[CONTENT_W])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(merge_color.red, merge_color.green, merge_color.blue, alpha=0.12)),
        ("BOX", (0, 0), (-1, -1), 1, merge_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    flow.append(banner)
    flow.append(Spacer(1, 14))

    # Severity KPI row — Total + Critical/High/Medium/Low counts
    flow.append(_kpi_row([
        ("Total Findings", str(data["total_findings"]), C_DARK),
        ("Critical", str(by_sev.get("Critical", 0)), C_CRITICAL),
        ("High", str(by_sev.get("High", 0)), C_HIGH),
        ("Medium", str(by_sev.get("Medium", 0)), C_MEDIUM),
        ("Low", str(by_sev.get("Low", 0)), C_LOW),
    ]))
    flow.append(Spacer(1, 16))

    # Single severity distribution chart
    if data["total_findings"] > 0:
        flow.append(Paragraph("Severity Distribution", styles["SubHeading"]))
        flow.append(_severity_distribution_chart(data["risk_distribution"]))
        flow.append(Spacer(1, 10))

    # Overall code quality assessment
    flow.append(Paragraph("Code Quality Assessment", styles["SubHeading"]))
    flow.append(_code_quality_block(data))
    flow.append(Spacer(1, 14))

    # Executive overview
    flow.append(_section_header("Executive Summary"))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(_p(data["executive_overview"]), styles["Body"]))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(f"<b>Suggested PR Title:</b> {_p(data['suggested_pr_title'])}", styles["Body"]))
    flow.append(Paragraph(f"<b>Estimated Review Time:</b> {data['review_time_minutes']} min", styles["Body"]))

    # Short, genuinely useful highlights (not decorative — tells the
    # reviewer what's already clean, so they know where NOT to spend time)
    highlights = data.get("positive_highlights") or []
    if highlights:
        flow.append(Spacer(1, 10))
        flow.append(Paragraph("What Went Right", styles["SubHeading"]))
        for h in highlights:
            flow.append(Paragraph(f"✓ {_p(h)}", styles["Body"]))

    return flow


# --------------------------------------------------------------------------- #
# Page 2+ — Prioritized Findings
# --------------------------------------------------------------------------- #
def _finding_card(item: dict) -> KeepTogether:
    severity = item.get("severity", "Medium")
    color = SEVERITY_COLOR.get(severity, C_GREY)

    header = Table(
        [[
            Paragraph(f"#{item['rank']}&nbsp;&nbsp;{_p(severity).upper()}", styles["SevBadge"]),
            Paragraph(_p(f"{item.get('file') or 'N/A'}:{item.get('line', 'N/A')}"), styles["SevLoc"]),
        ]],
        colWidths=[CONTENT_W * 0.32, CONTENT_W * 0.68],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("LEFTPADDING", (0, 0), (0, 0), 10),
        ("LEFTPADDING", (1, 0), (1, 0), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    flow = [header]

    body_rows = [[Paragraph(f"<b>{_p(item.get('title'))}</b>", styles["SubHeading"])]]

    tags = []
    if item.get("cwe"):
        tags.append(f"CWE: {_p(item['cwe'])}")
    if item.get("owasp"):
        tags.append(f"OWASP: {_p(item['owasp'])}")
    if item.get("tools"):
        tags.append("Detected by: " + _p(", ".join(item["tools"])))
    if tags:
        body_rows.append([Paragraph("  ·  ".join(tags), styles["BodyMuted"])])

    body_rows.append([Paragraph(f"<b>Description:</b> {_p(item.get('description'))}", styles["Body"])])
    if item.get("root_cause"):
        body_rows.append([Paragraph(f"<b>Root Cause:</b> {_p(item['root_cause'])}", styles["Body"])])
    body_rows.append([Paragraph(f"<b>Recommendation:</b> {_p(item.get('recommendation'))}", styles["Body"])])

    body = Table(body_rows, colWidths=[CONTENT_W])
    body.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    flow.append(body)

    if item.get("code_snippet"):
        flow.append(Paragraph("CODE SNIPPET", styles["CodeLabel"]))
        flow.append(_code_block(item["code_snippet"]))

    flow.append(Spacer(1, 12))
    return KeepTogether(flow)


def _page_findings(data: dict) -> list:
    prioritized = data.get("prioritized_fixes") or []
    flow = [_section_header(f"Prioritized Findings ({len(prioritized)})"), Spacer(1, 10)]

    if not prioritized:
        flow.append(Paragraph("No findings were surfaced by this scan — nothing to review.", styles["Body"]))
        return flow

    for item in prioritized:
        flow.append(_finding_card(item))

    return flow


# --------------------------------------------------------------------------- #
# Final page — Remediation Roadmap
# --------------------------------------------------------------------------- #
_ROADMAP_BUCKETS = [
    ("immediate_actions", "Immediate Actions (Critical)", C_CRITICAL),
    ("fix_this_week", "Fix This Week (High)", C_HIGH),
    ("can_be_deferred", "Can Be Deferred (Medium)", C_MEDIUM),
    ("nice_to_have", "Nice to Have (Low / Info)", C_LOW),
]


def _roadmap_section(bucket_key: str, heading: str, color, items: list) -> list:
    if not items:
        return []
    header = Table([[Paragraph(heading, styles["RoadmapHeading"])]], colWidths=[CONTENT_W])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    rows = [[header]]
    for entry in items:
        rows.append([Paragraph(f"☐ {_p(entry)}", styles["RoadmapItem"])])
    body = Table(rows, colWidths=[CONTENT_W])
    body.setStyle(TableStyle([
        ("LEFTPADDING", (0, 1), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
    ]))
    return [body, Spacer(1, 10)]


def _page_next_steps(data: dict) -> list:
    """
    Remediation roadmap — buckets findings by urgency (immediate / this
    week / deferrable / nice-to-have) using prsummaryagent's already-built
    `executive_recommendations`, instead of one flat undifferentiated
    checklist. Falls back to the flat `developer_checklist` only if the
    bucketed data is unavailable, so nothing regresses for older data.
    """
    flow = [_section_header("Remediation Roadmap"), Spacer(1, 10)]

    recs = data.get("executive_recommendations") or {}
    has_bucketed = any(recs.get(key) for key, _, _ in _ROADMAP_BUCKETS)

    if has_bucketed:
        for key, heading, color in _ROADMAP_BUCKETS:
            flow += _roadmap_section(key, heading, color, recs.get(key) or [])
        return flow

    # Fallback: flat checklist
    checklist = data.get("developer_checklist") or []
    if not checklist:
        flow.append(Paragraph("No outstanding action items — this change is ready to merge.", styles["Body"]))
        return flow
    for line in checklist:
        flow.append(Paragraph(f"☐ {_p(line.replace('- [ ] ', ''))}", styles["ChecklistItem"]))
    return flow


# --------------------------------------------------------------------------- #
# Header / footer / page numbers
# --------------------------------------------------------------------------- #
def _make_header_footer(project_name: str):
    def _draw(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_DARK)
        canvas.setFont("Helvetica-Bold", 8.5)
        canvas.drawString(MARGIN, PAGE_H - 12 * mm, f"{project_name} — PR Review")
        canvas.setFillColor(C_GREY)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 12 * mm,
                                datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        canvas.setStrokeColor(C_BORDER)
        canvas.line(MARGIN, PAGE_H - 14 * mm, PAGE_W - MARGIN, PAGE_H - 14 * mm)

        canvas.setFillColor(C_GREY)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(PAGE_W / 2, 10 * mm, f"Page {doc.page}")
        canvas.drawString(MARGIN, 10 * mm, "Confidential — Generated by AI Code Review Pipeline")
        canvas.restoreState()
    return _draw


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_pr_pdf(deep_result: dict, data: dict, output_path: str, project_name: str = "PR Review") -> str:
    """
    Renders a lean, developer-focused PR review PDF to `output_path` and
    returns that path. `deep_result` and `data` are exactly what the
    existing pipeline already produces — nothing here recomputes any
    number, it only lays out what matters for a reviewer: merge status,
    severity counts, one severity chart, code quality assessment, full
    finding-level detail, and a bucketed remediation roadmap.
    """
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=20 * mm, bottomMargin=16 * mm,
        title=f"{project_name} — PR Summary Report",
    )

    story = []
    story += _page_summary(deep_result, data, project_name)
    story.append(PageBreak())
    story += _page_findings(data)
    story.append(PageBreak())
    story += _page_next_steps(data)

    draw_header_footer = _make_header_footer(project_name)
    doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    return output_path