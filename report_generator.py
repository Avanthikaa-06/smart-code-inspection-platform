"""
JSON Security Report — colorized edition.

The report content is still exactly the same JSON payload as before
(same fields, same structure, nothing renamed or removed) — this file
only changes *how it's rendered*:

  - The JSON is syntax-highlighted like a code editor (keys, strings,
    numbers, booleans each get their own color).
  - Every "severity" value is colored to match the same red/orange/
    amber/gray/blue scale already used in the Streamlit dashboard, so
    the PDF and the live app look like the same product.
  - Each finding gets a colored left-hand strip (its severity color),
    so a reader can gauge the shape of the report at a glance without
    reading a single line of JSON.
  - A short color legend sits at the top so nobody has to guess what
    the colors mean.

Public API is unchanged: generate_pdf_report(report, submitted_code) -> bytes
"""
import json
import re
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SEVERITIES = ["Critical", "High", "Medium", "Low", "Info"]

# Same palette as the Streamlit dashboard's severity chips — the PDF
# should look like it came from the same product, not a different tool.
_SEVERITY_COLORS = {
    "Critical": "#F43F5E",
    "High": "#F97316",
    "Medium": "#F59E0B",
    "Low": "#94A3B8",
    "Info": "#38BDF8",
}
_BRAND_DARK = HexColor("#101B36")
_BRAND_BLUE = HexColor("#2563EB")
_PANEL_BG = HexColor("#0F172A")

# JSON syntax-highlight palette (VS Code "Dark+"-ish)
_KEY_COLOR = "#7DD3FC"        # light blue
_STRING_COLOR = "#86EFAC"     # green
_NUMBER_COLOR = "#FCD34D"     # amber
_BOOL_NULL_COLOR = "#C4B5FD"  # violet
_PUNCT_COLOR = "#64748B"      # slate


# --------------------------------------------------------------------------- #
# Payload shaping (unchanged from before — same fields, same source data)
# --------------------------------------------------------------------------- #
def _value(*values, default=None):
    for value in values:
        if value not in (None, ""):
            return value
    return default


def _severity_counts(summary: dict) -> dict:
    by_severity = summary.get("by_severity", {}) if isinstance(summary, dict) else {}
    return {severity: by_severity.get(severity, 0) for severity in SEVERITIES}


def _finding_payload(finding: dict) -> dict:
    return {
        "tool": _value(finding.get("tool"), default="unknown"),
        "severity": _value(finding.get("severity"), default="Medium"),
        "line": _value(finding.get("line"), finding.get("line_start")),
        "title": _value(finding.get("title"), finding.get("category"), default="Finding"),
        "description": _value(finding.get("description"), finding.get("message"), default=""),
        "cwe": _value(finding.get("cwe"), finding.get("cwe_id")),
        "owasp": _value(finding.get("owasp"), finding.get("owasp_category")),
        "recommendation": _value(
            finding.get("recommendation"),
            default="Review and remediate this finding before release.",
        ),
    }


def _report_payload(report: dict) -> dict:
    source = report if isinstance(report, dict) else {}
    summary = source.get("summary", {})
    findings = source.get("findings", [])
    return {
        "file": source.get("file", "submitted source"),
        "language": source.get("language", "unknown"),
        "analysis_time": source.get("analysis_time") or datetime.now().isoformat(timespec="seconds"),
        "security_score": summary.get("security_score", 100),
        "total_findings": summary.get("total", len(findings)),
        "severity_counts": _severity_counts(summary),
        "findings": [_finding_payload(finding) for finding in findings],
    }


# --------------------------------------------------------------------------- #
# JSON syntax highlighting — turns json.dumps() output into colored XML
# markup that reportlab's Paragraph can render, line by line, so it still
# reads exactly like formatted JSON (same brackets, same indentation).
# --------------------------------------------------------------------------- #
def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _color_scalar(token: str) -> str:
    """Colorize a single JSON scalar token: "string", number, true/false/null, or a bracket/brace."""
    stripped = token.strip()
    if not stripped:
        return token

    trailing_comma = stripped.endswith(",")
    core = stripped[:-1] if trailing_comma else stripped
    comma = "," if trailing_comma else ""

    if core.startswith('"') and core.endswith('"') and len(core) >= 2:
        value = core[1:-1]
        # Special case: severity values get the same color as the app's
        # own severity chips, not the generic string color.
        color = _SEVERITY_COLORS.get(value, _STRING_COLOR)
        return f'<font color="{color}">"{_xml_escape(value)}"</font>{comma}'
    if core in ("true", "false"):
        return f'<font color="{_BOOL_NULL_COLOR}">{core}</font>{comma}'
    if core == "null":
        return f'<font color="{_BOOL_NULL_COLOR}">null</font>{comma}'
    if re.match(r"^-?\d+(\.\d+)?$", core):
        return f'<font color="{_NUMBER_COLOR}">{core}</font>{comma}'
    if core in ("{", "}", "[", "]", "{}", "[]"):
        return f'<font color="{_PUNCT_COLOR}">{_xml_escape(core)}</font>{comma}'
    # Fallback (shouldn't normally hit this for standard json.dumps output)
    return f"{_xml_escape(core)}{comma}"


def _highlight_json_line(line: str) -> str:
    """Colorize one line of json.dumps(..., indent=2) output."""
    key_match = re.match(r'^(\s*)"([^"]+)":\s*(.*)$', line)
    if key_match:
        indent, key, rest = key_match.groups()
        indent_html = indent.replace(" ", "&nbsp;")
        colored_key = f'<font color="{_KEY_COLOR}">"{_xml_escape(key)}"</font>'
        colored_rest = _color_scalar(rest) if rest else ""
        sep = ":&nbsp;" if rest else ":"
        return f"{indent_html}{colored_key}{sep}{colored_rest}"

    # A line that's just a bracket/brace/array element (no "key": prefix)
    leading_ws = len(line) - len(line.lstrip(" "))
    indent_html = ("&nbsp;" * leading_ws)
    return indent_html + _color_scalar(line.strip())


def _highlighted_json_paragraphs(payload: dict, style: ParagraphStyle) -> list:
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    return [Paragraph(_highlight_json_line(line) or "&nbsp;", style) for line in json_text.splitlines()]


# --------------------------------------------------------------------------- #
# PDF assembly
# --------------------------------------------------------------------------- #
def _build_styles():
    mono = ParagraphStyle(name="JsonMono", fontName="Courier", fontSize=8, leading=11, textColor=colors.white)
    title = ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=18, textColor=colors.white, spaceAfter=4)
    subtitle = ParagraphStyle(name="ReportSubtitle", fontName="Helvetica", fontSize=10, textColor=HexColor("#94A3B8"), spaceAfter=2)
    legend = ParagraphStyle(name="Legend", fontName="Helvetica", fontSize=8, textColor=HexColor("#CBD5E1"))
    section = ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=11, textColor=HexColor("#7DD3FC"), spaceBefore=10, spaceAfter=4)
    return {"mono": mono, "title": title, "subtitle": subtitle, "legend": legend, "section": section}


def _score_color(score: int) -> str:
    if score >= 90:
        return "#22C55E"
    if score >= 70:
        return "#38BDF8"
    if score >= 40:
        return "#F59E0B"
    return "#F43F5E"


def _legend_row(styles) -> Table:
    cells = []
    for sev in SEVERITIES:
        color = _SEVERITY_COLORS[sev]
        cells.append(Paragraph(f'<font color="{color}">&#9632;</font> {sev}', styles["legend"]))
    t = Table([cells], colWidths=[3.2 * cm] * len(cells))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _PANEL_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _header_block(payload: dict, styles) -> list:
    story = [
        Paragraph("AI Code Review &amp; Security Analysis — JSON Report", styles["title"]),
        Paragraph(
            f'{payload["file"]} &middot; {str(payload["language"]).title()} &middot; {payload["analysis_time"]}',
            styles["subtitle"],
        ),
    ]
    score = payload["security_score"]
    score_color = _score_color(score)
    counts = payload["severity_counts"]
    count_text = " &nbsp;|&nbsp; ".join(
        f'<font color="{_SEVERITY_COLORS[s]}">{s}: {counts[s]}</font>' for s in SEVERITIES
    )
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        f'<font color="{score_color}"><b>Security Score: {score}/100</b></font> '
        f'&nbsp;&middot;&nbsp; Total findings: {payload["total_findings"]} '
        f'&nbsp;&middot;&nbsp; {count_text}',
        styles["subtitle"],
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Legend — severity color key:", styles["legend"]))
    story.append(_legend_row(styles))
    return story


def _metadata_json_block(payload: dict, styles) -> list:
    """Header metadata rendered as its own small colorized JSON block
    (file, language, time, score, counts) — same JSON, just isolated
    from the (potentially long) findings array so it reads at a glance."""
    meta = {k: v for k, v in payload.items() if k != "findings"}
    paragraphs = _highlighted_json_paragraphs(meta, styles["mono"])
    table = Table([[p] for p in paragraphs], colWidths=[24 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _PANEL_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, _BRAND_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return [Paragraph("Report Metadata (JSON)", styles["section"]), table]


def _finding_json_block(finding: dict, index: int, styles) -> Table:
    """One finding, still literally its own JSON object, with a colored
    left strip (severity color) so the report is scannable without
    reading text — same idea as a git-diff gutter, applied to JSON."""
    severity = finding.get("severity", "Medium")
    strip_color = HexColor(_SEVERITY_COLORS.get(severity, "#94A3B8"))

    paragraphs = _highlighted_json_paragraphs(finding, styles["mono"])
    json_cell = Table([[p] for p in paragraphs], colWidths=[22.5 * cm])
    json_cell.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))

    row = Table([[f"#{index}", json_cell]], colWidths=[0.9 * cm, 22.9 * cm])
    row.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), strip_color),
        ("BACKGROUND", (1, 0), (1, 0), _PANEL_BG),
        ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
        ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (0, 0), 9),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, strip_color),
    ]))
    return row


def generate_pdf_report(report: dict, submitted_code: str = "") -> bytes:
    payload = _report_payload(report)
    styles = _build_styles()

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=24,
        leftMargin=24,
        topMargin=24,
        bottomMargin=24,
        title="JSON Security Report",
    )

    story = []
    story.append(_solid_background_spacer())
    story.extend(_header_block(payload, styles))
    story.append(Spacer(1, 0.3 * cm))
    story.extend(_metadata_json_block(payload, styles))
    story.append(Spacer(1, 0.35 * cm))

    if payload["findings"]:
        story.append(Paragraph(f'Findings (JSON, {len(payload["findings"])} total)', styles["section"]))
        for i, finding in enumerate(payload["findings"], start=1):
            story.append(_finding_json_block(finding, i, styles))
            story.append(Spacer(1, 0.18 * cm))
    else:
        story.append(Paragraph("No findings — empty JSON array.", styles["legend"]))

    doc.build(story, onFirstPage=_paint_page_background, onLaterPages=_paint_page_background)
    return buffer.getvalue()


def _solid_background_spacer():
    # Spacer placeholder kept for story ordering; actual background is
    # painted per-page via onFirstPage/onLaterPages below so it covers
    # the full page including margins.
    return Spacer(1, 0.01 * cm)


def _paint_page_background(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(_BRAND_DARK)
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], stroke=0, fill=1)
    canvas.restoreState()