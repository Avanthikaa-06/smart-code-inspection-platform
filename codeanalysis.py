"""
CodeAnalysisAgent
------------------
Self-contained agent responsible for code-quality / design analysis.

Directly imports and executes:
    - ast                            (built-in, in-process AST analysis)
    - pylint.lint.Run                (Python API, in-process — no subprocess)
    - radon.complexity.cc_visit      (Python API, in-process — no subprocess)
    - radon.metrics.mi_visit         (Python API, in-process — no subprocess)
    - pmd                            (subprocess — PMD has no official Python API)

No wrapper/helper module is used — every tool invocation, its severity
mapping, and finding normalization live in this file.
"""

import ast
import csv
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Callable, List, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from pylint.lint import Run as PylintRun
from pylint.reporters.json_reporter import JSONReporter
from radon.complexity import cc_rank, cc_visit
from radon.metrics import mi_visit

from schemas import AgentResult, Category, Finding, Severity, ToolStatus

# --------------------------------------------------------------------------- #
# Severity maps (tool-specific vocab -> internal Severity enum)
# --------------------------------------------------------------------------- #
PYLINT_SEVERITY_MAP = {
    "fatal": Severity.CRITICAL,
    "error": Severity.HIGH,
    "warning": Severity.MEDIUM,
    "convention": Severity.LOW,
    "refactor": Severity.MEDIUM,
    "info": Severity.INFO,
}
PMD_SEVERITY_MAP = {
    "1": Severity.CRITICAL,
    "2": Severity.HIGH,
    "3": Severity.MEDIUM,
    "4": Severity.LOW,
    "5": Severity.LOW,
}


# --------------------------------------------------------------------------- #
# Local execution helpers (self-contained; not shared with other agents)
# --------------------------------------------------------------------------- #
def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _venv_scripts_dir() -> str:
    return os.path.join(_repo_root(), ".venv", "Scripts")


def _resolve_executable(name: str, env_var: str = None) -> Optional[str]:
    if env_var and os.environ.get(env_var):
        configured = os.environ[env_var]
        if shutil.which(configured) or os.path.exists(configured):
            return configured
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        os.path.join(_venv_scripts_dir(), name),
        os.path.join(_venv_scripts_dir(), f"{name}.exe"),
        os.path.join(_venv_scripts_dir(), f"{name}.bat"),
        os.path.join(_venv_scripts_dir(), f"{name}.cmd"),
    ]
    return next((path for path in candidates if os.path.exists(path)), None)


def _run_command(cmd: list, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _write_code_to_disk(code: str, language: str) -> str:
    suffix = ".py" if (language or "").lower() == "python" else ".java"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="sandbox_eval_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code or "")
    return path


def _snippet(filepath: str, line: int, radius: int = 2) -> Optional[str]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        if not lines:
            return None
        start = max(1, int(line or 1) - radius)
        end = min(len(lines), int(line or 1) + radius)
        return "\n".join(f"{idx:>4} {lines[idx - 1]}" for idx in range(start, end + 1))
    except Exception:
        return None


def _snippet_from_source(code: str, line: int, radius: int = 2) -> Optional[str]:
    """
    Same idea as _snippet(), but reads from the in-memory source string
    instead of a file path. Needed because the LLM design-review pass
    runs against a logical filename (e.g. "submitted.py") that never
    actually exists on disk, so _snippet() would silently return None
    for every LLM-sourced finding.
    """
    try:
        lines = (code or "").splitlines()
        if not lines:
            return None
        start = max(1, int(line or 1) - radius)
        end = min(len(lines), int(line or 1) + radius)
        return "\n".join(f"{idx:>4} {lines[idx - 1]}" for idx in range(start, end + 1))
    except Exception:
        return None


def _tool_status(tool: str, agent: str, status: str, message: str = None,
                  findings_count: int = 0, duration_seconds: float = None) -> ToolStatus:
    return ToolStatus(
        tool=tool,
        agent=agent,
        status=status,
        message=message,
        findings_count=int(findings_count or 0),
        duration_seconds=round(duration_seconds, 3) if duration_seconds is not None else None,
    )


def _run_tool(tool: str, agent: str, callback: Callable[[], list]) -> tuple:
    start = time.perf_counter()
    try:
        findings = callback()
        duration = time.perf_counter() - start
        return findings, _tool_status(tool, agent, "success", findings_count=len(findings), duration_seconds=duration)
    except FileNotFoundError as exc:
        duration = time.perf_counter() - start
        return [], _tool_status(tool, agent, "skipped", str(exc), 0, duration)
    except subprocess.TimeoutExpired:
        duration = time.perf_counter() - start
        return [], _tool_status(tool, agent, "failed", "Tool execution timed out.", 0, duration)
    except Exception as exc:
        duration = time.perf_counter() - start
        return [], _tool_status(tool, agent, "failed", str(exc), 0, duration)


def _recommendation_for(tool: str, category: Category) -> str:
    if category == Category.COMPLEXITY:
        return "Reduce branching and split the code into smaller units with focused responsibilities."
    if tool == "pylint":
        return "Address the linter warning or document why the current implementation is intentional."
    if tool == "pmd":
        return "Refactor the Java construct according to the rule guidance and project coding standards."
    return "Review and remediate this issue before release."


def _standardize_finding(item, *, file: str, language: str, category: Category = Category.CODE_SMELL) -> dict:
    """Normalize a raw Finding/dict into the shared output schema for this agent."""
    data = item.to_dict() if hasattr(item, "to_dict") else dict(item)

    title = data.get("title") or data.get("category") or "Finding"
    description = data.get("description") or data.get("message") or ""
    line = int(data.get("line_start") or data.get("line") or 1)
    severity_raw = data.get("severity")
    severity = severity_raw.value if isinstance(severity_raw, Severity) else str(severity_raw or "Medium").capitalize()
    tool = data.get("tool") or "unknown"
    cwe = data.get("cwe_id") or data.get("cwe")
    owasp = data.get("owasp_category") or data.get("owasp")

    return {
        "id": data.get("id") or f"{tool.upper()}-{abs(hash((tool, title, line, description))) % 100000:05d}",
        "agent": "code_analysis",
        "agent_source": "code_analysis",
        "tool": tool,
        "file": file,
        "language": (language or "").lower(),
        "line": line,
        "line_start": line,
        "line_end": int(data.get("line_end") or line),
        "category": title,
        "taxonomy": data.get("taxonomy") or (category.value if hasattr(category, "value") else str(category)),
        "title": title,
        "description": description,
        "message": description,
        "severity": severity,
        "confidence": data.get("confidence", "medium"),
        "cwe": cwe,
        "cwe_id": cwe,
        "owasp": owasp,
        "owasp_category": owasp,
        "code_snippet": data.get("code_snippet") or _snippet(file, line),
        "recommendation": data.get("recommendation") or _recommendation_for(tool, category),
        "source": data.get("source", "tool"),
    }


# --------------------------------------------------------------------------- #
# Tool: pylint — direct Python API (pylint.lint.Run), no subprocess
# --------------------------------------------------------------------------- #
def _run_pylint(filepath: str, id_prefix="PYL") -> list:
    """
    Runs Pylint in-process via pylint.lint.Run + JSONReporter, capturing the
    report into an in-memory buffer instead of shelling out to the `pylint`
    CLI and parsing stdout.
    """
    output_buffer = io.StringIO()
    reporter = JSONReporter(output_buffer)
    try:
        # exit=False prevents pylint.lint.Run from calling sys.exit() with
        # the lint score as an exit code, which would otherwise raise
        # SystemExit inside this process.
        PylintRun([filepath, "--score=n"], reporter=reporter, exit=False)
    except SystemExit:
        # Defensive: older/newer pylint versions have varied slightly on
        # whether exit=False fully suppresses SystemExit.
        pass

    raw_output = output_buffer.getvalue()
    data = json.loads(raw_output) if raw_output.strip() else []

    findings = []
    for i, result in enumerate(data):
        line = result.get("line", 1)
        findings.append(Finding(
            id=f"{id_prefix}-{i+1:04d}",
            agent_source="code_analysis",
            tool="pylint",
            file=filepath,
            language="python",
            line_start=line,
            line_end=line,
            category=Category.CODE_SMELL,
            title=result.get("symbol", "Code Smell").replace("-", " ").title(),
            description=result.get("message", ""),
            severity=PYLINT_SEVERITY_MAP.get(str(result.get("type", "warning")).lower(), Severity.LOW),
            confidence="high",
            code_snippet=_snippet(filepath, line),
            recommendation="Fix the Pylint issue or suppress it with a narrow justification when appropriate.",
        ))
    return findings


# --------------------------------------------------------------------------- #
# Tool: radon — direct Python API (cc_visit + mi_visit), no subprocess
# --------------------------------------------------------------------------- #
def _run_radon(filepath: str, id_prefix="RAD") -> list:
    """
    Runs Radon in-process:
      - cc_visit()  -> cyclomatic complexity per function/method/class
      - mi_visit()  -> file-level maintainability index

    Both are called as plain Python function calls against the source text;
    no `radon` CLI subprocess is spawned.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    findings, idx = [], 0

    # --- Cyclomatic complexity ---
    for block in cc_visit(source):
        rank = cc_rank(block.complexity)
        if rank in ("A", "B"):
            continue
        idx += 1
        line = getattr(block, "lineno", 1)
        block_kind = "Method" if getattr(block, "is_method", False) else "Function"
        findings.append(Finding(
            id=f"{id_prefix}-{idx:04d}",
            agent_source="code_analysis",
            tool="radon",
            file=filepath,
            language="python",
            line_start=line,
            line_end=getattr(block, "endline", line),
            category=Category.COMPLEXITY,
            title=f"High Cyclomatic Complexity ({block.name})",
            description=f"{block_kind} '{block.name}' has complexity rank {rank} ({block.complexity}).",
            severity=Severity.HIGH if rank in ("E", "F") else Severity.MEDIUM,
            confidence="high",
            code_snippet=_snippet(filepath, line),
            recommendation="Split this code path into smaller functions and simplify conditional branches.",
        ))

    # --- Maintainability index (file-level) ---
    try:
        mi_score = mi_visit(source, multi=True)
    except Exception:
        mi_score = None
    if isinstance(mi_score, (int, float)) and mi_score < 65:
        idx += 1
        findings.append(Finding(
            id=f"{id_prefix}-{idx:04d}",
            agent_source="code_analysis",
            tool="radon",
            file=filepath,
            language="python",
            line_start=1,
            line_end=1,
            category=Category.COMPLEXITY,
            title="Low Maintainability Index",
            description=f"File-level maintainability index is {mi_score:.1f} (below the 65 'moderate' threshold).",
            severity=Severity.HIGH if mi_score < 40 else Severity.MEDIUM,
            confidence="medium",
            code_snippet=_snippet(filepath, 1),
            recommendation="Reduce overall file complexity and length, and improve documentation coverage to raise the maintainability index.",
        ))

    return findings


# --------------------------------------------------------------------------- #
# Tool: ast (native Python AST, in-process — no subprocess needed)
# --------------------------------------------------------------------------- #
class _PythonAstVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.findings = []
        self._idx = 0

    def _add(self, node, title: str, description: str, severity: Severity):
        self._idx += 1
        line = getattr(node, "lineno", 1)
        self.findings.append(Finding(
            id=f"AST-{self._idx:04d}",
            agent_source="code_analysis",
            tool="python-ast",
            file=self.filepath,
            language="python",
            line_start=line,
            line_end=getattr(node, "end_lineno", line),
            category=Category.CODE_SMELL,
            title=title,
            description=description,
            severity=severity,
            confidence="high",
            code_snippet=_snippet(self.filepath, line),
            recommendation="Refactor the construct into a smaller, documented, and easier-to-test unit.",
        ))

    def visit_FunctionDef(self, node):
        if ast.get_docstring(node) is None:
            self._add(node, "Missing Function Docstring", f"Function '{node.name}' has no docstring.", Severity.LOW)
        arg_count = len(node.args.args) + len(node.args.kwonlyargs)
        if arg_count > 5:
            self._add(node, "Long Parameter List", f"Function '{node.name}' accepts {arg_count} parameters.", Severity.MEDIUM)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def _run_python_ast(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        tree = ast.parse(f.read())
    visitor = _PythonAstVisitor(filepath)
    visitor.visit(tree)
    return visitor.findings


# --------------------------------------------------------------------------- #
# Tool: pmd (Java)
# --------------------------------------------------------------------------- #
def _run_pmd(code: str, logical_file: str = "submitted.java", id_prefix: str = "PMD") -> list:
    exe = _resolve_executable("pmd", "PMD_CMD") or _resolve_executable("pmd.bat", "PMD_CMD")
    if not exe:
        raise FileNotFoundError("PMD executable was not found. Set PMD_CMD or add pmd to PATH.")

    filepath = _write_code_to_disk(code, "java")
    try:
        ruleset = os.environ.get("PMD_RULESET", "category/java/bestpractices.xml,category/java/errorprone.xml")
        res = _run_command([exe, "check", "-d", filepath, "-R", ruleset, "-f", "csv", "--no-cache"], timeout=60)
        if res.returncode not in (0, 4):
            raise RuntimeError((res.stderr or "PMD failed.").strip())

        lines = res.stdout.strip().splitlines()
        if len(lines) <= 1:
            return []
        findings = []
        for i, row in enumerate(csv.DictReader(lines)):
            line = int(row.get("Line") or row.get("line") or 1)
            priority = row.get("Priority") or row.get("priority") or "3"
            rule = row.get("Rule") or row.get("rule") or "PMD Finding"
            problem = row.get("Problem") or row.get("problem") or ""
            ruleset_name = row.get("Rule set") or row.get("Ruleset") or row.get("rule set") or "PMD"
            findings.append(Finding(
                id=f"{id_prefix}-{i+1:04d}",
                agent_source="code_analysis",
                tool="pmd",
                file=logical_file,
                language="java",
                line_start=line,
                line_end=line,
                category=Category.CODE_SMELL,
                title=rule.replace("_", " ").replace("-", " ").title(),
                description=f"{problem} (Rule set: {ruleset_name})".strip(),
                severity=PMD_SEVERITY_MAP.get(str(priority).strip(), Severity.MEDIUM),
                confidence="high",
                code_snippet=_snippet(filepath, line),
                recommendation="Apply the PMD rule guidance and refactor the reported Java code.",
            ))
        return findings
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


# --------------------------------------------------------------------------- #
# Optional LLM design review (unchanged behavior, kept local to this agent)
# --------------------------------------------------------------------------- #
class SingleLLMDesignFinding(BaseModel):
    title: str = Field(..., description="Short name of the issue found")
    severity: Literal["Critical", "High", "Medium", "Low", "Info"] = Field(..., description="Risk severity level")
    line_start: int = Field(..., description="Line number where issue begins")
    line_end: int = Field(..., description="Line number where issue ends")
    description: str = Field(..., description="Human-readable description of structural debt")
    recommendation: str = Field(..., description="Concrete remediation guidance")


class LLMCodeAnalysisOutput(BaseModel):
    findings: List[SingleLLMDesignFinding] = Field(default_factory=list, description="Extracted design flaws")


def _run_langchain_design_analysis(code: str, language: str, filepath: str, api_key: str) -> List[dict]:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=api_key.strip(),
            temperature=0.1,
        )
        structured_llm = llm.with_structured_output(LLMCodeAnalysisOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a senior software architect reviewing {language} source code. "
                "Report maintainability, reliability, complexity, and design problems only. "
                "Do not report security vulnerabilities."
            )),
            ("human", "Review this {language} source code:\n\n```{language}\n{code}\n```"),
        ])
        response: LLMCodeAnalysisOutput = (prompt | structured_llm).invoke({"language": language, "code": code})

        findings = []
        for idx, item in enumerate(response.findings):
            try:
                severity = Severity(item.severity)
            except ValueError:
                severity = Severity.MEDIUM
            findings.append(_standardize_finding(
                Finding(
                    id=f"LLM-ANLYS-{idx+1:04d}",
                    agent_source="code_analysis",
                    tool="langchain-gemini",
                    file=filepath,
                    language=language.lower(),
                    line_start=max(1, item.line_start),
                    line_end=max(item.line_start, item.line_end),
                    category=Category.DESIGN_ISSUE,
                    title=item.title.strip(),
                    description=item.description,
                    severity=severity,
                    confidence="medium",
                    recommendation=item.recommendation,
                    code_snippet=_snippet_from_source(code, item.line_start),
                ),
                file=filepath,
                language=language,
                category=Category.DESIGN_ISSUE,
            ))
        return findings
    except Exception as exc:
        raise RuntimeError(f"LangChain design analysis failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Public entry point used by the orchestrator
# --------------------------------------------------------------------------- #
def analyze(code: str, filepath: str, language: str, api_key: str = None, use_llm: bool = True) -> AgentResult:
    if not code or not code.strip():
        return AgentResult(agent_name="code_analysis", status="success", findings=[], error="Empty code. Analysis bypassed.")

    lang = (language or "").strip().lower()
    findings, statuses = [], []

    if lang == "python":
        tmp_path = _write_code_to_disk(code, "python")
        try:
            for tool_name, callback, category in [
                ("pylint", lambda: _run_pylint(tmp_path), Category.CODE_SMELL),
                ("radon", lambda: _run_radon(tmp_path), Category.COMPLEXITY),
                ("python-ast", lambda: _run_python_ast(tmp_path), Category.CODE_SMELL),
            ]:
                local_findings, status = _run_tool(tool_name, "code_analysis", callback)
                statuses.append(status.to_dict())
                findings.extend(_standardize_finding(item, file=filepath, language=lang, category=category) for item in local_findings)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    elif lang == "java":
        local_findings, status = _run_tool("pmd", "code_analysis", lambda: _run_pmd(code, filepath))
        statuses.append(status.to_dict())
        findings.extend(_standardize_finding(item, file=filepath, language=lang, category=Category.CODE_SMELL) for item in local_findings)
    else:
        return AgentResult(agent_name="code_analysis", status="failed", findings=[], error=f"Unsupported language: {language}")

    if use_llm and api_key and api_key.strip():
        llm_findings, status = _run_tool(
            "langchain-gemini",
            "code_analysis",
            lambda: _run_langchain_design_analysis(code, lang, filepath, api_key),
        )
        statuses.append(status.to_dict())
        findings.extend(llm_findings)

    failed = [s for s in statuses if s["status"] == "failed"]
    return AgentResult(
        agent_name="code_analysis",
        status="partial" if failed else "success",
        findings=findings,
        error="; ".join(s["message"] for s in failed if s.get("message")) or None,
        tool_statuses=statuses,
    )