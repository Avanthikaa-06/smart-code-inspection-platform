"""
SecurityAnalysisAgent
----------------------
Self-contained agent responsible for security / vulnerability analysis.

Directly imports and executes:
    - bandit.core.manager / config / test_set  (Python API, in-process —
      no subprocess; Bandit ships an official importable API)
    - semgrep     (subprocess — no stable, officially supported Python API
                   for running analyses; CLI remains the supported surface)
    - javalang    (in-process Java AST parser, used for Java security-relevant
                   structural analysis)

No wrapper/helper module is used — every tool invocation, its severity
mapping, and finding normalization live in this file.
"""

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

from bandit.core import config as bandit_config
from bandit.core import manager as bandit_manager
from bandit.core import test_set as bandit_test_set

from schemas import AgentResult, Category, Finding, Severity, ToolStatus

# --------------------------------------------------------------------------- #
# Severity maps (tool-specific vocab -> internal Severity enum)
# --------------------------------------------------------------------------- #
SEMGREP_SEVERITY_MAP = {"ERROR": Severity.CRITICAL, "WARNING": Severity.HIGH, "INFO": Severity.MEDIUM}
BANDIT_SEVERITY_MAP = {"HIGH": Severity.CRITICAL, "MEDIUM": Severity.HIGH, "LOW": Severity.MEDIUM}
SPOTBUGS_PRIORITY_MAP = {"1": Severity.HIGH, "2": Severity.MEDIUM, "3": Severity.LOW}


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
    instead of a file path. Needed because the LLM security audit pass
    runs AFTER the temp file has already been deleted in analyze()'s
    finally block, so _snippet(tmp_path, ...) would silently return None
    for every LLM-sourced finding — starving the Remediation Agent of
    code_snippet, which it relies on for before_snippet.
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


def _standardize_finding(item, *, file: str, language: str, category: Category = Category.VULNERABILITY) -> dict:
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
        "agent": "security",
        "agent_source": "security_vulnerability",
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
        "recommendation": data.get("recommendation")
        or "Review the vulnerable code path, validate untrusted input, and use the safer API recommended by the reporting tool.",
        "source": data.get("source", "tool"),
    }


# --------------------------------------------------------------------------- #
# Tool: semgrep
# --------------------------------------------------------------------------- #
def _run_semgrep(filepath: str, language: str, id_prefix="SEM") -> list:
    exe = _resolve_executable("semgrep")
    if not exe:
        raise FileNotFoundError("Semgrep executable was not found.")

    configs = ["p/owasp-top-ten", "p/security-audit"]
    cmd = [exe, "--json", "--quiet"]
    for config in configs:
        cmd += ["--config", config]
    cmd.append(filepath)

    res = _run_command(cmd, timeout=120)
    if res.returncode not in (0, 1):
        raise RuntimeError((res.stderr or "Semgrep failed.").strip())
    data = json.loads(res.stdout) if res.stdout else {"results": []}

    findings = []
    for i, result in enumerate(data.get("results", [])):
        extra = result.get("extra", {})
        meta = extra.get("metadata", {})
        cwe = meta.get("cwe")
        owasp = meta.get("owasp")
        line = result.get("start", {}).get("line", 1)
        findings.append(Finding(
            id=f"{id_prefix}-{i+1:04d}",
            agent_source="security_vulnerability",
            tool="semgrep",
            file=filepath,
            language=(language or "").lower(),
            line_start=line,
            line_end=result.get("end", {}).get("line", line),
            category=Category.VULNERABILITY,
            title=result.get("check_id", "Semgrep Finding").split(".")[-1].replace("-", " ").title(),
            description=extra.get("message", "Semgrep reported a security issue."),
            severity=SEMGREP_SEVERITY_MAP.get(str(extra.get("severity", "WARNING")).upper(), Severity.MEDIUM),
            confidence=str(meta.get("confidence", "medium")).lower(),
            cwe_id=", ".join(cwe) if isinstance(cwe, list) else cwe,
            owasp_category=", ".join(owasp) if isinstance(owasp, list) else owasp,
            code_snippet=extra.get("lines") or _snippet(filepath, line),
            recommendation="Follow the Semgrep rule guidance and replace the unsafe construct with a vetted safe API.",
        ))
    return findings


# --------------------------------------------------------------------------- #
# Tool: bandit — direct Python API (bandit.core.manager/config/test_set),
# no subprocess
# --------------------------------------------------------------------------- #
def _run_bandit(filepath: str, id_prefix="BAN") -> list:
    """
    Runs Bandit in-process using its official core API instead of shelling
    out to the `bandit` CLI:
      - config.BanditConfig()   builds an (empty/default) Bandit config
      - test_set.BanditTestSet() explicitly resolves the active plugin/test
        set against that config, so we know up front which checks will run
      - manager.BanditManager() discovers the target file, executes the
        resolved test set against it, and exposes issues as Issue objects
    """
    b_conf = bandit_config.BanditConfig()
    profile = {"include": set(), "exclude": set()}
    b_ts = bandit_test_set.BanditTestSet(b_conf, profile)

    mgr = bandit_manager.BanditManager(b_conf, "file")
    # Use the explicitly-built test set rather than the one BanditManager
    # would otherwise construct internally from the same config/profile.
    mgr.b_ts = b_ts
    mgr.discover_files([filepath])
    mgr.run_tests()

    findings = []
    for i, result in enumerate(mgr.get_issue_list()):
        data = result.as_dict()
        cwe = data.get("issue_cwe")
        line = data.get("line_number", 1)
        findings.append(Finding(
            id=f"{id_prefix}-{i+1:04d}",
            agent_source="security_vulnerability",
            tool="bandit",
            file=filepath,
            language="python",
            line_start=line,
            line_end=line,
            category=Category.VULNERABILITY,
            title=data.get("test_name", "Bandit Finding").replace("_", " ").title(),
            description=data.get("issue_text", ""),
            severity=BANDIT_SEVERITY_MAP.get(str(data.get("issue_severity", "MEDIUM")).upper(), Severity.MEDIUM),
            confidence=str(data.get("issue_confidence", "medium")).lower(),
            cwe_id=f"CWE-{cwe['id']}" if cwe and cwe.get("id") else None,
            code_snippet=data.get("code") or _snippet(filepath, line),
            recommendation="Apply the Bandit test guidance and remove the insecure Python construct.",
        ))
    return findings


# --------------------------------------------------------------------------- #
# Tool: javalang (Java AST — security-relevant structural analysis)
# --------------------------------------------------------------------------- #
def _java_class_name(code: str) -> str:
    try:
        import javalang

        tree = javalang.parse.parse(code)
        for type_decl in tree.types:
            if getattr(type_decl, "name", None):
                return type_decl.name
    except Exception:
        pass
    return "Submitted"


def _run_javalang_ast(code: str, logical_file: str = "submitted.java", id_prefix: str = "JVL") -> list:
    try:
        import javalang
    except ImportError as exc:
        raise FileNotFoundError("javalang is not installed.") from exc

    tree = javalang.parse.parse(code)
    findings, idx = [], 0

    # Flag risky Java constructs that javalang can see directly in the AST.
    for _, node in tree.filter(javalang.tree.MethodInvocation):
        if node.member in ("exec", "Runtime.exec"):
            idx += 1
            line = node.position.line if node.position else 1
            findings.append(Finding(
                id=f"{id_prefix}-{idx:04d}",
                agent_source="security_vulnerability",
                tool="javalang",
                file=logical_file,
                language="java",
                line_start=line,
                line_end=line,
                category=Category.VULNERABILITY,
                title="Potential Command Injection",
                description=f"Call to '{node.member}' can allow arbitrary command execution if arguments are not sanitized.",
                severity=Severity.HIGH,
                confidence="medium",
                cwe_id="CWE-78",
                recommendation="Avoid invoking shell processes with unsanitized input; use safe APIs or strict allow-lists.",
            ))

    for _, method in tree.filter(javalang.tree.MethodDeclaration):
        line = method.position.line if method.position else 1
        parameter_count = len(method.parameters or [])
        if parameter_count > 5:
            idx += 1
            findings.append(Finding(
                id=f"{id_prefix}-{idx:04d}",
                agent_source="security_vulnerability",
                tool="javalang",
                file=logical_file,
                language="java",
                line_start=line,
                line_end=line,
                category=Category.CODE_SMELL,
                title="Long Parameter List",
                description=f"Method '{method.name}' accepts {parameter_count} parameters, increasing misuse risk at call sites.",
                severity=Severity.MEDIUM,
                confidence="high",
                recommendation="Introduce a parameter object or split this method into smaller responsibilities.",
            ))
    return findings


# --------------------------------------------------------------------------- #
# Optional LLM security audit (unchanged behavior, kept local to this agent)
# --------------------------------------------------------------------------- #
class SingleLLMSecurityFinding(BaseModel):
    title: str = Field(..., description="Vulnerability name")
    severity: Literal["Low", "Medium", "High", "Critical"] = Field(..., description="Risk severity level")
    line: Optional[int] = Field(None, description="Line number or null if global")
    description: str = Field(..., description="Explanation and exploit vector")
    recommendation: str = Field(..., description="Concrete remediation guidance")
    cwe: Optional[str] = Field(None, description="CWE id, for example CWE-89")
    owasp: Optional[str] = Field(None, description="OWASP category")


class LLMSecurityAnalysisOutput(BaseModel):
    findings: List[SingleLLMSecurityFinding] = Field(default_factory=list, description="Identified vulnerabilities")


def _run_langchain_security_audit(code: str, language: str, filepath: str, api_key: str) -> list:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=api_key.strip(),
            temperature=0.1,
        )
        structured_llm = llm.with_structured_output(LLMSecurityAnalysisOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an application security static-analysis reviewer for {language}. "
                "Report exploitable security weaknesses only. Do not report style or design issues."
            )),
            ("human", "Audit this {language} source code:\n\n```{language}\n{code}\n```"),
        ])
        response: LLMSecurityAnalysisOutput = (prompt | structured_llm).invoke({"language": language, "code": code})
        return [{
            "agent": "security",
            "agent_source": "security_vulnerability",
            "tool": "langchain-gemini",
            "file": filepath,
            "language": language,
            "line": item.line or 1,
            "line_start": item.line or 1,
            "line_end": item.line or 1,
            "category": item.title,
            "taxonomy": "Vulnerability",
            "title": item.title,
            "description": item.description,
            "message": item.description,
            "severity": item.severity,
            "cwe": item.cwe,
            "cwe_id": item.cwe,
            "owasp": item.owasp,
            "owasp_category": item.owasp,
            "recommendation": item.recommendation,
            "confidence": "medium",
            "code_snippet": _snippet_from_source(code, item.line or 1),
            "source": "llm",
        } for item in response.findings]
    except Exception as exc:
        raise RuntimeError(f"LangChain security audit failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Public entry points used by the orchestrator
# --------------------------------------------------------------------------- #
def analyze(code: str, filepath: str, language: str, api_key: str = None, use_llm: bool = True) -> AgentResult:
    if not code or not code.strip():
        return AgentResult(agent_name="security_vulnerability", status="success", findings=[], error="Empty code. Analysis bypassed.")

    lang = (language or "").strip().lower()
    findings, statuses = [], []
    tmp_path = _write_code_to_disk(code, lang)
    try:
        tool_plan = [("semgrep", lambda: _run_semgrep(tmp_path, lang))]
        if lang == "python":
            tool_plan.append(("bandit", lambda: _run_bandit(tmp_path)))
        elif lang == "java":
            tool_plan.append(("javalang", lambda: _run_javalang_ast(code, filepath)))

        else:
            return AgentResult(agent_name="security_vulnerability", status="failed", findings=[], error=f"Unsupported language: {language}")

        for tool_name, callback in tool_plan:
            local_findings, status = _run_tool(tool_name, "security", callback)
            statuses.append(status.to_dict())
            findings.extend(_standardize_finding(item, file=filepath, language=lang) for item in local_findings)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

    if use_llm and api_key and api_key.strip():
        llm_findings, status = _run_tool(
            "langchain-gemini",
            "security",
            lambda: _run_langchain_security_audit(code, lang, filepath, api_key),
        )
        statuses.append(status.to_dict())
        findings.extend(llm_findings)

    failed = [s for s in statuses if s["status"] == "failed"]
    return AgentResult(
        agent_name="security_vulnerability",
        status="partial" if failed else "success",
        findings=findings,
        error="; ".join(s["message"] for s in failed if s.get("message")) or None,
        tool_statuses=statuses,
    )


def scan(filepath: str, language: str) -> AgentResult:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return analyze(f.read(), filepath, language, api_key=None, use_llm=False)
    except Exception as exc:
        return AgentResult(agent_name="security_vulnerability", status="failed", findings=[], error=f"Scanner exception: {exc}")