from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"

class Category(str, Enum):
    VULNERABILITY = "Vulnerability"
    CODE_SMELL = "Code Smell"
    COMPLEXITY = "Complexity"
    DESIGN_ISSUE = "Design Issue"

@dataclass
class Finding:
    id: str
    agent_source: str
    tool: str
    file: str
    language: str
    line_start: int
    line_end: int
    category: Category
    title: str
    description: str
    severity: Severity
    confidence: str = "medium"
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> dict:
        category_value = self.category.value if hasattr(self.category, "value") else str(self.category)
        severity_value = self.severity.value if hasattr(self.severity, "value") else str(self.severity)
        return {
            "agent": "security" if self.agent_source == "security_vulnerability" else "code_analysis",
            "agent_source": self.agent_source,
            "tool": self.tool,
            "file": self.file,
            "language": self.language,
            "line": self.line_start,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "category": self.title,
            "taxonomy": category_value,
            "title": self.title,
            "description": self.description,
            "severity": severity_value,
            "message": self.description,
            "confidence": self.confidence,
            "cwe": self.cwe_id,
            "cwe_id": self.cwe_id,
            "owasp": self.owasp_category,
            "owasp_category": self.owasp_category,
            "code_snippet": self.code_snippet,
            "recommendation": self.recommendation,
            "source": "tool",
        }

@dataclass
class ToolStatus:
    tool: str
    agent: str
    status: str
    message: Optional[str] = None
    findings_count: int = 0
    duration_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "agent": self.agent,
            "status": self.status,
            "message": self.message,
            "findings_count": self.findings_count,
            "duration_seconds": self.duration_seconds,
        }

@dataclass
class AgentResult:
    agent_name: str
    status: str = "success"
    findings: list = field(default_factory=list)
    error: Optional[str] = None
    tool_statuses: list = field(default_factory=list)
