import ast
import os
from dataclasses import dataclass
from typing import Optional


SUPPORTED_LANGUAGES = {"python", "java"}


@dataclass
class LanguageDetectionResult:
    language: str
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def _extension_language(file_name: Optional[str]) -> Optional[str]:
    ext = os.path.splitext(file_name or "")[1].lower()
    if ext == ".py":
        return "python"
    if ext == ".java":
        return "java"
    return None


def _valid_python(code: str) -> bool:
    try:
        ast.parse(code or "")
        return bool((code or "").strip())
    except SyntaxError:
        return False


def _valid_java(code: str) -> bool:
    if not (code or "").strip():
        return False
    try:
        import javalang

        javalang.parse.parse(code)
        return True
    except Exception:
        return False


def detect_language(code: str, file_name: Optional[str] = None) -> LanguageDetectionResult:
    code = code or ""
    ext_lang = _extension_language(file_name)
    py_ok = _valid_python(code)
    java_ok = _valid_java(code)

    if ext_lang == "python" and py_ok:
        return LanguageDetectionResult("python", 0.98, "Python extension and syntax parse matched.")
    if ext_lang == "java" and java_ok:
        return LanguageDetectionResult("java", 0.98, "Java extension and javalang parse matched.")
    if py_ok and not java_ok:
        confidence = 0.9 if ext_lang in (None, "python") else 0.75
        return LanguageDetectionResult("python", confidence, "Python syntax parse succeeded.")
    if java_ok and not py_ok:
        confidence = 0.9 if ext_lang in (None, "java") else 0.75
        return LanguageDetectionResult("java", confidence, "Java syntax parse succeeded.")
    if ext_lang:
        return LanguageDetectionResult(ext_lang, 0.55, "Detected from file extension; syntax validation did not fully pass.")

    return LanguageDetectionResult("unknown", 0.0, "Unable to detect Python or Java from syntax or filename.")
