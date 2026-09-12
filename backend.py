import os
import re
import shutil
import subprocess
import tempfile
import ast
import requests
from language_detector import detect_language

# ── Code Validation Modules ───────────────────────────────────

def validate_python(code: str) -> dict:
    if not code or not code.strip():
        return {"valid": False, "error_type": "Empty Submission", "message": "No code was provided.", "line": None}
    try:
        ast.parse(code)
        return {"valid": True, "error_type": None, "message": None, "line": None}
    except SyntaxError as e:
        return {"valid": False, "error_type": "Syntax Error", "message": e.msg or "Invalid Python syntax", "line": e.lineno}
    except (ValueError, IndentationError, TabError, Exception) as e:
        return {"valid": False, "error_type": "Invalid Python Code", "message": str(e), "line": getattr(e, "lineno", None)}

def extract_public_class_name(code: str) -> str:
    match = re.search(r'public\s+(?:final\s+|abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)', code or '')
    return match.group(1) if match else "TempClass"

def validate_java(code: str) -> dict:
    if not code or not code.strip():
        return {"valid": False, "error_type": "Empty Submission", "message": "No code was provided.", "line": None}
    
    if shutil.which("javac") is None:
        if "class " not in code:
            return {"valid": False, "error_type": "Syntax Error", "message": "No class definition found.", "line": None}
        if code.count("{") != code.count("}"):
            return {"valid": False, "error_type": "Syntax Error", "message": "Mismatched curly braces.", "line": None}
        return {"valid": True, "error_type": None, "message": None, "line": None}

    temp_dir = tempfile.mkdtemp(prefix="java_validate_")
    file_path = os.path.join(temp_dir, f"{extract_public_class_name(code)}.java")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        res = subprocess.run(["javac", file_path], cwd=temp_dir, capture_output=True, text=True, timeout=20)
        
        if res.returncode == 0:
            return {"valid": True, "error_type": None, "message": None, "line": None}
            
        stderr = res.stderr or "Unknown compilation error"
        line_match = re.search(r'\.java:(\d+):', stderr)
        err_lines = [l for l in stderr.splitlines() if l.strip()]
        msg = re.sub(r'^.*\.java:\d+:\s*(error:)?\s*', '', err_lines[0] if err_lines else stderr).strip()
        
        return {"valid": False, "error_type": "Compilation Error", "message": msg or "Compilation failed", "line": int(line_match.group(1)) if line_match else None}
    except subprocess.TimeoutExpired:
        return {"valid": False, "error_type": "Timeout Error", "message": "Compilation timed out.", "line": None}
    except Exception as e:
        return {"valid": False, "error_type": "Compilation Error", "message": str(e), "line": None}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def validate_code(code: str, language: str) -> dict:
    lang = (language or "").strip().lower()
    if lang in ("python", "java"):
        return validate_python(code) if lang == "python" else validate_java(code)
    return {"valid": False, "error_type": "Unsupported Language", "message": f"Validation not supported for '{language}'.", "line": None}

# ── GitHub Repository & URL Handlers ────────────────────────────

GITHUB_URL_PATTERN = re.compile(r'^https?://github\.com/[\w.-]+/[\w.-]+(\.git)?/?$')
GITHUB_BLOB_PATTERN = re.compile(r'^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$')

def clone_and_extract(repo_url: str, max_files: int = 25) -> dict:
    url = (repo_url or "").strip()
    if not GITHUB_URL_PATTERN.match(url) or shutil.which("git") is None:
        return {"success": False, "error": "Invalid URL or git not installed.", "files": []}

    temp_dir = tempfile.mkdtemp(prefix="repo_clone_")
    try:
        res = subprocess.run(["git", "clone", "--depth", "1", url, temp_dir], capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            return {"success": False, "error": "Repository inaccessible.", "files": []}

        collected = []
        for root, _, files in os.walk(temp_dir):
            if ".git" in root: continue
            for fname in files:
                if fname.endswith((".py", ".java")):
                    try:
                        with open(os.path.join(root, fname), "r", encoding="utf-8", errors="ignore") as f:
                            code = f.read()
                            detected = detect_language(code, fname)
                            if detected.language not in ("python", "java"):
                                continue
                            collected.append({
                                "file_name": os.path.relpath(os.path.join(root, fname), temp_dir),
                                "language": detected.language,
                                "language_confidence": detected.confidence,
                                "language_reason": detected.reason,
                                "code": code,
                            })
                    except Exception:
                        continue
                if len(collected) >= max_files: break
            if len(collected) >= max_files: break
            
        return {"success": True, "error": None, "files": collected} if collected else {"success": False, "error": "No supported files found.", "files": []}
    except Exception as e:
        return {"success": False, "error": f"Processing error: {e}", "files": []}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def download_file(url: str) -> dict:
    url = (url or "").strip()
    match = GITHUB_BLOB_PATTERN.match(url)
    if match:
        user, repo, branch, file_path = match.groups()
        url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"

    if not url.lower().endswith((".py", ".java")):
        return {"success": False, "error": "Unsupported file type.", "code": None, "file_name": None, "language": None}

    try:
        res = requests.get(url, timeout=15)
        if res.status_code != 200: raise requests.RequestException()
        code = res.text
        file_name = url.split("/")[-1] or "file"
        detected = detect_language(code, file_name)
        return {
            "success": True,
            "error": None,
            "code": code,
            "file_name": file_name,
            "language": detected.language,
            "language_confidence": detected.confidence,
            "language_reason": detected.reason,
        }
    except Exception as e:
        return {"success": False, "error": f"Download failed: {e}", "code": None, "file_name": None, "language": None}

# ── Pipeline Interface Core ─────────────────────────────────────

def _build_result(code: str, language: str = None, file_name: str = None) -> dict:
    code = code or ""
    detected = detect_language(code, file_name)
    lang = detected.language if detected.language in ("python", "java") else "unknown"

    validation = validate_code(code, lang)
    if not file_name:
        suffix = ".py" if lang == "python" else ".java" if lang == "java" else ".txt"
        file_name = f"submission{suffix}"

    return {
        "language": lang,
        "detected_language": lang,
        "language_confidence": detected.confidence,
        "language_reason": detected.reason,
        "selected_language": (language or "").strip().lower() or None,
        "file_name": file_name,
        "filepath": file_name,
        "lines": len(code.splitlines()) if code.strip() else 0, "characters": len(code),
        "valid": bool(validation.get("valid")), "error_type": validation.get("error_type"),
        "message": validation.get("message"), "error_line": validation.get("line"), "code": code,
    }

def process_pasted_code(code: str, language: str = None) -> dict:
    return _build_result(code, language, None)

def process_uploaded_file(code: str, language: str = None, file_name: str = None) -> dict:
    return _build_result(code, language, file_name)

def process_url_file(code: str, language: str = None, file_name: str = None) -> dict:
    return _build_result(code, language, file_name)
