"""
llm_router.py
-------------
Centralized LLM access point for the whole platform. This is the ONLY
place that knows there are two providers and how failover between them
works — agents (severityagent, remediationagent, and, where applicable,
prsummaryagent / conversationalagent) should never build their own
Gemini/Groq clients directly.

Design goals (per the LLM failover requirement):
    - The end user is NEVER asked for an API key. Keys come from the
      server/deployment environment only.
    - PRIMARY = Gemini (existing implementation already uses it).
    - BACKUP  = Groq. Chosen because:
        * it's already reachable through LangChain's standard chat-model
          interface (`langchain-groq`), so it plugs into the same
          `prompt | llm.with_structured_output(schema)` pattern the
          agents already use — no schema/prompt changes needed.
        * it has a genuinely free tier suitable as an emergency backup.
        * it adds exactly one new dependency, per "do not introduce
          unnecessary infrastructure."
    - Failover is automatic and silent to the end user: quota errors,
      auth errors, timeouts, and network errors on the primary all fall
      through to the backup transparently.
    - If both providers fail (or neither is configured), callers get a
      single `RouterError` and are expected to apply their OWN existing
      local/offline fallback (severityagent and remediationagent already
      have one — this file does not duplicate that logic).

Environment variables / Streamlit secrets expected:
    GEMINI_API_KEY       -> primary
    BACKUP_LLM_API_KEY   -> backup (Groq)
"""

import os
from typing import Optional, Type, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

PRIMARY_GEMINI_MODEL = "gemini-3.5-flash-lite"
BACKUP_GROQ_MODEL = "llama-3.3-70b-versatile"


class RouterError(RuntimeError):
    """Raised only when BOTH primary and backup have failed (or neither
    is configured). Callers should catch this exactly where they already
    catch the old direct-Gemini-chain exceptions and apply their existing
    offline fallback — no new except-branch shape is required."""


def _get_secret(name: str) -> Optional[str]:
    """Streamlit secrets first (deployment), then environment variables
    (local dev / non-Streamlit callers). Never raises if secrets aren't
    available — e.g. when this module is imported outside a Streamlit run."""
    try:
        import streamlit as st
        if name in st.secrets:
            value = st.secrets[name]
            if value and str(value).strip():
                return str(value).strip()
    except Exception:
        pass
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def get_primary_key() -> Optional[str]:
    return _get_secret("GEMINI_API_KEY")


def get_backup_key() -> Optional[str]:
    return _get_secret("BACKUP_LLM_API_KEY")


def primary_configured() -> bool:
    return bool(get_primary_key())


def backup_configured() -> bool:
    return bool(get_backup_key())


def any_configured() -> bool:
    """Used by callers to decide whether to attempt the LLM path at all
    (mirrors the old `bool(api_key)` check, but now covers either provider)."""
    return primary_configured() or backup_configured()


def _build_gemini_chain(schema: Type[T], prompt: ChatPromptTemplate, temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI
    key = get_primary_key()
    if not key:
        raise RouterError("Primary (Gemini) API key not configured on the server.")
    llm = ChatGoogleGenerativeAI(model=PRIMARY_GEMINI_MODEL, google_api_key=key, temperature=temperature)
    return prompt | llm.with_structured_output(schema)


def _build_groq_chain(schema: Type[T], prompt: ChatPromptTemplate, temperature: float):
    from langchain_groq import ChatGroq
    key = get_backup_key()
    if not key:
        raise RouterError("Backup (Groq) API key not configured on the server.")
    llm = ChatGroq(model=BACKUP_GROQ_MODEL, groq_api_key=key, temperature=temperature)
    return prompt | llm.with_structured_output(schema)


def run_structured(
    schema: Type[T],
    prompt: ChatPromptTemplate,
    input_dict: dict,
    temperature: float = 0.1,
) -> T:
    """
    generate_response()-style entry point used by every agent:
        try primary Gemini -> on failure -> try backup Groq -> return.

    Triggers covered by "on failure" (all handled the same way, since we
    can't rely on every provider raising the same exception subclasses):
    quota/rate-limit, exhausted key, auth failure, timeout, service
    unavailable, or any other network/API exception — plus the trivial
    case where a key simply isn't configured at all.

    Raises RouterError only if both attempts fail; existing agent code
    already wraps its call site in try/except and has its own local
    fallback for that case, so no new error-handling shape is required
    upstream.
    """
    primary_error: Optional[BaseException] = None

    if primary_configured():
        try:
            chain = _build_gemini_chain(schema, prompt, temperature)
            return chain.invoke(input_dict)
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
            primary_error = exc
    else:
        primary_error = RouterError("Primary (Gemini) API key not configured on the server.")

    if backup_configured():
        try:
            chain = _build_groq_chain(schema, prompt, temperature)
            return chain.invoke(input_dict)
        except Exception as backup_error:  # noqa: BLE001
            raise RouterError(
                f"Both LLM providers failed. Primary: {primary_error}. Backup: {backup_error}"
            ) from backup_error

    raise RouterError(f"Primary LLM failed and no backup is configured. Primary error: {primary_error}")