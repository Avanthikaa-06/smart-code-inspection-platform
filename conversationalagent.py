from __future__ import annotations

import difflib
import hashlib
import html
import io
import re
import time
import uuid
from datetime import datetime
from typing import Optional, Callable

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components

from rag_engine import embed_query, embed_texts

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable


# ============================================================================
# CONSTANTS
# ============================================================================
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

MAX_FINDINGS_IN_CONTEXT = 6
MAX_KB_CHUNKS_IN_CONTEXT = 4
MAX_CODE_CHARS_IN_CONTEXT = 6000
MAX_HISTORY_TURNS_IN_PROMPT = 3
CODE_VIEWER_CONTEXT_LINES = 5

NO_FINDING_MSG = "No matching finding was detected in the uploaded project."
NO_INFO_MSG = "This information is not available in the current scan."

_SEVERITY_EMOJI = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🔵", "Info": "⚪"}

_OWASP_2021_SLUGS = {
    "A01": "A01_2021-Broken_Access_Control",
    "A02": "A02_2021-Cryptographic_Failures",
    "A03": "A03_2021-Injection",
    "A04": "A04_2021-Insecure_Design",
    "A05": "A05_2021-Security_Misconfiguration",
    "A06": "A06_2021-Vulnerable_and_Outdated_Components",
    "A07": "A07_2021-Identification_and_Authentication_Failures",
    "A08": "A08_2021-Software_and_Data_Integrity_Failures",
    "A09": "A09_2021-Security_Logging_and_Monitoring_Failures",
    "A10": "A10_2021-Server-Side_Request_Forgery_%28SSRF%29",
}

SLASH_COMMANDS = {
    "/summary": "Summarize the current scan (findings, severities, score) — no AI call needed.",
    "/findings": "List every finding in the current scan.",
    "/high": "Show only High severity findings.",
    "/critical": "Show only Critical severity findings.",
    "/security": "Show only Security-agent findings.",
    "/remediation": "List findings that have a generated fix.",
    "/owasp": "Group current findings by OWASP category.",
    "/cwe<NN>": "Look up a specific CWE, e.g. /cwe89 for CWE-89.",
    "/code": "Preview the uploaded source code.",
    "/explain": "Explain the most recently discussed finding or remediation.",
    "/export": "Show how to export this conversation.",
    "/help": "List all available commands.",
}

_METADATA_REQUEST_TRIGGERS = (
    "what sources", "which sources", "show sources", "what source", "which source",
    "what cwe", "which cwe", "the cwe", "cwe is this", "cwe for this", "cwe number",
    "what owasp", "which owasp", "owasp category", "owasp is this",
    "related finding", "related findings", "show related", "show me related",
    "raw finding", "raw findings", "scan metadata", "current scan details",
)


# ============================================================================
# BOT CHARACTER — a rounded white robot head with a dark visor and glowing
# eyes (matching the reference mascot look), rendered as inline SVG so it's
# crisp and identical everywhere instead of depending on an emoji font.
# Two versions: a small head-only badge for chat avatars, and a bigger
# "hero" version with a body and a bouncing "Hi!" speech bubble for the
# empty state. There is also a small FIXED-POSITION badge (see
# _render_persistent_avatar) that reuses the same head SVG but stays
# pinned to the viewport instead of scrolling with the message list.
# ============================================================================
def _bot_face_svg(size: int) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <circle cx="6" cy="46" r="7" fill="#8B5FF5"/>
        <circle cx="94" cy="46" r="7" fill="#8B5FF5"/>
        <rect x="10" y="8" width="80" height="72" rx="36" fill="#f4f6fb" stroke="#d8dce8" stroke-width="1.5"/>
        <rect x="20" y="34" width="60" height="34" rx="17" fill="#12131f"/>
        <circle class="bot-eye" cx="38" cy="51" r="6.5" fill="#ffffff"/>
        <circle class="bot-eye" cx="62" cy="51" r="6.5" fill="#ffffff"/>
    </svg>"""


def _bot_hero_svg() -> str:
    return """<svg width="128" height="156" viewBox="0 0 120 150" xmlns="http://www.w3.org/2000/svg">
        <g class="bot-hi-bubble">
            <rect x="68" y="0" width="48" height="32" rx="10" fill="#ffffff"/>
            <path d="M78 30 L84 40 L91 30 Z" fill="#ffffff"/>
            <text x="92" y="21" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#6D5FF0">Hi!</text>
        </g>
        <circle cx="10" cy="54" r="7" fill="#8B5FF5"/>
        <circle cx="100" cy="54" r="7" fill="#8B5FF5"/>
        <rect x="14" y="16" width="82" height="72" rx="36" fill="#f4f6fb" stroke="#d8dce8" stroke-width="1.5"/>
        <rect x="26" y="42" width="60" height="34" rx="17" fill="#12131f"/>
        <circle class="bot-eye" cx="44" cy="59" r="6.5" fill="#ffffff"/>
        <circle class="bot-eye" cx="68" cy="59" r="6.5" fill="#ffffff"/>
        <rect x="24" y="94" width="64" height="48" rx="22" fill="#f4f6fb" stroke="#d8dce8" stroke-width="1.5"/>
        <circle cx="48" cy="118" r="3" fill="#8B5FF5"/>
        <circle cx="58" cy="118" r="3" fill="#6D5FF0"/>
        <circle cx="68" cy="118" r="3" fill="#ff6b8a"/>
    </svg>"""


_BOT_AVATAR_SVG = f'<div class="conv-avatar-bot">{_bot_face_svg(22)}</div>'
_BOT_HERO_SVG = f'<div class="conv-empty-bot">{_bot_hero_svg()}</div>'


def _render_persistent_avatar(thinking: bool) -> None:
    """
    Small fixed-position mascot badge (req. 1/7): stays visible the whole
    time the Conversational Assistant is open, independent of scroll
    position, unlike the per-message avatars which scroll with the chat.
    `thinking` toggles the glow/pulse state (req. 8/9) — pass
    st.session_state.conv_awaiting_answer so it pulses exactly while a
    reply is being generated and returns to idle the instant it lands.
    Purely decorative (pointer-events: none in CSS) so it can never
    intercept clicks, typing, or scrolling (req. 10).
    """
    cls = "conv-fixed-avatar thinking" if thinking else "conv-fixed-avatar"
    st.markdown(f'<div class="{cls}">{_bot_face_svg(24)}</div>', unsafe_allow_html=True)


# ============================================================================
# STYLES — Simplified centered chatbot (matches wireframe)
# ============================================================================
def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cyb-bg: rgba(13, 18, 32, 0.85);
            --cyb-border: rgba(129, 118, 245, 0.22);
            --cyb-border-strong: rgba(109, 95, 240, 0.65);
            --cyb-cyan: #8B5FF5;
            --cyb-violet: #6D5FF0;
            --cyb-text: #E6E6F5;
            --cyb-text-dim: #9494b8;
        }

        [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background: #0a0a18 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-app_shell"] {
            max-width: 900px;
            margin: 1.2rem auto;
            background: linear-gradient(180deg, #12122a 0%, #0d0d20 100%) !important;
            border: 1.5px solid var(--cyb-border-strong) !important;
            border-radius: 26px !important;
            padding: 0.6rem 1.2rem 1.2rem 1.2rem;
            box-shadow: 0 0 0 1px rgba(0,0,0,0.4), 0 20px 60px rgba(0,0,0,0.5);
        }

        .conv-header-title-row {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 0.65rem;
            padding: 0.5rem 0;
        }
        .conv-header-icon-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.1rem;
            height: 2.1rem;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--cyb-violet), var(--cyb-cyan));
            line-height: 0;
        }
        .conv-shield-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            line-height: 0;
        }
        .conv-header-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.01em;
        }
        div[data-testid="stDownloadButton"] button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            border: 1px solid #3a3a6a !important;
            background: #1a1a35 !important;
            color: #d8d8f0 !important;
            padding: 0.5rem 1.1rem !important;
        }
        div[data-testid="stDownloadButton"] button:hover {
            border-color: var(--cyb-cyan) !important;
            background: #22224a !important;
            color: #ffffff !important;
        }
        div[data-testid="stDownloadButton"] {
            display: flex;
            justify-content: flex-end;
        }

        .conv-empty-state {
            text-align: center;
            color: var(--cyb-text-dim);
            padding: 3.2rem 1rem;
        }
        .conv-scroll-area {
            max-height: 60vh;
            overflow-y: auto;
            padding: 0.4rem 0.2rem;
        }

        .conv-row { display: flex; margin-bottom: 0.9rem; }
        .conv-row.user { justify-content: flex-end; }
        .conv-row.assistant { align-items: flex-start; gap: 0.6rem; }

        .conv-avatar-bot {
            flex: none;
            width: 2.3rem;
            height: 2.3rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #eef1fb, #dde3fa);
            box-shadow: 0 2px 10px rgba(109,95,240,0.3);
            margin-top: 2px;
            overflow: hidden;
        }
        .conv-avatar-bot svg { display: block; }

        .conv-empty-bot {
            width: 128px;
            margin: 0 auto 0.6rem auto;
            display: flex;
            align-items: center;
            justify-content: center;
            filter: drop-shadow(0 14px 22px rgba(109,95,240,0.35));
            animation: botFloat 3s ease-in-out infinite;
        }
        @keyframes botFloat {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-8px); }
        }

        .bot-eye {
            transform-box: fill-box;
            transform-origin: center;
            animation: botBlink 4.5s infinite;
        }
        .bot-eye:nth-of-type(2) { animation-delay: 0.12s; }
        @keyframes botBlink {
            0%, 90%, 100% { transform: scaleY(1); }
            95% { transform: scaleY(0.15); }
        }
        .bot-hi-bubble {
            transform-box: fill-box;
            transform-origin: center;
            animation: botHiPop 2.4s ease-in-out infinite;
        }
        @keyframes botHiPop {
            0%, 100% { transform: scale(1) translateY(0); }
            50% { transform: scale(1.06) translateY(-3px); }
        }

        .conv-fixed-avatar {
            position: fixed;
            right: 22px;
            bottom: 92px;
            width: 46px;
            height: 46px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #eef1fb, #dde3fa);
            box-shadow: 0 4px 18px rgba(109,95,240,0.35);
            z-index: 999;
            pointer-events: none;
            animation: botFloat 3s ease-in-out infinite;
        }
        .conv-fixed-avatar svg { display: block; }
        .conv-fixed-avatar.thinking {
            animation: botThinkPulse 1.1s ease-in-out infinite;
        }
        @keyframes botThinkPulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(139,95,245,0.55), 0 4px 18px rgba(109,95,240,0.35); }
            50%      { box-shadow: 0 0 0 8px rgba(139,95,245,0), 0 4px 18px rgba(109,95,240,0.45); }
        }
        @media (max-width: 640px) {
            .conv-fixed-avatar {
                width: 38px;
                height: 38px;
                right: 14px;
                bottom: 78px;
            }
        }

        .conv-bubble-user {
            background: linear-gradient(135deg, #6d5ff0, #5245c9);
            color: #ffffff;
            border-radius: 16px 16px 4px 16px;
            padding: 0.75rem 1.15rem;
            max-width: 78%;
            font-size: 0.95rem;
            line-height: 1.55;
            margin-left: auto;
            box-shadow: 0 8px 24px rgba(82, 69, 201, 0.25);
        }

        div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-ai_card_"] {
            background: #14142b !important;
            border: 1px solid #2a2a52 !important;
            border-radius: 4px 18px 18px 18px !important;
            max-width: 100%;
            margin-bottom: 0.2rem !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        }
        div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-ai_card_"] p,
        div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-ai_card_"] li {
            color: #e6e6f5 !important;
            font-size: 0.93rem;
            line-height: 1.6;
        }
        div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-ai_card_"] strong {
            color: #ff6b8a !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-ai_card_"] code {
            background: rgba(139,95,245,0.15) !important;
            color: #c4b5fd !important;
            border-radius: 5px;
            padding: 0.1rem 0.4rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-ai_card_"] pre {
            background: rgba(6,10,18,0.85) !important;
            border: 1px solid #2a2a52 !important;
            border-radius: 10px !important;
        }

        .conv-thinking-wrap { padding: 0.15rem 0; }
        .conv-ts-row {
            display: flex;
            align-items: center;
            gap: 0.3rem;
            margin-top: 0.3rem;
        }
        .conv-row.user .conv-ts-row { justify-content: flex-end; }
        .conv-ts { color: #64748B; font-size: 0.68rem; }
        .conv-check {
            color: #38BDF8;
            font-size: 0.75rem;
            line-height: 0;
        }

        .conv-thinking-dot {
            display: inline-block;
            color: #6d5ff0;
            font-size: 0.6rem;
            margin-right: 0.2rem;
            animation: thinkPulse 1.2s ease-in-out infinite;
        }
        .conv-thinking-dot:nth-child(2) { animation-delay: 0.2s; }
        .conv-thinking-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes thinkPulse {
            0%, 100% { opacity: 0.25; transform: translateY(0); }
            50% { opacity: 1; transform: translateY(-3px); }
        }

        .conv-source-chip {
            display: inline-block;
            padding: 0.12rem 0.6rem;
            border-radius: 999px;
            font-size: 0.65rem;
            font-weight: 600;
            margin-right: 0.3rem;
            margin-top: 0.4rem;
            background: rgba(139,95,245,0.15);
            color: #c4b5fd;
            border: 1px solid #2a2a52;
        }

        div[data-testid="stForm"] {
            border: 1.5px solid #2a2a52 !important;
            border-radius: 18px !important;
            background: #12122a !important;
            padding: 0.5rem 0.6rem !important;
            margin-top: 0.8rem;
        }
        div[data-testid="stForm"] div[data-testid="stTextInput"] input {
            background-color: transparent !important;
            border: none !important;
            color: #e6e6f5 !important;
            padding: 0.9rem 1.2rem !important;
            font-size: 0.95rem !important;
            box-shadow: none !important;
        }
        div[data-testid="stForm"] div[data-testid="stTextInput"] input::placeholder {
            color: #5c5c80 !important;
        }
        div[data-testid="stForm"] button[kind="primaryFormSubmit"],
        div[data-testid="stForm"] button[kind="primary"] {
            border-radius: 12px !important;
            background: linear-gradient(135deg, #6d5ff0 0%, #5245c9 100%) !important;
            border: none !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            height: 100% !important;
        }
        div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover,
        div[data-testid="stForm"] button[kind="primary"]:hover {
            opacity: 0.9;
        }

        div[data-testid="stForm"].chat-input-pulse {
            animation: chatInputPulse 1.6s ease-out 1;
        }
        @keyframes chatInputPulse {
            0%   { box-shadow: 0 0 0 0 rgba(139,95,245,0.65); border-color: var(--cyb-cyan) !important; }
            70%  { box-shadow: 0 0 0 16px rgba(139,95,245,0); border-color: var(--cyb-cyan) !important; }
            100% { box-shadow: 0 0 0 0 rgba(139,95,245,0); border-color: #2a2a52 !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe(value) -> str:
    return html.escape(str(value if value is not None else ""))


# ============================================================================
# SESSION STATE — single continuous conversation (no sidebar / multi-chat)
# by default. When the caller passes a `messages_key` (see
# render_conversational_assistant's new `state_key` argument), the message
# list lives under that key instead of the shared "conv_messages" key, so a
# specific analysis (e.g. a History record) gets its own isolated,
# already-persisted conversation instead of sharing the one global list.
# Every other helper below reads/writes messages through _get_msgs(), which
# resolves to whichever key was set here, so no second/duplicate list is
# ever created for the same conversation.
# ============================================================================
def _init_state(messages_key: str = "conv_messages") -> None:
    st.session_state.setdefault(messages_key, [])
    st.session_state["_conv_messages_key"] = messages_key
    st.session_state.setdefault("conv_last_focus", None)
    st.session_state.setdefault("conv_pending_question", None)
    st.session_state.setdefault("conv_finding_index", None)
    st.session_state.setdefault("conv_finding_index_key", None)
    st.session_state.setdefault("conv_current_code", "")
    st.session_state.setdefault("conv_current_language", "")
    st.session_state.setdefault("conv_expanded_finding", None)
    st.session_state.setdefault("conv_input_nonce", 0)
    st.session_state.setdefault("conv_awaiting_answer", False)
    st.session_state.setdefault("_pdf_cache_key", None)
    st.session_state.setdefault("_pdf_bytes", b"")


def _messages_key() -> str:
    return st.session_state.get("_conv_messages_key", "conv_messages")


def _get_msgs() -> list:
    """The single source of truth for this render's conversation list —
    every read/append goes through here so the active analysis's history
    is only ever stored once (no parallel/duplicate list)."""
    return st.session_state[_messages_key()]


def _scroll_to_chat_input() -> None:
    """
    Injects a tiny script that smooth-scrolls the chat's input bar into
    view, pulses its border, and then focuses the text field. Called once
    per render_conversational_assistant(..., focus_input=True) call — the
    caller (app.py) is responsible for making sure that flag is True on
    only one render per button click (e.g. via a one-shot
    st.session_state.pop(...) flag before calling this function), so this
    helper itself does no session-state bookkeeping of its own.

    Retries for a short window instead of checking once: right after a
    sub-tab switch a lot of the DOM has just been replaced, and this
    injected <script> (running inside its own components.html iframe) can
    easily execute a tick before Streamlit's own re-render has actually
    mounted the chat input further down the page. A single immediate
    querySelector then finds nothing and silently no-ops — which is what
    made this look broken. Mirrors app.py's _scroll_to_anchor helper,
    which uses the same retry pattern for the same reason.
    """
    components.html(
        """
        <script>
        (function() {
            function run() {
                const doc = window.parent.document;
                const wrapper = doc.querySelector('div[class*="st-key-chat_input_container"]');
                if (!wrapper) return false;

                wrapper.scrollIntoView({behavior: 'smooth', block: 'center'});

                const formEl = wrapper.querySelector('div[data-testid="stForm"]');
                if (formEl) {
                    formEl.classList.remove('chat-input-pulse');
                    void formEl.offsetWidth; // restart animation if triggered again
                    formEl.classList.add('chat-input-pulse');
                    setTimeout(() => formEl.classList.remove('chat-input-pulse'), 1700);
                }

                const inputEl = wrapper.querySelector('input[type="text"]');
                if (inputEl) {
                    setTimeout(() => inputEl.focus(), 450);
                }
                return true;
            }

            if (!run()) {
                let attempts = 0;
                const interval = setInterval(function() {
                    attempts += 1;
                    if (run() || attempts > 20) {
                        clearInterval(interval);
                    }
                }, 100);
            }
        })();
        </script>
        """,
        height=0,
    )


def _active_session() -> dict:
    # Proxy dict so retrieval helpers below (unchanged) can keep reading
    # ["messages"] / ["last_focus"] without knowing about the flat
    # session_state keys that back a single continuous conversation.
    return {
        "messages": _get_msgs(),
        "last_focus": st.session_state.conv_last_focus,
    }


def _set_last_focus(focus) -> None:
    st.session_state.conv_last_focus = focus


# ============================================================================
# RETRIEVAL + LOGIC (unchanged)
# ============================================================================
def _findings_fingerprint(findings: list) -> str:
    ids = [str(f.get("id") or f.get("title") or "") + str(f.get("line") or "") for f in findings]
    return hashlib.md5("|".join(ids).encode("utf-8", errors="ignore")).hexdigest()


def _finding_to_blob(f: dict) -> str:
    parts = [
        f.get("title") or f.get("category") or "", f.get("description") or f.get("message") or "",
        f.get("recommendation") or "", f.get("cwe") or f.get("cwe_id") or "",
        f.get("owasp") or f.get("owasp_category") or "", f.get("tool") or "", f.get("severity") or "",
    ]
    remediation = f.get("remediation")
    if remediation:
        parts += [remediation.get("root_cause") or "", remediation.get("explanation") or "",
                  remediation.get("guideline_reference") or "", remediation.get("prevention_tip") or ""]
    return " | ".join(p for p in parts if p)


def _get_finding_index(findings: list):
    if not findings:
        return None
    key = _findings_fingerprint(findings)
    if st.session_state.get("conv_finding_index_key") == key and st.session_state.get("conv_finding_index"):
        return st.session_state["conv_finding_index"]
    blobs = [_finding_to_blob(f) for f in findings]
    try:
        embeddings = embed_texts(blobs)
        embeddings = np.asarray(embeddings, dtype="float32") if len(embeddings) else None
    except Exception:
        embeddings = None
    index = {"embeddings": embeddings, "findings": findings}
    st.session_state["conv_finding_index"] = index
    st.session_state["conv_finding_index_key"] = key
    return index


_KEYWORD_HINTS = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}


def _keyword_score(question: str, f: dict) -> float:
    q = question.lower()
    score = 0.0
    for field_name in ("cwe", "cwe_id", "owasp", "owasp_category", "tool", "file", "title", "category"):
        val = str(f.get(field_name) or "").lower()
        if val and val in q:
            score += 2.0
    for word, sev in _KEYWORD_HINTS.items():
        if word in q and f.get("severity") == sev:
            score += 1.0
    return score


def _augment_query_for_followup(question: str) -> str:
    focus = _active_session().get("last_focus")
    if not focus:
        return question
    pronoun_pattern = r"\b(it|this|that|the fix|the issue|the vulnerability)\b"
    has_pronoun = re.search(pronoun_pattern, question, re.IGNORECASE)
    has_specific_term = any(str(focus.get(k) or "").lower() in question.lower() for k in ("title", "cwe", "cwe_id", "owasp", "owasp_category"))
    if has_pronoun and not has_specific_term:
        extra = " ".join(str(focus.get(k) or "") for k in ("title", "cwe", "cwe_id", "owasp", "owasp_category"))
        return f"{question} ({extra})"
    return question


def _literal_term_match(question: str, findings: list) -> list:
    q = (question or "").lower()
    candidates = []
    for f in findings:
        title = str(f.get("title") or f.get("category") or "").lower().strip()
        blob = _finding_to_blob(f).lower()
        if title and title in q:
            candidates.append((3.0, f))
            continue
        title_words = [w for w in re.findall(r"[a-z0-9]+", title) if len(w) >= 3]
        if title_words and all(w in q for w in title_words):
            candidates.append((2.0, f))
            continue
        q_words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) >= 4]
        overlap = sum(1 for w in q_words if w in blob)
        if overlap >= 2:
            candidates.append((float(overlap), f))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [f for _, f in candidates]


def retrieve_findings(question: str, findings: list, top_k: int = MAX_FINDINGS_IN_CONTEXT) -> list:
    if not findings:
        return []
    search_text = _augment_query_for_followup(question)
    index = _get_finding_index(findings)
    scored = []
    if index and index.get("embeddings") is not None:
        try:
            q_emb = embed_query(search_text)
            if q_emb is not None:
                sims = index["embeddings"] @ np.asarray(q_emb, dtype="float32")
                norms = np.linalg.norm(index["embeddings"], axis=1) * (np.linalg.norm(q_emb) or 1e-9)
                norms[norms == 0] = 1e-9
                sims = sims / norms
                for i, f in enumerate(findings):
                    scored.append((float(sims[i]) + _keyword_score(search_text, f), f))
        except Exception:
            scored = []
    if not scored:
        for f in findings:
            scored.append((_keyword_score(search_text, f) + 0.01, f))
    named_sev = next((sev for word, sev in _KEYWORD_HINTS.items() if word in question.lower()), None)
    if named_sev:
        filtered = [(s, f) for s, f in scored if f.get("severity") == named_sev]
        scored = filtered or scored
    scored.sort(key=lambda pair: pair[0], reverse=True)
    result = [f for s, f in scored[:top_k] if s > 0]
    if not result:
        result = _literal_term_match(search_text, findings)[:top_k]
    return result


def retrieve_kb(question: str) -> list:
    pipeline = st.session_state.get("rag_pipeline")
    if pipeline is None:
        return []
    if not pipeline.store.is_ready():
        try:
            pipeline.build_index(use_cache=True)
        except Exception:
            return []
    if not pipeline.store.is_ready():
        return []
    try:
        result = pipeline.query(question, top_k=MAX_KB_CHUNKS_IN_CONTEXT)
        return result.get("results", []) if result.get("success") else []
    except Exception:
        return []


def retrieve_code_context(question: str, uploaded_code: str) -> Optional[str]:
    if not uploaded_code:
        return None
    lines = uploaded_code.splitlines()
    line_match = re.search(r"\bline\s*[:#]?\s*(\d+)\b", question, re.IGNORECASE)
    if line_match:
        n = int(line_match.group(1))
        start, end = max(0, n - 9), min(len(lines), n + 8)
        snippet = "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, end))
        return f"[Lines {start + 1}-{end} of uploaded file]\n{snippet}"
    broad_terms = ("summarize", "summary", "explain my code", "what does this", "overview", "authentication logic", "find auth", "explain this function", "explain this class")
    if any(term in question.lower() for term in broad_terms):
        joined = uploaded_code[:MAX_CODE_CHARS_IN_CONTEXT]
        truncated = len(uploaded_code) > MAX_CODE_CHARS_IN_CONTEXT
        return joined + ("\n... [truncated]" if truncated else "")
    return None


def retrieve_pr_context(question: str, pr_summary: Optional[dict]) -> Optional[str]:
    if not pr_summary:
        return None
    triggers = ("merge", "mergeability", "review time", "which issue should i fix", "prioritiz", "overview", "code health", "summarize my scan", "how many")
    if not any(t in question.lower() for t in triggers):
        return None
    merge = pr_summary.get("merge_recommendation", {})
    lines = [
        f"Merge recommendation: {merge.get('label')} — {merge.get('description')}",
        f"Mergeability score: {pr_summary.get('mergeability_score')}",
        f"Total findings: {pr_summary.get('total_findings')}",
        f"Auto-fixable: {pr_summary.get('auto_fixable', {}).get('ratio_pct')}%",
        f"Executive overview: {pr_summary.get('executive_overview')}",
    ]
    top_fixes = pr_summary.get("prioritized_fixes", [])[:5]
    if top_fixes:
        lines.append("Top prioritized fixes:")
        for item in top_fixes:
            lines.append(f"  #{item['rank']} [{item['severity']}] {item['title']} ({item['file']}:{item['line']})")
    return "\n".join(lines)


def _detect_mode(question: str) -> str:
    q = question.lower()
    if re.search(r"explain\s+(cwe-?\d+|owasp\s*a?\d*)", q):
        return "vuln_learning"
    if "remediation" in q and ("explain" in q or "why" in q):
        return "remediation_explain"
    if re.search(r"explain\s+(this|my)\s+(function|code|class)", q) or "summarize my code" in q:
        return "code_explanation"
    if "generate" in q and any(t in q for t in ("secure version", "unit test", "java example", "python example", "comment", "fixed code")):
        return "code_generation"
    if "chart" in q and ("explain" in q or "read" in q):
        return "chart_explain"
    return "general"


_MODE_TEMPLATES = {
    "vuln_learning": "Structure your answer with EXACTLY these markdown headers: **Definition**, **Example**, **Attack Scenario**, **Impact**, **Real-World Example**, **Prevention**, **Secure Example**, **References**.",
    "code_explanation": "Structure your answer with EXACTLY these markdown headers: **Purpose**, **Flow**, **Complexity**, **Possible Issues**, **Best Practice**.",
    "remediation_explain": "Structure your answer with EXACTLY these markdown headers: **Root Cause**, **Why Current Code Is Unsafe**, **Before**, **After**, **Why Fix Works**, **Risk Reduced**, **References**.",
    "code_generation": "Respond with a short explanation followed by a single fenced code block.",
}


def _make_result(answer: str, findings_ctx=None, kb_ctx=None, llm_ok=False, mode="deterministic", sources=None, confidence=100) -> dict:
    return {
        "answer": answer, "findings": findings_ctx or [], "kb_hits": kb_ctx or [],
        "llm_ok": llm_ok, "mode": mode, "sources": sources if sources is not None else [],
        "confidence": confidence, "timestamp": datetime.now().strftime("%H:%M:%S"),
        "timestamp_iso": datetime.now().isoformat(),
    }


def _cmd_summary(findings: list, deep_result: dict) -> dict:
    summary = (deep_result or {}).get("summary") or {}
    by_sev = summary.get("by_severity", {})
    score = summary.get("security_score", "N/A")
    lines = [f"**Scan Summary** — {summary.get('total', len(findings))} total finding(s), security score **{score}/100**.", ""]
    for sev in ("Critical", "High", "Medium", "Low", "Info"):
        if by_sev.get(sev):
            lines.append(f"- {_SEVERITY_EMOJI[sev]} {sev}: {by_sev[sev]}")
    return _make_result("\n".join(lines), findings_ctx=findings[:MAX_FINDINGS_IN_CONTEXT])


def _cmd_list_findings(findings: list, severity: Optional[str] = None, agent: Optional[str] = None) -> dict:
    subset = findings
    if severity:
        subset = [f for f in subset if f.get("severity") == severity]
    if agent:
        subset = [f for f in subset if f.get("agent") == agent]
    if not subset:
        return _make_result(NO_FINDING_MSG)
    lines = [f"**{len(subset)} finding(s)** matched:", ""]
    for f in subset[:20]:
        lines.append(f"- {_SEVERITY_EMOJI.get(f.get('severity'), '⚪')} **{f.get('title')}** — {f.get('file') or 'uploaded file'}:{f.get('line') or f.get('line_start') or 'N/A'}")
    return _make_result("\n".join(lines), findings_ctx=subset[:MAX_FINDINGS_IN_CONTEXT])


def _cmd_remediation(findings: list) -> dict:
    subset = [f for f in findings if f.get("remediation")]
    if not subset:
        return _make_result("No remediations have been generated yet for this scan.")
    lines = [f"**{len(subset)} finding(s)** have a generated fix:", ""]
    for f in subset[:20]:
        conf = (f.get("remediation") or {}).get("application_confidence", "unknown")
        lines.append(f"- **{f.get('title')}** — confidence: `{conf}`")
    return _make_result("\n".join(lines), findings_ctx=subset[:MAX_FINDINGS_IN_CONTEXT])


def _cmd_owasp(findings: list) -> dict:
    groups: dict = {}
    for f in findings:
        cat = f.get("owasp") or f.get("owasp_category")
        if cat:
            groups.setdefault(cat, []).append(f)
    if not groups:
        return _make_result(NO_FINDING_MSG)
    lines = ["**Findings grouped by OWASP category:**", ""]
    for cat, items in groups.items():
        lines.append(f"- **{cat}** — {len(items)} finding(s)")
    flat = [f for items in groups.values() for f in items]
    return _make_result("\n".join(lines), findings_ctx=flat[:MAX_FINDINGS_IN_CONTEXT])


def _cmd_cwe(cwe_number: str, findings: list) -> dict:
    matches = [f for f in findings if cwe_number in str(f.get("cwe") or f.get("cwe_id") or "")]
    kb_hits = retrieve_kb(f"CWE-{cwe_number}")
    lines = [f"**CWE-{cwe_number}**"]
    if matches:
        lines.append(f"\nFound in {len(matches)} current finding(s):")
        for f in matches:
            lines.append(f"- {f.get('title')} — {f.get('file')}:{f.get('line') or f.get('line_start')}")
    else:
        lines.append(f"\n{NO_FINDING_MSG}")
    if kb_hits:
        lines.append("\nKnowledge base excerpt:")
        lines.append(f"> {kb_hits[0].get('text', '')[:400]}")
    return _make_result("\n".join(lines), findings_ctx=matches, kb_ctx=kb_hits)


def _cmd_code_preview(uploaded_code: str, language: str) -> dict:
    if not uploaded_code:
        return _make_result(NO_INFO_MSG)
    preview = uploaded_code[:1500]
    truncated = len(uploaded_code) > 1500
    return _make_result(f"```{language or ''}\n{preview}{'...' if truncated else ''}\n```")


def _cmd_help() -> dict:
    lines = ["**Available commands:**", ""]
    for cmd, desc in SLASH_COMMANDS.items():
        lines.append(f"- `{cmd}` — {desc}")
    return _make_result("\n".join(lines))


def _handle_slash_command(question: str, findings: list, deep_result: dict, uploaded_code: str, language: str) -> Optional[dict]:
    if not question.startswith("/"):
        return None
    parts = question.split(maxsplit=1)
    cmd = parts[0].lower()
    if cmd == "/summary": return _cmd_summary(findings, deep_result)
    if cmd == "/findings": return _cmd_list_findings(findings)
    if cmd == "/high": return _cmd_list_findings(findings, severity="High")
    if cmd == "/critical": return _cmd_list_findings(findings, severity="Critical")
    if cmd == "/security": return _cmd_list_findings(findings, agent="security")
    if cmd == "/remediation": return _cmd_remediation(findings)
    if cmd == "/owasp": return _cmd_owasp(findings)
    if cmd.startswith("/cwe"):
        num = re.sub(r"\D", "", cmd) or (re.sub(r"\D", "", parts[1]) if len(parts) > 1 else "")
        if num: return _cmd_cwe(num, findings)
    if cmd == "/code": return _cmd_code_preview(uploaded_code, language)
    if cmd == "/export": return _make_result("Use the export button to download this conversation.")
    if cmd == "/help": return _cmd_help()
    if cmd == "/explain":
        focus = _active_session().get("last_focus")
        if focus: return None
        return _make_result("Ask about a specific finding first, then `/explain` will elaborate on it.")
    return _make_result(f"Unknown command `{cmd}`. Type `/help` to see available commands.")


def _maybe_instant_search(question: str, findings: list) -> Optional[dict]:
    q = question.strip()
    if not q or "?" in q or len(q.split()) > 3 or q.startswith("/"): return None
    if any(q.lower().startswith(w) for w in ("explain", "show", "what", "why", "how", "which", "summarize")): return None
    ql = q.lower()
    matches = [f for f in findings if ql in (str(f.get("title", "")) + str(f.get("description", "")) + str(f.get("message", ""))).lower()]
    if not matches: return None
    lines = [f"**{len(matches)} finding(s) matched** '{q}':", ""]
    for f in matches[:15]:
        lines.append(f"- {_SEVERITY_EMOJI.get(f.get('severity'), '⚪')} **{f.get('title')}** — {f.get('file') or 'uploaded file'}:{f.get('line') or f.get('line_start') or 'N/A'}")
    return _make_result("\n".join(lines), findings_ctx=matches[:MAX_FINDINGS_IN_CONTEXT])


def _explain_chart(deep_result: dict) -> dict:
    summary = (deep_result or {}).get("summary") or {}
    by_sev = summary.get("by_severity", {})
    score = summary.get("security_score", "N/A")
    lines = ["**Severity Breakdown**", ""]
    for sev in ("Critical", "High", "Medium", "Low", "Info"):
        if by_sev.get(sev):
            lines.append(f"- {_SEVERITY_EMOJI[sev]} {sev}: {by_sev[sev]}")
    lines.append(f"\n**Overall Risk Score:** {score}/100")
    return _make_result("\n".join(lines))


_SYSTEM_INSTRUCTIONS = """You are the AI Secure Code Assistant. Answer developer questions about their uploaded code, findings, remediation, and secure coding knowledge (OWASP / CWE).

STRICT RULES:
1. Never invent findings, line numbers, CWE/OWASP IDs, or facts not in the CONTEXT.
2. If no matching finding exists, say exactly: "No matching finding was detected in the uploaded project."
3. Be concise, technically precise, and developer-friendly.
4. Use conversation history only for pronouns/follow-ups.
"""


def _build_history_block(messages: list) -> str:
    recent = [m for m in messages if m["role"] in ("user", "assistant")][-2 * MAX_HISTORY_TURNS_IN_PROMPT:]
    if not recent: return ""
    lines = ["CONVERSATION HISTORY:"]
    for m in recent:
        speaker = "Developer" if m["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {m['content'][:500]}")
    return "\n".join(lines)


def _build_prompt(question: str, findings_ctx: list, kb_ctx: list, code_ctx: Optional[str],
                  pr_ctx: Optional[str], history_block: str, mode: str) -> str:
    context_blocks = []
    if findings_ctx:
        block = ["CURRENT SCAN FINDINGS:"]
        for f in findings_ctx:
            block.append(
                f"- [{f.get('severity')}] {f.get('title') or f.get('category')} "
                f"(file: {f.get('file') or 'uploaded file'}, line: {f.get('line') or f.get('line_start') or 'N/A'}, "
                f"tool: {f.get('tool')}, CWE: {f.get('cwe') or f.get('cwe_id') or 'N/A'}, "
                f"OWASP: {f.get('owasp') or f.get('owasp_category') or 'N/A'})\n"
                f"  Description: {f.get('description') or f.get('message') or 'N/A'}\n"
                f"  Recommendation: {f.get('recommendation') or 'N/A'}"
            )
            remediation = f.get("remediation")
            if remediation:
                block.append(f"  Remediation: {remediation.get('root_cause') or 'N/A'} | Before: {remediation.get('before_snippet') or 'N/A'} | After: {remediation.get('after_snippet') or 'N/A'}")
        context_blocks.append("\n".join(block))
    if kb_ctx:
        block = ["KNOWLEDGE BASE EXCERPTS:"]
        for chunk in kb_ctx:
            block.append(f"- (Source: {chunk.get('source')}) {chunk.get('text')}")
        context_blocks.append("\n".join(block))
    if code_ctx: context_blocks.append(f"UPLOADED CODE EXCERPT:\n{code_ctx}")
    if pr_ctx: context_blocks.append(f"PR / SCAN SUMMARY:\n{pr_ctx}")
    if history_block: context_blocks.append(history_block)
    context_str = "\n\n".join(context_blocks) if context_blocks else "(no relevant context)"
    mode_instruction = _MODE_TEMPLATES.get(mode, "")
    return f"{_SYSTEM_INSTRUCTIONS}\n\n{('FORMAT: ' + mode_instruction) if mode_instruction else ''}\n\nCONTEXT:\n{context_str}\n\nDEVELOPER QUESTION:\n{question}\n\nAnswer using only the context above."


def _call_gemini(prompt: str, api_key: str, timeout: int = 30) -> dict:
    if not api_key:
        return {"success": False, "text": None, "error": "No Gemini API key provided."}
    try:
        response = requests.post(GEMINI_ENDPOINT, params={"key": api_key},
                                  json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=timeout)
    except requests.RequestException as exc:
        return {"success": False, "text": None, "error": f"Could not reach Gemini API: {exc}"}
    if response.status_code != 200:
        return {"success": False, "text": None, "error": f"Gemini API error (HTTP {response.status_code})."}
    try:
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"success": True, "text": text, "error": None}
    except (KeyError, IndexError, ValueError):
        return {"success": False, "text": None, "error": "Unexpected response from Gemini API."}


def _compute_confidence(findings_ctx: list, kb_ctx: list, llm_ok: bool) -> int:
    base = 55 if llm_ok else 20
    base += min(len(findings_ctx) * 8, 30)
    base += min(len(kb_ctx) * 5, 15)
    return int(min(base, 97))


def _sources_used(findings_ctx: list, kb_ctx: list) -> list:
    sources = []
    if findings_ctx: sources.append("Current Scan")
    seen = set()
    for f in findings_ctx:
        owasp = f.get("owasp") or f.get("owasp_category")
        cwe = f.get("cwe") or f.get("cwe_id")
        if owasp and owasp not in seen:
            sources.append(f"OWASP {owasp}")
            seen.add(owasp)
        if cwe and cwe not in seen:
            sources.append(cwe if str(cwe).upper().startswith("CWE") else f"CWE {cwe}")
            seen.add(cwe)
    for c in kb_ctx:
        src = c.get("source")
        if src and src not in sources: sources.append(src)
    return sources


def _cwe_link(cwe_value: str) -> Optional[str]:
    digits = re.sub(r"\D", "", str(cwe_value or ""))
    return f"https://cwe.mitre.org/data/definitions/{digits}.html" if digits else None


def _owasp_link(owasp_value: str) -> Optional[str]:
    match = re.search(r"A0?(\d{1,2})", str(owasp_value or ""), re.IGNORECASE)
    if not match: return "https://owasp.org/Top10/"
    key = f"A{int(match.group(1)):02d}"
    slug = _OWASP_2021_SLUGS.get(key)
    return f"https://owasp.org/Top10/{slug}/" if slug else "https://owasp.org/Top10/"


def _answer_for(question: str, findings: list, pr_summary: Optional[dict], uploaded_code: str,
                language: str, deep_result: dict, api_key: str) -> dict:
    slash_result = _handle_slash_command(question, findings, deep_result, uploaded_code, language)
    if slash_result: return slash_result
    if _detect_mode(question) == "chart_explain": return _explain_chart(deep_result)
    instant = _maybe_instant_search(question, findings)
    if instant: return instant

    findings_ctx = retrieve_findings(question, findings)
    kb_ctx = retrieve_kb(question)
    code_ctx = retrieve_code_context(question, uploaded_code)
    pr_ctx = retrieve_pr_context(question, pr_summary)
    history_block = _build_history_block(_active_session()["messages"])
    mode = _detect_mode(question)

    if not (api_key or "").strip():
        fallback = ("I need a valid Gemini API key to generate free-form answers. "
                    "Here's what matched in the current scan:\n\n" +
                    ("\n".join(f"- [{f.get('severity')}] {f.get('title')}" for f in findings_ctx) if findings_ctx else NO_FINDING_MSG))
        return {
            "answer": fallback, "findings": findings_ctx, "kb_hits": kb_ctx, "llm_ok": False, "mode": mode,
            "sources": _sources_used(findings_ctx, kb_ctx), "confidence": _compute_confidence(findings_ctx, kb_ctx, False),
            "timestamp": datetime.now().strftime("%H:%M:%S"), "timestamp_iso": datetime.now().isoformat(),
        }

    prompt = _build_prompt(question, findings_ctx, kb_ctx, code_ctx, pr_ctx, history_block, mode)
    result = _call_gemini(prompt, api_key)
    llm_ok = result["success"]
    answer_text = result["text"] if llm_ok else f"I couldn't generate an answer right now ({result['error']})."

    return {
        "answer": answer_text, "findings": findings_ctx, "kb_hits": kb_ctx, "llm_ok": llm_ok, "mode": mode,
        "sources": _sources_used(findings_ctx, kb_ctx), "confidence": _compute_confidence(findings_ctx, kb_ctx, llm_ok),
        "timestamp": datetime.now().strftime("%H:%M:%S"), "timestamp_iso": datetime.now().isoformat(),
    }


def _wants_metadata(question: str) -> bool:
    q = (question or "").lower().strip()
    if q.startswith("/"): return True
    return any(trigger in q for trigger in _METADATA_REQUEST_TRIGGERS)


# ============================================================================
# HELPERS
# ============================================================================
def _copy_block(label: str, content: str, language: str = "text") -> None:
    if not content: return
    popover_fn = getattr(st, "popover", None)
    if popover_fn:
        with popover_fn(label):
            st.code(content, language=language)
    else:
        with st.expander(label):
            st.code(content, language=language)


def _unified_diff_html(before: str, after: str) -> Optional[str]:
    before_lines = (before or "").splitlines()
    after_lines = (after or "").splitlines()
    if not before_lines and not after_lines: return None
    diff = list(difflib.unified_diff(before_lines, after_lines, lineterm="", n=2))
    rows = []
    for line in diff:
        if line.startswith(("+++", "---")): continue
        escaped = html.escape(line)
        if line.startswith("@@"):
            rows.append(f"<div class='conv-diff-ctx'>{escaped}</div>")
        elif line.startswith("+"):
            rows.append(f"<div class='conv-diff-add'>{escaped}</div>")
        elif line.startswith("-"):
            rows.append(f"<div class='conv-diff-del'>{escaped}</div>")
        else:
            rows.append(f"<div class='conv-diff-line'>{escaped}</div>")
    if not rows: return None
    return f"<div class='conv-diff-block'>{''.join(rows)}</div>"


def _render_diff(before: str, after: str) -> None:
    diff_html = _unified_diff_html(before, after)
    if diff_html:
        st.markdown(diff_html, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Before")
            st.code(before or "N/A")
        with col2:
            st.caption("After")
            st.code(after or "N/A")


def _render_code_context_viewer(code: str, target_line, context: int = CODE_VIEWER_CONTEXT_LINES) -> None:
    if not code or not target_line: return
    try:
        target_line = int(target_line)
    except (TypeError, ValueError):
        return
    lines = code.splitlines()
    if not lines: return
    start = max(0, target_line - 1 - context)
    end = min(len(lines), target_line + context)
    rows = []
    for i in range(start, end):
        ln = i + 1
        text = html.escape(lines[i])
        row_cls = "conv-code-row target" if ln == target_line else "conv-code-row"
        rows.append(f"<div class='{row_cls}'><span class='conv-code-ln'>{ln}</span><span class='conv-code-txt'>{text}</span></div>")
    st.markdown(f"<div class='conv-code-viewer'>{''.join(rows)}</div>", unsafe_allow_html=True)


# ============================================================================
# EXPORT — PDF only
# ============================================================================
def _conversation_to_pdf(messages: list) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    user_style = ParagraphStyle("UserMsg", parent=styles["Normal"], textColor=HexColor("#0EA5E9"), fontSize=10.5, spaceAfter=4, fontName="Helvetica-Bold")
    assistant_style = ParagraphStyle("AssistantMsg", parent=styles["Normal"], textColor=HexColor("#1E293B"), fontSize=10, spaceAfter=10, leading=14)
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], textColor=HexColor("#64748B"), fontSize=8.5)
    story = [Paragraph("AI Secure Code Assistant — Conversation Log", styles["Title"]), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), meta_style), Spacer(1, 12)]
    for m in messages:
        who = "Developer" if m["role"] == "user" else "Assistant"
        story.append(Paragraph(f"{who} ({_safe(m.get('ts', ''))})", user_style))
        story.append(Paragraph(_safe(m["content"]).replace("\n", "<br/>"), assistant_style))
        story.append(HRFlowable(width="100%", color=HexColor("#E2E8F0")))
        story.append(Spacer(1, 6))
    doc.build(story)
    return buffer.getvalue()


def _get_export_pdf(messages: list) -> bytes:
    """
    Only regenerate the PDF when the conversation has actually changed,
    instead of rebuilding the whole reportlab document on every rerun.
    """
    if not messages:
        return b""
    cache_key = f"{len(messages)}_{messages[-1].get('ts_iso', '')}"
    if st.session_state.get("_pdf_cache_key") != cache_key:
        st.session_state["_pdf_bytes"] = _conversation_to_pdf(messages)
        st.session_state["_pdf_cache_key"] = cache_key
    return st.session_state["_pdf_bytes"]


# ============================================================================
# RENDERING
# ============================================================================
def _render_message_footer(meta: dict, key_prefix: str) -> None:
    if not meta.get("show_footer"):
        return
    findings = meta.get("findings") or []
    sources = meta.get("sources") or []
    if sources:
        chips = "".join(f"<span class='conv-source-chip'>{_safe(s)}</span>" for s in sources)
        st.markdown(chips, unsafe_allow_html=True)
    if findings:
        with st.expander(f"Related findings ({len(findings)})", expanded=False):
            for i, f in enumerate(findings):
                sev = f.get("severity", "Info")
                owasp = f.get("owasp") or f.get("owasp_category")
                cwe = f.get("cwe") or f.get("cwe_id")
                st.markdown(
                    f"{_SEVERITY_EMOJI.get(sev, '⚪')} **{_safe(f.get('title'))}** — "
                    f"{_safe(f.get('file') or 'uploaded file')}:{_safe(f.get('line') or f.get('line_start') or 'N/A')}"
                    f"{'  ·  CWE ' + _safe(cwe) if cwe else ''}{'  ·  ' + _safe(owasp) if owasp else ''}"
                )
                link_bits = []
                if cwe and _cwe_link(cwe):
                    link_bits.append(f"[CWE docs]({_cwe_link(cwe)})")
                if owasp:
                    link_bits.append(f"[OWASP docs]({_owasp_link(owasp)})")
                if link_bits:
                    st.markdown(" · ".join(link_bits))
                code = st.session_state.get("conv_current_code") or ""
                line = f.get("line") or f.get("line_start")
                remediation = f.get("remediation")
                cols = st.columns(2)
                with cols[0]:
                    if code and line and st.button("View code", key=f"{key_prefix}_code_{i}"):
                        st.session_state.conv_expanded_finding = f"{key_prefix}_{i}"
                with cols[1]:
                    if remediation and st.button("View fix diff", key=f"{key_prefix}_diff_{i}"):
                        st.session_state.conv_expanded_finding = f"{key_prefix}_diff_{i}"
                if st.session_state.conv_expanded_finding == f"{key_prefix}_{i}" and code and line:
                    _render_code_context_viewer(code, line)
                if st.session_state.conv_expanded_finding == f"{key_prefix}_diff_{i}" and remediation:
                    _render_diff(remediation.get("before_snippet") or f.get("code_snippet") or "", remediation.get("after_snippet") or "")


def _render_turn(user_msg: dict, assistant_msg: dict, idx: int) -> None:
    # User bubble — right aligned, with a double blue checkmark by the timestamp.
    st.markdown(
        f"""<div class="conv-row user">
                <div>
                    <div class="conv-bubble-user">{_safe(user_msg['content'])}</div>
                    <div class="conv-ts-row">
                        <span class="conv-ts">{_safe(user_msg.get('ts',''))}</span>
                        <span class="conv-check">✓✓</span>
                    </div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )

    avatar_col, card_col = st.columns([0.06, 0.94], gap="small")
    with avatar_col:
        st.markdown(_BOT_AVATAR_SVG, unsafe_allow_html=True)
    with card_col:
        with st.container(border=True, key=f"ai_card_{idx}"):
            st.markdown(assistant_msg["content"])
        meta = assistant_msg.get("meta") or {}
        if meta:
            _render_message_footer(meta, key_prefix=f"turn_{idx}")
        _copy_block("Copy", assistant_msg["content"], language="markdown")
        st.markdown(f"<div class='conv-ts'>{_safe(assistant_msg.get('ts',''))}</div>", unsafe_allow_html=True)


def _render_pending_turn(user_msg: dict) -> None:
    """
    Renders just the user bubble plus a lightweight 'thinking' placeholder
    on the assistant side, for a question that has been submitted but
    doesn't have an answer yet.
    """
    st.markdown(
        f"""<div class="conv-row user">
                <div>
                    <div class="conv-bubble-user">{_safe(user_msg['content'])}</div>
                    <div class="conv-ts-row">
                        <span class="conv-ts">{_safe(user_msg.get('ts',''))}</span>
                        <span class="conv-check">✓✓</span>
                    </div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )
    avatar_col, card_col = st.columns([0.06, 0.94], gap="small")
    with avatar_col:
        st.markdown(
            f'<div class="conv-avatar-bot thinking">{_bot_face_svg(22)}</div>',
            unsafe_allow_html=True,
        )
    with card_col:
        with st.container(border=True, key="ai_card_pending"):
            st.markdown(
                """<div class="conv-thinking-wrap">
                        <span class="conv-thinking-dot">●</span>
                        <span class="conv-thinking-dot">●</span>
                        <span class="conv-thinking-dot">●</span>
                   </div>""",
                unsafe_allow_html=True,
            )


def _submit_question(question: str, on_new_message: Optional[Callable[[str, str], None]] = None) -> None:
    """
    Step 1 of sending a message: append ONLY the user's question and mark
    it as awaiting a reply, then the caller reruns immediately. This is
    what makes the question bubble appear on screen right away instead of
    waiting for the (potentially slow) Gemini call to finish first.

    `on_new_message`, when the caller supplies it, is invoked right after
    the append with ("user", question) so a mid-turn failure never loses
    the question that was actually asked (mirrors
    analysis_history.append_chat_message's own per-message contract).
    """
    question = question.strip()
    if not question:
        return
    user_ts = datetime.now().strftime("%H:%M:%S")
    _get_msgs().append(
        {"role": "user", "content": question, "ts": user_ts, "ts_iso": datetime.now().isoformat()}
    )
    st.session_state.conv_awaiting_answer = True
    if on_new_message:
        on_new_message("user", question)


def _answer_pending_question(findings: list, pr_summary: Optional[dict], uploaded_code: str,
                             language: str, deep_result: dict, api_key: str,
                             on_new_message: Optional[Callable[[str, str], None]] = None) -> None:
    """
    Step 2: runs on the NEXT script pass (after the question is already
    visible on screen). Generates the answer and appends it.
    """
    msgs = _get_msgs()
    if not msgs or msgs[-1]["role"] != "user":
        st.session_state.conv_awaiting_answer = False
        return

    question = msgs[-1]["content"]
    result = _answer_for(question, findings, pr_summary, uploaded_code, language, deep_result, api_key)

    meta = {
        "findings": result["findings"], "kb_hits": result["kb_hits"], "mode": result["mode"],
        "sources": result["sources"], "confidence": result["confidence"],
        "show_footer": _wants_metadata(question),
    }

    if result["findings"]:
        _set_last_focus(result["findings"][0])

    msgs.append(
        {"role": "assistant", "content": result["answer"], "meta": meta, "ts": result["timestamp"], "ts_iso": result["timestamp_iso"]}
    )
    st.session_state.conv_awaiting_answer = False
    if on_new_message:
        on_new_message("assistant", result["answer"])


# ============================================================================
# MAIN ENTRY — matches wireframe: header → conversation → input bar
# ============================================================================
def render_conversational_assistant(
    deep_result: dict, findings: list, pr_summary: Optional[dict], uploaded_code: str,
    language: str, api_key: str, focus_input: bool = False,
    on_new_message: Optional[Callable[[str, str], None]] = None,
    empty_state_text: Optional[str] = None,
    state_key: Optional[str] = None,
) -> None:
    """
    focus_input: pass True on the single render right after the caller's
    own "jump to assistant" button was clicked (app.py already does this
    via a one-shot st.session_state.pop(...) flag before calling this
    function). When True, the chat input bar is scrolled into view,
    briefly pulsed, and focused. Leave as False on every other render
    (tab switches back to this same tab, sending a message, etc.) so the
    page doesn't jump around unexpectedly.

    on_new_message: optional callback(role, content) invoked right after a
        message is appended to the conversation — once for the user's
        question, once for the assistant's answer. app.py uses this to
        persist each turn to analysis_history the moment it happens, so a
        mid-turn failure never loses the question that was actually asked.
        Safe to omit: when None (the default — used by the live
        Paste/Upload/GitHub/URL tabs), nothing is persisted and behavior
        is exactly as before this fix.

    empty_state_text: optional override for the secondary line shown in
        the empty-conversation state (e.g. "No previous conversation for
        this analysis." right after opening a History record with no
        saved messages yet). Falls back to the original generic copy when
        omitted, so existing tabs look exactly as before.

    state_key: identifies which analysis this conversation belongs to
        (e.g. "history"). Only takes effect together with on_new_message:
        in that case the conversation list is stored under
        `chat_messages_<state_key>` in st.session_state — the very same
        key app.py already loads from / refills from analysis_history
        before calling this function — instead of the shared
        "conv_messages" key, so a persisted analysis keeps its own
        history (no cross-analysis bleed) without ever creating a second,
        duplicate copy of it. When on_new_message is None, state_key is
        ignored and the original shared "conv_messages" key is used, so
        the live tabs' existing single continuous conversation is
        completely unaffected by this change.

        IMPORTANT: the caller (app.py) must actually pass this argument
        for a History record's persisted chat to be visible — without it,
        this function falls back to the shared "conv_messages" key while
        app.py separately preloads the record's saved turns into
        "chat_messages_<state_key>", and the two never meet.
    """
    _inject_styles()
    messages_key = (
        f"chat_messages_{state_key}" if (state_key and on_new_message is not None) else "conv_messages"
    )
    _init_state(messages_key=messages_key)
    findings = findings or (deep_result or {}).get("findings", []) or []
    st.session_state["conv_current_code"] = uploaded_code or ""
    st.session_state["conv_current_language"] = language or ""

    # Persistent mascot (req. 1/7): rendered once per script run, outside
    # the scrolling message list, so it stays put while the conversation
    # scrolls underneath it. Reflects the *current* awaiting state, so it
    # pulses during generation and goes idle the run after the answer lands.
    _render_persistent_avatar(thinking=st.session_state.conv_awaiting_answer)

    with st.container(border=True, key="app_shell"):

        # ---- Header: icon + title left, Download PDF right ----
        header_title, header_right = st.columns([3, 1])
        with header_title:
            st.markdown(
                """<div class="conv-header-title-row">
                        <span class="conv-header-icon-badge">
                            <svg width="18" height="20" viewBox="0 0 24 26" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 1L21 4.5V11.5C21 17.5 17 22.5 12 25C7 22.5 3 17.5 3 11.5V4.5L12 1Z"
                                      stroke="#ffffff" stroke-width="1.8" fill="none"/>
                                <path d="M8 12.5L11 15.5L16.5 9.5" stroke="#ffffff" stroke-width="2"
                                      stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </span>
                        <span class="conv-header-title">AI Secure Code Assistant</span>
                    </div>""",
                unsafe_allow_html=True,
            )
        with header_right:
            current_msgs = _get_msgs()
            pdf_data = _get_export_pdf(current_msgs) if current_msgs else b""
            st.download_button(
                "↓ Download PDF", data=pdf_data, file_name="conversation.pdf", mime="application/pdf",
                disabled=not current_msgs, use_container_width=True, key="export_pdf",
            )
        st.markdown('<div style="border-bottom:1px solid #22224a; margin-bottom:0.8rem;"></div>', unsafe_allow_html=True)

        if not (api_key or "").strip():
            st.caption("No Gemini API key detected — slash commands still work.")
        if not (deep_result and findings):
            st.info("Run Deep Analysis first so the assistant can ground answers in your scan.")

        # ---- Conversation area ----
        msgs = _get_msgs()
        awaiting = st.session_state.conv_awaiting_answer

        has_pending = bool(msgs) and msgs[-1]["role"] == "user" and awaiting
        complete_msgs = msgs[:-1] if has_pending else msgs

        if not complete_msgs and not has_pending:
            empty_body = _safe(
                empty_state_text
                or "I can help you understand findings, vulnerabilities, remediation, and your code analysis."
            )
            st.markdown(
                f"""<div class="conv-empty-state">
                    {_BOT_HERO_SVG}
                    <div style="font-weight:600; color:#e6e6f5; margin-bottom:0.3rem;">
                        Hi! I'm your Code Security Assistant.
                    </div>
                    <div>{empty_body}</div>
                   </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="conv-scroll-area">', unsafe_allow_html=True)
            pairs = list(zip(complete_msgs[0::2], complete_msgs[1::2]))
            for idx, (user_msg, assistant_msg) in enumerate(pairs):
                _render_turn(user_msg, assistant_msg, idx=idx)
            if has_pending:
                _render_pending_turn(msgs[-1])
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Bottom input bar ----
        with st.container(key="chat_input_container"):
            with st.form(key=f"conv_form_{st.session_state.conv_input_nonce}", clear_on_submit=True, border=False):
                in_col, btn_col = st.columns([6.2, 1])
                with in_col:
                    typed = st.text_input(
                        "Ask", placeholder="Ask something…", label_visibility="collapsed", disabled=awaiting,
                    )
                with btn_col:
                    submitted = st.form_submit_button("➤ Send", use_container_width=True, type="primary", disabled=awaiting)

    if focus_input:
        _scroll_to_chat_input()

    submitted_question = typed.strip() if (submitted and typed and typed.strip()) else None

    if submitted_question and not awaiting:
        # Phase 1: show the question immediately, THEN rerun.
        st.session_state.conv_input_nonce += 1
        _submit_question(submitted_question, on_new_message=on_new_message)
        st.rerun()

    if awaiting and has_pending:
        # Phase 2: generate the answer and rerun once more to show it as a
        # finished turn with its full footer/copy button.
        _answer_pending_question(
            findings, pr_summary, uploaded_code, language, deep_result, api_key,
            on_new_message=on_new_message,
        )
        st.rerun()
        