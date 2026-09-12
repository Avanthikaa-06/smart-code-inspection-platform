import html
import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

import auth
import llm_router
from analysis_history import (
    init_db,
    new_analysis_id,
    report_path_for,
    report_belongs_to_user,
    save_history_record,
    list_history_for_user,
    get_history_record,
    delete_history_record,
    get_chat_history,
    append_chat_message,
)

from agents.orchestrator import run_analysis
from backend import (
    clone_and_extract,
    download_file,
    process_pasted_code,
    process_uploaded_file,
    process_url_file,
)
from rag_engine import RAGPipeline
from findings_display import render_findings_display
from agents.prsummaryagent import generate_pr_summary
from agents.conversationalagent import render_conversational_assistant
from pr_summary_pdf import generate_pr_pdf

# ==============================================================================
# STREAMLIT PAGE CONFIG & STYLES  (Glassmorphism Pro - Pink/Violet Theme)
# ==============================================================================
st.set_page_config(
    page_title="Smart Code Inspection Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton, section[data-testid="stSidebar"] { display: none !important; }

/* ================================================================
   SCROLL-TRIGGERED REVEAL SYSTEM (IntersectionObserver driven)
   Placed early so later :hover / active-state rules below win the
   cascade on ties, keeping hover effects intact on revealed elements.
   ================================================================ */
.scroll-reveal,
.scroll-reveal-left,
.scroll-reveal-right,
.card-reveal,
.premium-card,
.badge-row,
.pipeline-strip,
.section-head,
.section-desc,
.status-chip,
.pr-download-bar,
.app-footer,
.chat-shell,
[data-testid="stAlert"],
[data-testid="stExpander"],
[data-testid="stTabs"],
.st-key-hero_actions_container {
    opacity: 0;
    transform: translateY(30px);
    transition: opacity 0.7s cubic-bezier(0.22, 1, 0.36, 1),
                transform 0.7s cubic-bezier(0.22, 1, 0.36, 1);
    will-change: opacity, transform;
}

.scroll-reveal.is-visible,
.scroll-reveal-left.is-visible,
.scroll-reveal-right.is-visible,
.card-reveal.is-visible,
.premium-card.is-visible,
.badge-row.is-visible,
.pipeline-strip.is-visible,
.section-head.is-visible,
.section-desc.is-visible,
.status-chip.is-visible,
.pr-download-bar.is-visible,
.app-footer.is-visible,
.chat-shell.is-visible,
[data-testid="stAlert"].is-visible,
[data-testid="stExpander"].is-visible,
[data-testid="stTabs"].is-visible,
.st-key-hero_actions_container.is-visible {
    opacity: 1;
    transform: translateY(0);
}

/* Directional variants — override the transform axis only */
.scroll-reveal-left { transform: translateX(-25px); }
.scroll-reveal-right { transform: translateX(25px); }
.scroll-reveal-left.is-visible,
.scroll-reveal-right.is-visible { transform: translateX(0); }

/* Card-specific timing (slightly snappier than full-section reveals) */
.card-reveal {
    transition: opacity 0.6s cubic-bezier(0.22, 1, 0.36, 1),
                transform 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}

/* Severity / status chips: subtle scale-in instead of a slide */
.status-chip { transform: scale(0.95); }
.status-chip.is-visible { transform: scale(1); }

/* Staggered children — e.g. hero heading -> subtitle -> badges */
.scroll-stagger > * {
    opacity: 0;
    transform: translateY(24px);
    transition: opacity 0.6s cubic-bezier(0.22, 1, 0.36, 1),
                transform 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}
.scroll-stagger.is-visible > *:nth-child(1) { transition-delay: 0s; }
.scroll-stagger.is-visible > *:nth-child(2) { transition-delay: 0.1s; }
.scroll-stagger.is-visible > *:nth-child(3) { transition-delay: 0.2s; }
.scroll-stagger.is-visible > *:nth-child(4) { transition-delay: 0.3s; }
.scroll-stagger.is-visible > * {
    opacity: 1;
    transform: translateY(0);
}

/* Action-button row under the hero (nav_col) fades in slightly after the hero */
.st-key-hero_actions_container {
    transition-delay: 0.15s;
}

@media (prefers-reduced-motion: reduce) {
    .scroll-reveal, .scroll-reveal-left, .scroll-reveal-right, .card-reveal,
    .premium-card, .badge-row, .pipeline-strip,
    .section-head, .section-desc, .status-chip, .scroll-stagger > *,
    .st-key-hero_actions_container {
        opacity: 1 !important;
        transform: none !important;
        transition: none !important;
    }
}

.stApp {
    background:
        radial-gradient(circle at 15% -10%, rgba(124,58,237,0.20) 0%, transparent 45%),
        radial-gradient(circle at 85% 0%, rgba(192,38,211,0.14) 0%, transparent 40%),
        radial-gradient(circle at 20% 0%, #1E1440 0%, #0D0818 55%);
    color: #F8FAFC;
}

/* ---------- Sticky Pill Navbar ---------- */
/* Both the bare class and the class combined with Streamlit's own
   data-testid attribute selector — whichever actually matches the real
   element, this guarantees our `position: sticky` has at least as much
   specificity as any of Streamlit's internal rules on the same wrapper
   div, so it can't silently lose that fight. */
.st-key-sticky_nav_container,
div[data-testid="stVerticalBlock"].st-key-sticky_nav_container,
div[data-testid="stVerticalBlockBorderWrapper"].st-key-sticky_nav_container {
    position: -webkit-sticky !important;
    position: sticky !important;
    top: 14px !important;
    z-index: 999999 !important;
    width: min(92%, 1200px);
    max-width: 1200px;
    margin: 0 auto 1.8rem auto !important;
    background: rgba(24, 14, 46, 0.86);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(167, 139, 250, 0.22);
    border-radius: 999px;
    padding: 0.5rem 1.3rem !important;
    box-shadow: 0 10px 34px rgba(0, 0, 0, 0.4);
    animation: navDrop 0.5s ease-out;
    align-self: flex-start;
}
@keyframes navDrop {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}
/* Sentinel marking where the navbar normally sits — invisible, takes no
   space, exists purely so JS can detect when scrolling has carried the
   navbar's original position out of view (i.e. it's now the sticky one). */
#nav-sentinel { height: 0; margin: 0; padding: 0; }

/* Brief fade + translateY pulse played each time the navbar transitions
   between its normal in-flow position and the stuck/sticky state (in
   either direction), so the change feels smooth rather than sudden.
   Purely a transition effect layered on top of the existing sticky
   pill — no change to its design, size, colors, or typography. */
.st-key-sticky_nav_container.navbar-pulse {
    animation: navbarPulse 0.5s cubic-bezier(0.22, 1, 0.36, 1);
}
@keyframes navbarPulse {
    0%   { opacity: 0.55; transform: translateY(-10px); }
    100% { opacity: 1; transform: translateY(0); }
}
/* Guard against any ancestor picking up a lingering `transform` — a
   transformed ancestor creates a new containing block and silently breaks
   position:sticky on its descendants (this does NOT touch overflow, so it
   can't break page scrolling the way the earlier fix did). Excludes the
   navbar and hero-actions containers themselves, since those rely on their
   own transform-based animations (navDrop / navbar-pulse / scroll-reveal). */
div[data-testid="stAppViewContainer"],
section[data-testid="stMain"],
.block-container,
div[data-testid="stVerticalBlock"]:not(.st-key-sticky_nav_container):not(.st-key-hero_actions_container),
div[data-testid="stVerticalBlockBorderWrapper"]:not(.st-key-sticky_nav_container):not(.st-key-hero_actions_container),
.element-container:not(.st-key-sticky_nav_container):not(.st-key-hero_actions_container) {
    transform: none !important;
}
/* small top breathing room — sticky nav now reserves its own space in flow */
.block-container { padding-top: 1.2rem !important; }


/* Force the row inside the pill to stay short and vertically centered
   instead of growing to whatever height the user-bar widget wants */
.st-key-sticky_nav_container [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    min-height: 0 !important;
    gap: 0.5rem;
}
.st-key-sticky_nav_container [data-testid="column"] {
    display: flex !important;
    align-items: center !important;
}
.st-key-sticky_nav_container [data-testid="stVerticalBlockBorderWrapper"],
.st-key-sticky_nav_container .element-container {
    margin: 0 !important;
}

.sticky-nav-brand {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    background: linear-gradient(100deg, #A78BFA 0%, #C026D3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    height: 100%;
}

/* ---------- Tab-switch fade/slide ---------- */
.tab-fade {
    animation: tabFadeSlide 0.35s ease both;
}
@keyframes tabFadeSlide {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---------- Chat message staggered entrance ---------- */
.chat-row { animation: chatRowIn 0.35s ease both; }
@keyframes chatRowIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-row:nth-child(1) { animation-delay: 0.02s; }
.chat-row:nth-child(2) { animation-delay: 0.06s; }
.chat-row:nth-child(3) { animation-delay: 0.10s; }
.chat-row:nth-child(4) { animation-delay: 0.14s; }
.chat-row:nth-child(5) { animation-delay: 0.18s; }
.chat-row:nth-child(n+6) { animation-delay: 0.20s; }

/* ---------- Hero ---------- */
.hero-wrap { text-align: center; margin-bottom: 1.2rem; }
.hero-title {
    font-size: 2.8rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(100deg, #A78BFA 0%, #7C3AED 45%, #C026D3 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 6s ease-in-out infinite;
}
@keyframes shimmer {
    0%   { background-position: 0% center; }
    50%  { background-position: 100% center; }
    100% { background-position: 0% center; }
}
.hero-subtitle { color: #C3CEEA; font-size: 1.05rem; margin-top: 0.45rem; font-weight: 300; }

/* ---------- Badge row (hero) ---------- */
.badge-row { display:flex; justify-content:center; gap:0.55rem; flex-wrap:wrap; margin-top:1.1rem; }
.hero-badge {
    padding:0.34rem 0.9rem; border-radius:999px; font-size:0.76rem; font-weight:600;
    background:rgba(167,139,250,0.1); color:#C4B5FD; border:1px solid rgba(167,139,250,0.25);
    letter-spacing:0.02em;
}

/* ---------- Section headers ---------- */
.section-head {
    display:flex; align-items:center; gap:0.55rem; margin: 1.4rem 0 0.7rem;
    font-size: 1.08rem; font-weight: 650; color: #F1F5F9; letter-spacing: -0.005em;
}
.section-head .bar { width: 4px; height: 18px; border-radius: 4px; background: linear-gradient(180deg, #A78BFA, #C026D3); }
.section-desc { color: #C3CEEA; font-size: 0.86rem; margin: -0.4rem 0 0.9rem; max-width: 60rem; }

/* ---------- Glass cards ---------- */
.premium-card {
    background: rgba(35, 20, 66, 0.45);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(167, 139, 250, 0.16);
    border-radius: 14px; padding: 1.4rem 1.6rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.28);
    margin-bottom: 1.1rem;
}
.premium-card:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(167, 139, 250, 0.32);
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.32);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.premium-card.card-flat {
    backdrop-filter: none;
    -webkit-backdrop-filter: none;
    background: rgba(35, 20, 66, 0.62);
}

/* Severity-based glow */
.glow-critical { box-shadow: 0 0 0 1px rgba(244,63,94,0.35), 0 0 32px rgba(244,63,94,0.22), 0 8px 30px rgba(0,0,0,0.3); border-color: rgba(244,63,94,0.4) !important; }
.glow-high     { box-shadow: 0 0 0 1px rgba(249,115,22,0.28), 0 0 22px rgba(249,115,22,0.14), 0 8px 30px rgba(0,0,0,0.28); border-color: rgba(249,115,22,0.32) !important; }
.glow-medium   { box-shadow: 0 0 0 1px rgba(245,158,11,0.2), 0 8px 30px rgba(0,0,0,0.26); }

/* ---------- Status chips ---------- */
.status-chip { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.01em; }
.status-valid { background: rgba(34, 197, 94, 0.14); color: #4ADE80; border: 1px solid rgba(34, 197, 94, 0.35); }
.status-invalid { background: rgba(244, 63, 94, 0.14); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.35); }

/* ---------- Confidence chips (Remediation tab) ---------- */
.conf-auto { background: rgba(34,197,94,0.14); color:#4ADE80; border:1px solid rgba(34,197,94,0.35); }
.conf-review { background: rgba(245,158,11,0.14); color:#FBBF24; border:1px solid rgba(245,158,11,0.35); }
.conf-manual { background: rgba(244,63,94,0.14); color:#FB7185; border:1px solid rgba(244,63,94,0.35); }
.conf-none { background: rgba(167,139,250,0.12); color:#C3CEEA; border:1px solid rgba(167,139,250,0.25); }

/* ---------- Metric tiles ---------- */
.metric-grid {
    display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 0.9rem; margin-bottom: 1.25rem;
    animation: metricGridIn 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes metricGridIn {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.metric-grid .metric-chip:nth-child(1) { animation: metricGridIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) both; animation-delay: 0.02s; }
.metric-grid .metric-chip:nth-child(2) { animation: metricGridIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) both; animation-delay: 0.07s; }
.metric-grid .metric-chip:nth-child(3) { animation: metricGridIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) both; animation-delay: 0.12s; }
.metric-grid .metric-chip:nth-child(4) { animation: metricGridIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) both; animation-delay: 0.17s; }
.metric-chip {
    background: rgba(26, 16, 48, 0.5); border: 1px solid rgba(167, 139, 250, 0.14);
    border-radius: 12px; padding: 1.05rem 1.1rem; text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.metric-chip:hover { transform: translateY(-4px); border-color: rgba(192,38,211,0.35); box-shadow: 0 10px 25px rgba(124,58,237,0.28); }
.metric-icon { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #A78BFA; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.35rem; opacity: 0.85; }
.metric-label { color: #C3CEEA; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.3rem; }
.metric-value { color: #F8FAFC; font-size: 1.25rem; font-weight: 700; overflow-wrap: anywhere; font-family: 'JetBrains Mono', monospace; }

/* ---------- Pipeline status strip ---------- */
.pipeline-strip { display:flex; flex-wrap:wrap; gap:0.6rem; margin-bottom:1.1rem; }
.pipeline-pill {
    display:flex; align-items:center; gap:0.45rem; padding:0.45rem 0.9rem; border-radius:999px;
    font-size:0.8rem; font-weight:600; border:1px solid rgba(167,139,250,0.2);
    background: rgba(26,16,48,0.5); color:#C3CEEA;
}
.pipeline-pill .dot { width:8px; height:8px; border-radius:50%; background:#475569; }
.pipeline-pill.done { color:#4ADE80; border-color:rgba(34,197,94,0.35); }
.pipeline-pill.done .dot { background:#4ADE80; box-shadow:0 0 8px rgba(74,222,128,0.6); }
.pipeline-pill.warn { color:#FBBF24; border-color:rgba(245,158,11,0.35); }
.pipeline-pill.warn .dot { background:#FBBF24; box-shadow:0 0 8px rgba(251,191,36,0.6); }
.pipeline-pill.fail { color:#FB7185; border-color:rgba(244,63,94,0.35); }
.pipeline-pill.fail .dot { background:#FB7185; box-shadow:0 0 8px rgba(251,113,133,0.6); }

/* ---------- Tool status table ---------- */
.tool-row { display:grid; grid-template-columns: 1.3fr 1fr 1fr 0.6fr 0.7fr; gap:0.75rem; color:#CBD5E1; padding:0.5rem 0; border-bottom:1px solid rgba(167,139,250,0.12); font-size:0.88rem; align-items:center; }
.tool-status-success { color:#4ADE80; font-weight:600; }
.tool-status-failed { color:#FB7185; font-weight:600; }
.tool-status-skipped { color:#C3CEEA; font-weight:600; }

/* ---------- Findings ---------- */
.finding-title { font-size: 1.02rem; font-weight: 650; margin-top: 0.6rem; color: #F8FAFC; }
.finding-meta { color:#C3CEEA; font-size: 0.8rem; margin-left: 0.6rem; font-family: 'JetBrains Mono', monospace; }
.finding-meta code { color: #C4B5FD; background: rgba(167,139,250,0.1); padding: 0.05rem 0.4rem; border-radius: 5px; }

/* ---------- Diff view ---------- */
.diff-before { background: rgba(244,63,94,0.08); border:1px solid rgba(244,63,94,0.28); border-radius:10px; padding:0.15rem; }
.diff-after  { background: rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.28); border-radius:10px; padding:0.15rem; }
.diff-label-before { color:#FB7185; font-weight:600; font-size:0.82rem; margin-bottom:0.35rem; }
.diff-label-after { color:#4ADE80; font-weight:600; font-size:0.82rem; margin-bottom:0.35rem; }

/* ---------- Severity Tabs (Custom Active Glow) ---------- */
.severity-tab-scope [data-baseweb="tab-list"] {
    gap: 10px;
    justify-content: flex-start;
    border-bottom: 1px solid rgba(167, 139, 250, 0.2);
    padding-bottom: 8px;
    margin-bottom: 1.1rem;
}
.severity-tab-scope [data-baseweb="tab"] {
    background: rgba(35, 20, 66, 0.45);
    border: 1px solid rgba(167, 139, 250, 0.22);
    border-radius: 999px;
    padding: 6px 20px;
    color: #C3CEEA;
    font-weight: 600;
    font-size: 0.88rem;
    transition: all 0.25s ease;
}
.severity-tab-scope [data-baseweb="tab"]:hover {
    border-color: rgba(192, 38, 211, 0.45);
    color: #F8FAFC;
    transform: translateY(-1px);
}
.severity-tab-scope [aria-selected="true"] {
    background: linear-gradient(105deg, rgba(124, 58, 237, 0.65), rgba(192, 38, 211, 0.65)) !important;
    color: #FFFFFF !important;
    border-color: #A78BFA !important;
    box-shadow: 0 0 16px rgba(124, 58, 237, 0.5), inset 0 0 8px rgba(192, 38, 211, 0.3) !important;
}

/* ---------- Buttons / inputs ---------- */
.stButton>button {
    border-radius: 999px !important; font-weight: 600 !important; letter-spacing: 0.01em;
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(124,58,237,0.25) !important;
    filter: brightness(1.1);
}
.stButton>button[kind="primary"] {
    background: linear-gradient(105deg, #7C3AED, #C026D3) !important;
    border: none !important; color: white !important;
    box-shadow: 0 6px 22px rgba(124,58,237,0.45) !important;
}
.stButton>button[kind="primary"]:hover {
    box-shadow: 0 10px 28px rgba(124,58,237,0.55) !important;
}
.stTextArea textarea, .stTextInput input {
    background-color: rgba(30, 20, 60, 0.55) !important;
    color: #F8FAFC !important;
    border-radius: 14px !important;
    border-color: rgba(167,139,250,0.22) !important;
}

/* ---------- Sub-tab button row ---------- */
.subtab-row-marker { margin-bottom: 0.35rem; }
div[data-testid="stHorizontalBlock"] > div .stButton>button {
    width: 100%;
}
.stButton>button[kind="secondary"] {
    background: rgba(35, 20, 66, 0.45) !important;
    color: #C4B5FD !important;
    border: 1px solid rgba(167,139,250,0.25) !important;
}
.stButton>button[kind="secondary"]:hover {
    border-color: rgba(192,38,211,0.4) !important;
    color: #F8FAFC !important;
}

/* ---------- Conversational assistant (chat) ---------- */
.chat-shell {
    background: rgba(26, 16, 48, 0.55);
    border: 1px solid rgba(167, 139, 250, 0.16);
    border-radius: 16px;
    padding: 0.4rem 0.4rem 1rem;
    margin-bottom: 1rem;
}
.chat-scroll { max-height: 55vh; overflow-y: auto; padding: 0.8rem 0.6rem; }
.chat-row { display: flex; margin: 0.5rem 0; }
.chat-row.user { justify-content: flex-end; }
.chat-row.assistant { justify-content: flex-start; }
.chat-bubble {
    max-width: 78%; padding: 0.75rem 1rem; border-radius: 14px; font-size: 0.92rem; line-height: 1.55;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}
.chat-bubble.user {
    background: linear-gradient(100deg, #7C3AED, #C026D3); color: white; border-bottom-right-radius: 4px;
}
.chat-bubble.assistant {
    background: rgba(40, 26, 74, 0.75); color: #F1F5F9; border: 1px solid rgba(167,139,250,0.18);
    border-bottom-left-radius: 4px;
}
.chat-bubble code { background: rgba(167,139,250,0.14); color: #C4B5FD; padding: 0.05rem 0.35rem; border-radius: 4px; }
.chat-empty-state { text-align: center; color: #C3CEEA; padding: 2.2rem 1rem; font-size: 0.9rem; }

/* ---------- Footer ---------- */
.app-footer { text-align:center; color:#7C7195; font-size:0.8rem; margin-top:3rem; padding-top:1.5rem; border-top:1px solid rgba(167,139,250,0.16); }
.app-footer b { color:#C3CEEA; }
.app-footer .foot-badges { margin-top:0.5rem; display:flex; justify-content:center; gap:0.5rem; flex-wrap:wrap; }
.foot-badge { padding:0.22rem 0.7rem; border-radius:999px; font-size:0.7rem; background:rgba(167,139,250,0.1); color:#C4B5FD; border:1px solid rgba(167,139,250,0.22); }

/* ---------- Download bar ---------- */
.pr-download-bar {
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.75rem;
    background: rgba(35, 20, 66, 0.55);
    border: 1px solid rgba(167, 139, 250, 0.22);
    border-radius: 14px; padding: 1rem 1.3rem; margin-bottom: 1.3rem;
}
.pr-download-bar .pr-download-title { font-weight:700; color:#F8FAFC; font-size:1rem; }
.pr-download-bar .pr-download-sub { color:#C3CEEA; font-size:0.82rem; margin-top:0.15rem; }

/* ---------- History card action row ---------- */
.history-confirm-banner {
    background: rgba(244,63,94,0.1); border:1px solid rgba(244,63,94,0.35);
    border-radius: 10px; padding: 0.7rem 1rem; margin: 0.4rem 0 0.6rem; color:#FECACA; font-size:0.88rem;
}

/* Guard scroll-to-anchor targets: an empty 0-height div can make
   scrollIntoView land at an odd spot on some browsers. */
#analysis-section, .analysis-anchor, #assistant-chat-anchor {
    display: block;
    min-height: 1px;
    scroll-margin-top: 90px;
}

@media (max-width: 760px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  .tool-row { grid-template-columns: 1fr; }
  .hero-title { font-size: 2.1rem; }
  .pr-download-bar { flex-direction: column; align-items: flex-start; }
}
</style>
""", unsafe_allow_html=True)


def _inject_scroll_reveal() -> None:
    """Sets up an IntersectionObserver in the parent document so sections and
    cards animate into view only when the user actually scrolls to them,
    rather than all at once on page load. Re-scans on DOM mutations so newly
    rendered Streamlit content (tab switches, reruns, new findings) is
    picked up automatically."""
    components.html(
        """
        <script>
        (function() {
            function init() {
                var doc = window.parent.document;

                var observer = doc.__scrollRevealObserver;
                if (!observer) {
                    observer = new IntersectionObserver(function(entries) {
                        entries.forEach(function(entry) {
                            if (entry.isIntersecting) {
                                entry.target.classList.add('is-visible');
                                observer.unobserve(entry.target);
                            }
                        });
                    }, { threshold: 0.12, rootMargin: "0px 0px -60px 0px" });
                    doc.__scrollRevealObserver = observer;

                    new MutationObserver(function() { scan(observer); })
                        .observe(doc.body, { childList: true, subtree: true });
                }

                scan(observer);
            }

            function scan(observer) {
                var doc = window.parent.document;
                var selector = [
                    '.scroll-reveal', '.scroll-reveal-left', '.scroll-reveal-right',
                    '.card-reveal', '.premium-card', '.badge-row',
                    '.pipeline-strip', '.section-head', '.section-desc',
                    '.status-chip', '.scroll-stagger',
                    '.pr-download-bar', '.app-footer', '.chat-shell',
                    '[data-testid="stAlert"]', '[data-testid="stExpander"]', '[data-testid="stTabs"]',
                    '.st-key-hero_actions_container'
                ].join(',');
                doc.querySelectorAll(selector).forEach(function(el) {
                    if (!el.dataset.revealBound) {
                        el.dataset.revealBound = "true";
                        observer.observe(el);
                    }
                });
            }

            init();
        })();
        </script>
        """,
        height=0,
    )


_inject_scroll_reveal()


def _inject_navbar_scroll_animation() -> None:
    """Watches a sentinel placed right where the navbar normally sits.
    While the sentinel is visible, the navbar is still in its normal
    top-of-page position; the moment scrolling carries the sentinel out
    of view, the (already `position: sticky`) navbar has effectively
    become the fixed one — that's when we play a brief fade+translateY
    pulse so it doesn't just appear/disappear abruptly. Scrolling back
    up brings the sentinel back into view and re-plays the same pulse,
    giving a smooth return to the normal in-flow state. This purely adds
    a transition moment on top of the existing sticky behavior — it does
    not change the navbar's design, size, colors, or position logic."""
    components.html(
        """
        <script>
        (function() {
            var doc = window.parent.document;

            function playPulse(nav) {
                nav.classList.remove('navbar-pulse');
                // force reflow so the animation restarts every time,
                // even if it's re-triggered before the previous one ends
                void nav.offsetWidth;
                nav.classList.add('navbar-pulse');
                window.setTimeout(function () {
                    nav.classList.remove('navbar-pulse');
                }, 500);
            }

            function setup() {
                var sentinel = doc.getElementById('nav-sentinel');
                var nav = doc.querySelector('.st-key-sticky_nav_container');
                if (!sentinel || !nav) return false;

                // Bulletproof enforcement: inline styles beat any external
                // stylesheet rule regardless of CSS specificity, so even if
                // some other Streamlit-internal rule is overriding our
                // `position: sticky` in the <style> block, this can't lose
                // that fight. Re-applied on every setup pass in case
                // Streamlit re-renders and drops inline styles on rerun.
                nav.style.position = 'sticky';
                nav.style.top = '14px';
                nav.style.zIndex = '999999';

                if (sentinel.dataset.navObserverBound) return true;
                sentinel.dataset.navObserverBound = 'true';

                var observer = new IntersectionObserver(function() {
                    playPulse(nav);
                }, { threshold: 0 });
                observer.observe(sentinel);
                return true;
            }

            if (!setup()) {
                var attempts = 0;
                var interval = setInterval(function () {
                    attempts += 1;
                    if (setup() || attempts > 50) {
                        clearInterval(interval);
                    }
                }, 100);
            }
        })();
        </script>
        """,
        height=0,
    )


# ==============================================================================
# AUTH GATE
# ==============================================================================
_EPHEMERAL_SESSION_PREFIXES = ("result_", "deep_", "prev_score_", "active_subtab_", "chat_", "history_record_id")


def _clear_ephemeral_session_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(_EPHEMERAL_SESSION_PREFIXES):
            del st.session_state[key]
    st.session_state["github_files_cached"] = None
    st.session_state["active_view"] = "code_submission"


init_db()
if not auth.require_auth():
    _clear_ephemeral_session_state()
    st.stop()


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
st.session_state.setdefault("active_view", "code_submission")
st.session_state.setdefault("gemini_api_key", llm_router.get_primary_key() or "")
st.session_state.setdefault("rag_pipeline", RAGPipeline())
st.session_state.setdefault("rag_status", None)
st.session_state.setdefault("github_files_cached", None)

for key in ["paste", "upload", "github", "url", "history"]:
    st.session_state.setdefault(f"result_{key}", None)
    st.session_state.setdefault(f"deep_{key}", None)
    st.session_state.setdefault(f"prev_score_{key}", None)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _safe(value) -> str:
    return html.escape(str(value or ""))


def _llm_enabled() -> bool:
    return llm_router.any_configured()


def _section_head(label: str) -> None:
    st.markdown(f"<div class='section-head'><div class='bar'></div>{_safe(label)}</div>", unsafe_allow_html=True)


def _section_desc(text: str) -> None:
    st.markdown(f"<div class='section-desc'>{_safe(text)}</div>", unsafe_allow_html=True)


def _scroll_to_anchor(anchor_id: str) -> None:
    """Scrolls parent window smoothly to the target element. Combines a
    poll (in case the element already exists) with a MutationObserver
    (in case it appears later, e.g. while a large results dashboard is
    still being rendered) so it doesn't lose the race against Streamlit's
    own render timing — the previous version only polled for 2.5s on a
    fixed timer and could miss slower-rendering content."""
    components.html(
        f"""
        <script>
        (function() {{
            var doc = window.parent.document;
            var done = false;

            function scrollToTarget() {{
                if (done) return true;
                var el = doc.getElementById("{anchor_id}");
                if (el) {{
                    done = true;
                    el.scrollIntoView({{behavior: "smooth", block: "start"}});
                    return true;
                }}
                return false;
            }}

            if (scrollToTarget()) return;

            // Catch it the instant it's added to the DOM, however long
            // Streamlit takes to finish streaming/rendering the section.
            var observer = new MutationObserver(function() {{
                if (scrollToTarget()) {{
                    observer.disconnect();
                }}
            }});
            observer.observe(doc.body, {{ childList: true, subtree: true }});

            // Poll as a fallback too, and give up after 10s either way.
            var attempts = 0;
            var interval = setInterval(function() {{
                attempts += 1;
                if (scrollToTarget() || attempts > 100) {{
                    clearInterval(interval);
                    observer.disconnect();
                }}
            }}, 100);
        }})();
        </script>
        """,
        height=0,
    )



def planned_tools(language: str) -> list[str]:
    if language == "python":
        return ["Pylint", "Radon", "Python AST", "Semgrep", "Bandit"]
    if language == "java":
        return ["PMD", "javalang AST", "Semgrep"]
    return []


def _save_analysis_to_history(result: dict, deep_result: dict, state_key: str) -> None:
    user = auth.current_user()
    if not user or not deep_result:
        return

    saved_flag_key = f"history_saved_{state_key}"
    if st.session_state.get(saved_flag_key) is deep_result:
        return

    user_id = user["username"]
    summary = deep_result.get("summary") or {}
    by_sev = summary.get("by_severity", {})

    analysis_id = new_analysis_id()
    pdf_path = report_path_for(user_id, analysis_id)

    api_key = st.session_state.gemini_api_key.strip()
    use_llm = _llm_enabled()

    try:
        pr_data = generate_pr_summary(
            deep_result,
            code=result.get("code", ""),
            file_name=result.get("file_name") or deep_result.get("file"),
            api_key=api_key,
            use_llm=use_llm,
        )
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        generate_pr_pdf(
            deep_result,
            pr_data,
            pdf_path,
            project_name=result.get("file_name") or "Code Analysis Project",
        )
        if not (os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0):
            pdf_path = None
    except Exception:
        pdf_path = None

    try:
        save_history_record(
            user_id=user_id,
            username=user.get("full_name") or user_id,
            file_name=result.get("file_name") or "untitled",
            language=result.get("language") or "unknown",
            security_score=summary.get("security_score", 100),
            total_findings=summary.get("total", 0),
            critical_count=by_sev.get("Critical", 0),
            high_count=by_sev.get("High", 0),
            medium_count=by_sev.get("Medium", 0),
            low_count=by_sev.get("Low", 0),
            analysis_payload={"result": result, "deep_result": deep_result},
            report_path=pdf_path,
        )
        st.session_state[saved_flag_key] = deep_result
    except Exception as exc:
        st.warning(f"Analysis completed, but could not be saved to History: {exc}")


def run_deep_analysis(result: dict, state_key: str) -> None:
    deep_key = f"deep_{state_key}"

    previous = st.session_state.get(deep_key)
    if previous and previous.get("summary"):
        st.session_state[f"prev_score_{state_key}"] = previous["summary"].get("security_score")

    key = st.session_state.gemini_api_key.strip()
    use_llm = _llm_enabled()
    progress = st.progress(0, text="Preparing analysis")
    status = st.empty()
    progress.progress(20, text="Language detected")
    with st.spinner("Running Code Analysis, Security, Severity, and Remediation agents..."):
        st.session_state[deep_key] = run_analysis(
            result["code"],
            result["language"],
            api_key=key if use_llm else None,
            use_llm=use_llm,
            filepath=result.get("file_name"),
            rag_pipeline=st.session_state.rag_pipeline,
        )
    progress.progress(100, text="Analysis complete")

    deep = st.session_state.get(deep_key) or {}
    if deep.get("success", True):
        status.success("Pipeline completed.")
        st.toast("Analysis pipeline completed successfully.")
    else:
        status.error(f"Pipeline completed with errors: {deep.get('error', 'Unknown error.')}")
        st.toast("Pipeline completed with errors — see details above.")

    _save_analysis_to_history(result, deep, state_key)


# ------------------------------------------------------------------------- #
# Plotly Visuals
# ------------------------------------------------------------------------- #
_SEVERITY_COLORS = {
    "Critical": "#F43F5E",
    "High": "#F97316",
    "Medium": "#F59E0B",
    "Low": "#A78BFA",
}

_SEVERITY_CHIP_STYLE = {
    "Critical": "background:rgba(244,63,94,0.16); color:#FB7185; border:1px solid rgba(244,63,94,0.4);",
    "High": "background:rgba(249,115,22,0.14); color:#FB923C; border:1px solid rgba(249,115,22,0.35);",
    "Medium": "background:rgba(245,158,11,0.14); color:#FBBF24; border:1px solid rgba(245,158,11,0.35);",
    "Low": "background:rgba(167,139,250,0.14); color:#A78BFA; border:1px solid rgba(167,139,250,0.35);",
    "Info": "background:rgba(167,139,250,0.1); color:#C3CEEA; border:1px solid rgba(167,139,250,0.22);",
}

_PLOTLY_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Poppins, sans-serif", color="#CBD5E1"),
    margin=dict(l=0, r=10, t=10, b=0),
)


def render_score_gauge(score: int, previous_score: int = None, chart_key: str = "score_gauge") -> None:
    if score >= 80:
        bar_color = "#4ADE80"
    elif score >= 50:
        bar_color = "#F59E0B"
    else:
        bar_color = "#F43F5E"

    indicator_kwargs = dict(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 34, "color": "#F8FAFC", "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#64748B", "size": 10}},
            "bar": {"color": bar_color, "thickness": 0.32},
            "bgcolor": "rgba(26,16,48,0.4)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(244,63,94,0.10)"},
                {"range": [50, 80], "color": "rgba(245,158,11,0.10)"},
                {"range": [80, 100], "color": "rgba(74,222,128,0.10)"},
            ],
        },
    )

    if previous_score is not None and previous_score != score:
        indicator_kwargs["mode"] = "gauge+number+delta"
        indicator_kwargs["delta"] = {
            "reference": previous_score,
            "increasing": {"color": "#4ADE80"},
            "decreasing": {"color": "#F43F5E"},
        }

    fig = go.Figure(go.Indicator(**indicator_kwargs))
    fig.update_layout(**_PLOTLY_BASE_LAYOUT, height=190)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=chart_key)


def render_severity_chart(summary: dict, chart_key: str = "severity_chart") -> None:
    by_severity = summary.get("by_severity", {})
    order = ["Critical", "High", "Medium", "Low"]
    values = [by_severity.get(s, 0) for s in order]
    colors = [_SEVERITY_COLORS[s] for s in order]

    fig = go.Figure(go.Bar(
        x=values,
        y=order,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[str(v) for v in values],
        textposition="outside",
        textfont=dict(color="#F8FAFC", family="JetBrains Mono", size=12),
        hovertemplate="%{y}: %{x} finding(s)<extra></extra>",
    ))
    fig.update_layout(
        **_PLOTLY_BASE_LAYOUT,
        height=180,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, tickfont=dict(color="#CBD5E1", size=12), autorange="reversed"),
        bargap=0.35,
    )
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=chart_key)
    st.markdown("</div>", unsafe_allow_html=True)


_CATEGORY_COLORS = ["#A78BFA", "#C026D3", "#F97316", "#4ADE80", "#F43F5E", "#A5B4CF"]


def render_category_donut(category_breakdown: dict, chart_key: str = "category_donut") -> None:
    if not category_breakdown:
        return
    labels = list(category_breakdown.keys())
    values = list(category_breakdown.values())
    colors = [_CATEGORY_COLORS[i % len(_CATEGORY_COLORS)] for i in range(len(labels))]

    fig = go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0D0818", width=2)),
        textinfo="value",
        textfont=dict(color="#F8FAFC", family="JetBrains Mono", size=12),
        hovertemplate="%{label}: %{value} finding(s)<extra></extra>",
    )
    fig_layout = go.Figure(fig)
    fig_layout.update_layout(
        **_PLOTLY_BASE_LAYOUT,
        height=230,
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, font=dict(color="#CBD5E1", size=11)),
    )
    st.markdown("<div class='premium-card scroll-reveal-right'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Findings by Category</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_layout, use_container_width=True, config={"displayModeBar": False}, key=chart_key)
    st.markdown("</div>", unsafe_allow_html=True)


def render_auto_fixable_bar(auto_fixable: dict, chart_key: str = "auto_fixable_bar") -> None:
    tiers = [
        ("Auto-Applicable", auto_fixable["auto_applicable"], "#4ADE80"),
        ("Needs Review", auto_fixable["needs_review"], "#FBBF24"),
        ("Manual Review", auto_fixable["manual_review"], "#FB7185"),
    ]
    fig = go.Figure()
    for label, value, color in tiers:
        fig.add_trace(go.Bar(
            x=[value], y=["Fix Confidence"], orientation="h", name=label,
            marker=dict(color=color, line=dict(width=0)),
            text=[str(value) if value else ""], textposition="inside",
            textfont=dict(color="#0D0818", family="JetBrains Mono", size=12),
            hovertemplate=f"{label}: %{{x}}<extra></extra>",
        ))
    fig.update_layout(
        **_PLOTLY_BASE_LAYOUT,
        height=130,
        barmode="stack",
        showlegend=True,
        legend=dict(orientation="h", y=-0.4, font=dict(color="#CBD5E1", size=11)),
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
    )
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='metric-label'>Auto-Fixable — {auto_fixable['ratio_pct']}% of findings have a ready-to-review fix</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=chart_key)
    st.markdown("</div>", unsafe_allow_html=True)


def render_tool_statuses(tool_statuses: list) -> None:
    if not tool_statuses:
        return

    status_class = {"success": "tool-status-success", "failed": "tool-status-failed", "skipped": "tool-status-skipped"}
    rows = [
        "<div class='tool-row' style='font-weight:600; color:#F8FAFC;'><div>Tool</div><div>Agent</div><div>Status</div><div>Findings</div><div>Duration</div></div>"
    ]
    for status in tool_statuses:
        duration = status.get("duration_seconds")
        duration_label = f"{duration:.3f}s" if isinstance(duration, (int, float)) else "N/A"
        state = (status.get("status") or "").lower()
        css_class = status_class.get(state, "")
        rows.append(
            "<div class='tool-row'>"
            f"<div>{_safe(status.get('tool') or 'N/A')}</div>"
            f"<div>{_safe(status.get('agent') or 'N/A')}</div>"
            f"<div class='{css_class}'>{_safe(status.get('status') or 'N/A')}</div>"
            f"<div>{_safe(status.get('findings_count', 0))}</div>"
            f"<div>{_safe(duration_label)}</div>"
            "</div>"
        )

    st.markdown(f"<div class='premium-card'>{''.join(rows)}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------- #
# Pipeline Status Strip
# ------------------------------------------------------------------------- #
def render_pipeline_strip(deep_result: dict) -> None:
    agent_status = deep_result.get("agent_status", {}) or {}
    findings = deep_result.get("findings", []) or []
    use_llm_ran = _llm_enabled()

    def _pill_state(status_value: str) -> str:
        s = (status_value or "").lower()
        if s == "success":
            return "done"
        if s == "partial":
            return "warn"
        if s in ("failed", "skipped"):
            return "fail"
        return ""

    code_state = _pill_state(agent_status.get("code_analysis"))
    sec_state = _pill_state(agent_status.get("security_vulnerability"))

    remediation_seen = any(("remediation" in f or "remediation_unavailable_reason" in f) for f in findings)
    remediation_applied = any(f.get("remediation") for f in findings)
    if not remediation_seen:
        rem_state, rem_label = "", "Remediation: not run"
    elif remediation_applied:
        rem_state, rem_label = "done", "Remediation: generated"
    elif use_llm_ran:
        rem_state, rem_label = "warn", "Remediation: no fixes returned"
    else:
        rem_state, rem_label = "fail", "Remediation: skipped (no LLM provider configured)"

    severity_state = "done" if findings else "warn"
    severity_label = f"Severity: {'Gemini' if use_llm_ran else 'Local'} normalized"

    pills = [
        ("done", "Language &amp; Syntax: validated"),
        (code_state, f"Code Analysis: {agent_status.get('code_analysis', 'not run')}"),
        (sec_state, f"Security Analysis: {agent_status.get('security_vulnerability', 'not run')}"),
        (severity_state, severity_label),
        (rem_state, rem_label),
        ("done" if deep_result.get("success", True) else "fail", "Report: ready" if deep_result.get("success", True) else "Report: generated with errors"),
    ]

    html_pills = "".join(
        f"<div class='pipeline-pill {state}'><div class='dot'></div>{_safe(label)}</div>"
        for state, label in pills
    )
    st.markdown(f"<div class='pipeline-strip'>{html_pills}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------- #
# Shared Finding Cards & Severity-Tabs Renderer
# ------------------------------------------------------------------------- #
def _summarize_subset(findings: list) -> dict:
    by_severity = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Info": 0}
    for f in findings:
        sev = str(f.get("severity") or "Medium").capitalize()
        if sev in by_severity:
            by_severity[sev] += 1
    actionable_total = by_severity["Critical"] + by_severity["High"] + by_severity["Medium"] + by_severity["Low"]
    return {"total": actionable_total, "by_severity": by_severity}


def _render_single_finding_card(
    finding: dict,
    show_owasp: bool = True,
    show_snippet: bool = True,
    show_remediation: bool = False,
    reveal_index: int = 0,
) -> None:
    line = finding.get("line") or finding.get("line_start") or "N/A"
    severity = str(finding.get("severity") or "Medium").capitalize()
    glow_class = {"Critical": "glow-critical", "High": "glow-high", "Medium": "glow-medium"}
    card_glow = glow_class.get(severity, "")

    tool = finding.get("tool") or finding.get("agent_source") or "unknown"
    title = finding.get("title") or finding.get("category") or "Finding"
    desc = finding.get("description") or finding.get("message") or ""
    rec = finding.get("recommendation") or "Review and remediate this finding before release."
    snippet = finding.get("code_snippet")
    remediation = finding.get("remediation")

    # Sequential scroll-reveal stagger for finding cards (0.1s per card, capped)
    reveal_delay = min(reveal_index * 0.1, 0.6)

    meta_bits = [f"Line {_safe(line)}", f"<code>{_safe(tool)}</code>"]
    if show_owasp:
        cwe = finding.get("cwe") or finding.get("cwe_id") or "N/A"
        owasp = finding.get("owasp") or finding.get("owasp_category") or "N/A"
        meta_bits.append(f"CWE: {_safe(cwe)}")
        meta_bits.append(f"OWASP: {_safe(owasp)}")
    meta_html = " &middot; ".join(meta_bits)

    st.markdown(f"""
    <div class="premium-card card-flat card-reveal {card_glow}" style="transition-delay:{reveal_delay}s;">
        <span class="status-chip" style="{_SEVERITY_CHIP_STYLE.get(severity, _SEVERITY_CHIP_STYLE['Medium'])}">{_safe(severity)}</span>
        <span class="finding-meta">{meta_html}</span>
        <div class="finding-title">{_safe(title)}</div>
        <div style="margin-top:0.4rem; color:#F8FAFC; opacity:0.88; font-size:0.92rem;">{_safe(desc)}</div>
        <div style="margin-top:0.6rem; color:#CBD5E1; font-size:0.9rem;"><b style="color:#C3CEEA;">Recommendation:</b> {_safe(rec)}</div>
    </div>""", unsafe_allow_html=True)

    if show_snippet and snippet:
        with st.expander(f"Code snippet — line {line}"):
            st.code(snippet, language=finding.get("language") or "text")

    if show_remediation and remediation:
        with st.expander(f"Remediation Solution — {title}"):
            if remediation.get("root_cause"):
                st.markdown(f"**Root Cause:** {remediation['root_cause']}")

            before = remediation.get("before_snippet") or snippet or ""
            after = remediation.get("after_snippet") or ""
            if before or after:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("<div class='diff-label-before'>Before</div>", unsafe_allow_html=True)
                    st.code(before or "(none)", language=finding.get("language") or "text")
                with c2:
                    st.markdown("<div class='diff-label-after'>After</div>", unsafe_allow_html=True)
                    st.code(after or "(no fix available)", language=finding.get("language") or "text")

            if remediation.get("explanation"):
                st.markdown(f"**Explanation:** {remediation['explanation']}")
            if remediation.get("guideline_reference"):
                st.markdown(f"**Secure Coding Guideline:** {remediation['guideline_reference']}")
            if remediation.get("prevention_tip"):
                st.markdown(f"**Prevention Tip:** {remediation['prevention_tip']}")


def render_severity_tabbed_findings(
    findings: list,
    show_owasp: bool = True,
    show_snippet: bool = True,
    show_remediation: bool = False,
) -> None:
    groups = {
        "Critical": [],
        "High": [],
        "Medium": [],
        "Low": [],
    }

    for f in findings:
        sev = str(f.get("severity") or "Medium").capitalize()
        if sev in groups:
            groups[sev].append(f)

    st.markdown('<div class="severity-tab-scope">', unsafe_allow_html=True)

    t_crit, t_high, t_med, t_low = st.tabs([
        f"Critical ({len(groups['Critical'])})",
        f"High ({len(groups['High'])})",
        f"Medium ({len(groups['Medium'])})",
        f"Low ({len(groups['Low'])})",
    ])

    with t_crit:
        if groups["Critical"]:
            for idx, item in enumerate(groups["Critical"]):
                _render_single_finding_card(
                    item,
                    show_owasp=show_owasp,
                    show_snippet=show_snippet,
                    show_remediation=show_remediation,
                    reveal_index=idx,
                )
        else:
            st.markdown('<div class="premium-card" style="text-align:center; color:#C3CEEA;">No Critical findings</div>', unsafe_allow_html=True)

    with t_high:
        if groups["High"]:
            for idx, item in enumerate(groups["High"]):
                _render_single_finding_card(
                    item,
                    show_owasp=show_owasp,
                    show_snippet=show_snippet,
                    show_remediation=show_remediation,
                    reveal_index=idx,
                )
        else:
            st.markdown('<div class="premium-card" style="text-align:center; color:#C3CEEA;">No High findings</div>', unsafe_allow_html=True)

    with t_med:
        if groups["Medium"]:
            for idx, item in enumerate(groups["Medium"]):
                _render_single_finding_card(
                    item,
                    show_owasp=show_owasp,
                    show_snippet=show_snippet,
                    show_remediation=show_remediation,
                    reveal_index=idx,
                )
        else:
            st.markdown('<div class="premium-card" style="text-align:center; color:#C3CEEA;">No Medium findings</div>', unsafe_allow_html=True)

    with t_low:
        if groups["Low"]:
            for idx, item in enumerate(groups["Low"]):
                _render_single_finding_card(
                    item,
                    show_owasp=show_owasp,
                    show_snippet=show_snippet,
                    show_remediation=show_remediation,
                    reveal_index=idx,
                )
        else:
            st.markdown('<div class="premium-card" style="text-align:center; color:#C3CEEA;">No Low findings</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------- #
# 1. Code & Security Analysis Tab (Organized by Severity Tabs)
# ------------------------------------------------------------------------- #
def render_code_security_tab(findings: list, state_key: str = "code_security") -> None:
    _section_desc(
        "Code quality issues and security vulnerabilities together in one view — detected by "
        "Pylint, Radon, Python AST, PMD (Java), Semgrep, and Bandit, plus an optional Gemini review."
    )

    code_findings = [f for f in findings if f.get("agent") == "code_analysis"]
    security_findings = [f for f in findings if f.get("agent") == "security"]
    all_findings = code_findings + security_findings
    summary = _summarize_subset(all_findings)
    by_s = summary["by_severity"]

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-chip"><div class="metric-icon">Total</div><div class="metric-value">{summary["total"]}</div></div>
        <div class="metric-chip"><div class="metric-icon">Critical</div><div class="metric-value" style="color:#FB7185">{by_s["Critical"]}</div></div>
        <div class="metric-chip"><div class="metric-icon">High</div><div class="metric-value" style="color:#FB923C">{by_s["High"]}</div></div>
        <div class="metric-chip"><div class="metric-icon">Medium</div><div class="metric-value" style="color:#FBBF24">{by_s["Medium"]}</div></div>
    </div>""", unsafe_allow_html=True)

    render_severity_chart(summary, chart_key=f"severity_chart_combined_{state_key}")

    _section_head("Code & Security Findings")
    render_severity_tabbed_findings(all_findings, show_owasp=True, show_snippet=True, show_remediation=False)


# ------------------------------------------------------------------------- #
# 2. Remediation Tab
# ------------------------------------------------------------------------- #
_CONF_CLASS = {
    "auto-applicable": "conf-auto",
    "needs-review": "conf-review",
    "manual-review-required": "conf-manual",
}
_CONF_LABEL = {
    "auto-applicable": "Auto Applicable",
    "needs-review": "Needs Review",
    "manual-review-required": "Manual Review Required",
}


def render_remediation_tab(findings: list) -> None:
    _section_desc(
        "AI-generated fixes for each detected finding: root cause, a minimal before/after diff, "
        "why the fix works, a prevention tip, and citations from the secure coding knowledge base. "
        "Requires a configured LLM provider on the server — without one, remediation is skipped entirely."
    )

    if not _llm_enabled():
        st.warning(
            "No LLM provider is configured on the server, so the Remediation Agent did not run. "
            "Configure GEMINI_API_KEY (and optionally BACKUP_LLM_API_KEY) and re-run Deep Analysis "
            "to generate fixes."
        )

    remediated = [f for f in findings if f.get("remediation")]
    unavailable = [f for f in findings if not f.get("remediation")]

    conf_counts = {"auto-applicable": 0, "needs-review": 0, "manual-review-required": 0}
    for f in remediated:
        c = f["remediation"].get("application_confidence")
        if c in conf_counts:
            conf_counts[c] += 1

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-chip"><div class="metric-icon">Fixes Generated</div><div class="metric-value">{len(remediated)}</div></div>
        <div class="metric-chip"><div class="metric-icon">Auto Applicable</div><div class="metric-value" style="color:#4ADE80">{conf_counts["auto-applicable"]}</div></div>
        <div class="metric-chip"><div class="metric-icon">Needs Review</div><div class="metric-value" style="color:#FBBF24">{conf_counts["needs-review"]}</div></div>
        <div class="metric-chip"><div class="metric-icon">Manual Review</div><div class="metric-value" style="color:#FB7185">{conf_counts["manual-review-required"]}</div></div>
    </div>""", unsafe_allow_html=True)

    if not findings:
        st.markdown('<div class="premium-card" style="text-align:center; color:#C3CEEA;">No findings to remediate.</div>', unsafe_allow_html=True)
        return

    _section_head(f"Remediation Cards ({len(remediated)} generated, {len(unavailable)} unavailable)")

    for idx, finding in enumerate(findings, start=1):
        title = finding.get("title") or finding.get("category") or f"Finding {idx}"
        severity = finding.get("severity", "Medium")
        remediation = finding.get("remediation")

        with st.expander(f"{idx}. {title}  ·  {severity}", expanded=False):
            if not remediation:
                st.info(finding.get("remediation_unavailable_reason") or "Remediation not available for this finding.")
                st.write("**Existing recommendation:**")
                st.write(finding.get("recommendation") or "Review and remediate this finding before release.")
                continue

            conf = remediation.get("application_confidence")
            conf_class = _CONF_CLASS.get(conf, "conf-none")
            conf_label = _CONF_LABEL.get(conf, conf or "Unknown")

            top1, top2, top3 = st.columns(3)
            top1.markdown(f"<span class='status-chip {conf_class}'>{_safe(conf_label)}</span>", unsafe_allow_html=True)
            top2.metric("Syntax Validation", "Passed" if remediation.get("syntax_valid") else "Needs Review")
            top3.metric("Priority", finding.get("priority", "N/A"))

            st.markdown("**Root Cause**")
            st.write(remediation.get("root_cause") or "N/A")

            before = remediation.get("before_snippet") or finding.get("code_snippet") or ""
            after = remediation.get("after_snippet") or ""
            diff_col1, diff_col2 = st.columns(2)
            with diff_col1:
                st.markdown("<div class='diff-label-before'>Before</div>", unsafe_allow_html=True)
                st.markdown("<div class='diff-before'>", unsafe_allow_html=True)
                st.code(before or "(not available)", language=finding.get("language") or "text")
                st.markdown("</div>", unsafe_allow_html=True)
            with diff_col2:
                st.markdown("<div class='diff-label-after'>After</div>", unsafe_allow_html=True)
                st.markdown("<div class='diff-after'>", unsafe_allow_html=True)
                st.code(after or "(no fix generated)", language=finding.get("language") or "text")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("**Why This Fix Works**")
            st.write(remediation.get("explanation") or "N/A")

            st.markdown("**Secure Coding Guideline**")
            st.write(remediation.get("guideline_reference") or "N/A")
            retrieved = remediation.get("retrieved_guidance") or []
            if retrieved:
                badge_html = "".join(
                    f"<span class='status-chip' style='background:rgba(192,38,211,0.12); color:#F0ABFC; border:1px solid rgba(192,38,211,0.3); margin-right:0.4rem;'>{_safe(g.get('source', 'Reference'))}</span>"
                    for g in retrieved
                )
                st.markdown(badge_html, unsafe_allow_html=True)

            st.markdown("**Prevention Tip**")
            st.write(remediation.get("prevention_tip") or "N/A")

            if remediation.get("validation_note"):
                st.caption(remediation["validation_note"])

            st.caption("Copy fixed code")
            st.code(after or "(no fix generated)", language=finding.get("language") or "text")


# ------------------------------------------------------------------------- #
# 3. PR Summary Tab
# ------------------------------------------------------------------------- #
_MERGE_CHIP_STYLE = {
    "blocked": "background:rgba(244,63,94,0.16); color:#FB7185; border:1px solid rgba(244,63,94,0.4);",
    "caution": "background:rgba(245,158,11,0.14); color:#FBBF24; border:1px solid rgba(245,158,11,0.35);",
    "approved": "background:rgba(34,197,94,0.14); color:#4ADE80; border:1px solid rgba(34,197,94,0.35);",
}


def render_pr_summary_tab(deep_result: dict, code: str, file_name: str, state_key: str) -> None:
    _section_desc(
        "A compiled, PR-review-style summary of every agent's output: merge recommendation, "
        "severity breakdown, prioritized fix list, and cross-tool agreement — paste-ready as an "
        "actual GitHub PR comment."
    )

    api_key = st.session_state.gemini_api_key.strip()
    use_llm = _llm_enabled()

    data = generate_pr_summary(
        deep_result,
        code=code,
        file_name=file_name,
        api_key=api_key,
        use_llm=use_llm,
    )

    st.markdown(
        "<div class='pr-download-bar'>"
        "<div><div class='pr-download-title'>📄 PR Summary Report</div>"
        "<div class='pr-download-sub'>Download this PR review as a PDF — merge recommendation, "
        "findings, and remediation roadmap.</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    try:
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        pdf_filename = f"{file_name or 'analysis'}_pr_summary_report.pdf"
        pdf_out_path = os.path.join(temp_dir, pdf_filename)

        pdf_path = generate_pr_pdf(
            deep_result,
            data,
            pdf_out_path,
            project_name=file_name or "Code Analysis Project",
        )

        if pdf_path and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Report",
                    f.read(),
                    file_name=pdf_filename,
                    mime="application/pdf",
                    key=f"download_pr_pdf_{state_key}",
                    type="primary",
                )
        else:
            st.error("Failed to generate PR Summary PDF report. File was not created.")
    except Exception as exc:
        st.error(f"Unable to generate PR Summary PDF report: {exc}")

    merge = data["merge_recommendation"]
    st.markdown(
        f"""
        <div class="premium-card">
            <span class="status-chip" style="{_MERGE_CHIP_STYLE.get(merge['key'], '')}">{_safe(merge['label'])}</span>
            <div style="margin-top:0.7rem; color:#F8FAFC; font-size:0.95rem;">{_safe(merge['description'])}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-chip"><div class="metric-icon">Findings</div><div class="metric-value">{data['total_findings']}</div></div>
            <div class="metric-chip"><div class="metric-icon">Auto-Fixable</div><div class="metric-value" style="color:#4ADE80">{data['auto_fixable']['ratio_pct']}%</div></div>
            <div class="metric-chip"><div class="metric-icon">Tool Agreement</div><div class="metric-value" style="color:#A78BFA">{data['agreement']['agreement_pct']}%</div></div>
            <div class="metric-chip"><div class="metric-icon">Est. Review Time</div><div class="metric-value">{data['review_time_minutes']} min</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    _section_head("Executive Overview")
    st.markdown(f"<div class='premium-card'>{_safe(data['executive_overview'])}</div>", unsafe_allow_html=True)
    st.caption(f"Suggested PR title: `{data['suggested_pr_title']}`")

    gauge_col, donut_col = st.columns([1, 1.2])
    with gauge_col:
        st.markdown("<div class='premium-card scroll-reveal-left' style='text-align:center;'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Mergeability Score</div>", unsafe_allow_html=True)
        render_score_gauge(
            data["mergeability_score"],
            st.session_state.get(f"prev_score_{state_key}"),
            chart_key=f"pr_summary_score_{state_key}",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with donut_col:
        render_category_donut(data["category_breakdown"], chart_key=f"category_donut_{state_key}")

    render_auto_fixable_bar(data["auto_fixable"], chart_key=f"autofix_bar_{state_key}")

    if data["owasp_coverage"]:
        _section_head(f"OWASP Top 10 Coverage ({len(data['owasp_coverage'])})")
        badge_html = "".join(
            f"<span class='status-chip' style='background:rgba(192,38,211,0.12); color:#F0ABFC; "
            f"border:1px solid rgba(192,38,211,0.3); margin-right:0.5rem; margin-bottom:0.4rem;'>{_safe(cat)}</span>"
            for cat in data["owasp_coverage"]
        )
        st.markdown(f"<div class='premium-card'>{badge_html}</div>", unsafe_allow_html=True)

    # Render findings in PR Summary: Severity tabs, NO snippets, NO remediation block
    all_findings = deep_result.get("findings", []) or []
    non_info_count = len([f for f in all_findings if str(f.get("severity") or "").capitalize() in ("Critical", "High", "Medium", "Low")])
    _section_head(f"Findings ({non_info_count})")
    render_severity_tabbed_findings(all_findings, show_owasp=True, show_snippet=False, show_remediation=False)

    if data["agreement_matrix"]:
        _section_head(f"Cross-Tool Agreement ({len(data['agreement_matrix'])})")
        _section_desc("Findings independently confirmed by more than one tool — the highest-confidence signal in this report.")
        for m in data["agreement_matrix"]:
            st.markdown(
                f"<div class='premium-card' style='padding:0.9rem 1.2rem;'>"
                f"<span class='finding-meta'><code>{_safe(m['file'])}:{_safe(m['line'])}</code></span> "
                f"<b>{_safe(m['title'])}</b> — confirmed by {_safe(', '.join(m['tools']))}</div>",
                unsafe_allow_html=True,
            )

    _section_head("What Went Right")
    st.markdown(
        "<div class='premium-card'>" + "".join(f"<div>{_safe(h)}</div>" for h in data["positive_highlights"]) + "</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------- #
# 4. AI Code Assistant Tab
# ------------------------------------------------------------------------- #
def render_assistant_tab(deep_result: dict, findings: list, result: dict, state_key: str) -> None:
    focus_input = st.session_state.pop(f"_scroll_to_assistant_{state_key}", False)

    _section_desc(
        "Ask questions about this scan — findings, severity reasoning, or how to fix something — "
        "grounded in the same data shown in the other tabs."
    )

    messages_key = f"chat_messages_{state_key}"
    loaded_marker_key = f"chat_loaded_record_{state_key}"

    record_id = st.session_state.get("history_record_id") if state_key == "history" else None
    user = auth.current_user()
    user_id = user["username"] if user else None
    persist_chat = record_id is not None and user_id is not None

    if persist_chat and st.session_state.get(loaded_marker_key) != record_id:
        persisted = get_chat_history(record_id, user_id)
        st.session_state[messages_key] = [
            {"role": m["role"], "content": m["message"]} for m in persisted
        ]
        st.session_state[loaded_marker_key] = record_id

    def _on_new_message(role: str, content: str) -> None:
        if persist_chat:
            append_chat_message(record_id, user_id, role, content)

    if not st.session_state.rag_status:
        with st.spinner("Preparing knowledge base for the assistant..."):
            st.session_state.rag_status = st.session_state.rag_pipeline.build_index(
                use_cache=True, api_key=st.session_state.gemini_api_key
            )

    api_key = st.session_state.gemini_api_key.strip()
    use_llm = _llm_enabled()
    pr_data = generate_pr_summary(
        deep_result,
        code=result.get("code", ""),
        file_name=result.get("file_name") or deep_result.get("file"),
        api_key=api_key,
        use_llm=use_llm,
    )

    st.markdown("<div id='assistant-chat-anchor'></div>", unsafe_allow_html=True)
    st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
    try:
        render_conversational_assistant(
            deep_result=deep_result,
            findings=findings,
            pr_summary=pr_data,
            uploaded_code=result.get("code", ""),
            language=result.get("language"),
            api_key=api_key,
            focus_input=focus_input,
            on_new_message=_on_new_message if persist_chat else None,
            empty_state_text="No previous conversation for this analysis." if persist_chat else None,
            state_key=state_key,
        )
    except Exception as exc:
        st.markdown(
            f"<div class='chat-empty-state'>The AI Code Assistant could not be loaded.<br/>"
            f"<span style='color:#FB7185;'>{_safe(exc)}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if focus_input:
        _scroll_to_anchor("assistant-chat-anchor")


# ------------------------------------------------------------------------- #
# Results Dashboard — Sub-tab Router
# ------------------------------------------------------------------------- #
def render_results_dashboard(result: dict, state_key: str = None) -> None:
    if not result:
        st.markdown(
            '<div class="premium-card" style="text-align:center; color:#C3CEEA;">'
            'Submit code above to see analysis results.</div>',
            unsafe_allow_html=True,
        )
        return

    confidence = f"{int((result.get('language_confidence') or 0) * 100)}%"
    lang_display = _safe(result.get("language") or "unknown")
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-chip"><div class="metric-icon">Language</div><div class="metric-value">{lang_display}</div></div>
        <div class="metric-chip"><div class="metric-icon">Confidence</div><div class="metric-value">{confidence}</div></div>
        <div class="metric-chip"><div class="metric-icon">Lines</div><div class="metric-value">{result.get("lines", 0)}</div></div>
        <div class="metric-chip"><div class="metric-icon">Characters</div><div class="metric-value">{result.get("characters", 0)}</div></div>
    </div>""", unsafe_allow_html=True)
    st.caption(result.get("language_reason") or "")

    valid = result.get("valid") and result.get("language") in ("python", "java")
    st.markdown(f'<div class="premium-card"><span class="status-chip status-{"valid" if valid else "invalid"}">Syntax {"Valid" if valid else "Invalid"}</span></div>', unsafe_allow_html=True)
    if not valid:
        st.error(f"**{result.get('error_type', 'Validation Error')}**: {result.get('message', 'Unable to validate code.')} (Line: {result.get('error_line') or 'N/A'})")

    if result.get("code", "").strip():
        with st.expander("Code Preview"):
            st.code(result["code"], language=result["language"] if result["language"] in ("python", "java") else "text", line_numbers=True)

    if not (valid and state_key):
        return

    if st.button("Run Deep Analysis", key=f"run_deep_{state_key}", type="primary"):
        st.session_state[f"deep_{state_key}"] = None
        run_deep_analysis(result, state_key)

    deep_result = st.session_state.get(f"deep_{state_key}")
    if not deep_result:
        return

    _section_head("Pipeline Status")
    render_pipeline_strip(deep_result)

    if deep_result.get("error"):
        st.warning(deep_result["error"])

    findings = deep_result.get("findings", []) or []

    sub_tab_key = f"active_subtab_{state_key}"
    st.session_state.setdefault(sub_tab_key, "code_security")

    # History view shows only the 4 core tabs (Chat Assistant removed); Code submission keeps Chat Assistant
    if state_key == "history":
        sub_tabs = [
            ("code_security", "Code & Security Analysis"),
            ("remediation", "Remediation"),
            ("pr_summary", "PR Summary"),
            ("dev_portal", "Developer Portal"),
        ]
        # Reset if previously pointing to assistant
        if st.session_state.get(sub_tab_key) == "assistant":
            st.session_state[sub_tab_key] = "code_security"
    else:
        sub_tabs = [
            ("code_security", "Code & Security Analysis"),
            ("remediation", "Remediation"),
            ("pr_summary", "PR Summary"),
            ("dev_portal", "Developer Portal"),
            ("assistant", "AI Code Assistant"),
        ]

    nav_cols = st.columns(len(sub_tabs))
    for col, (tab_id, label) in zip(nav_cols, sub_tabs):
        is_active = st.session_state[sub_tab_key] == tab_id
        if col.button(
            label,
            key=f"subtab_btn_{state_key}_{tab_id}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state[sub_tab_key] = tab_id
            if tab_id == "assistant":
                st.session_state[f"_scroll_to_assistant_{state_key}"] = True
            st.rerun()

    active_tab = st.session_state[sub_tab_key]

    st.markdown('<div class="tab-fade">', unsafe_allow_html=True)
    if active_tab == "code_security":
        render_code_security_tab(findings, state_key=state_key)
    elif active_tab == "remediation":
        render_remediation_tab(findings)
    elif active_tab == "pr_summary":
        render_pr_summary_tab(deep_result, result.get("code", ""), result.get("file_name") or deep_result.get("file"), state_key)
    elif active_tab == "dev_portal":
        render_findings_display(deep_result, result, state_key)
    elif active_tab == "assistant" and state_key != "history":
        render_assistant_tab(deep_result, findings, result, state_key)
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------- #
# 5. History View
# ------------------------------------------------------------------------- #
def render_history_view() -> None:
    st.markdown(
        '<div class="hero-title scroll-reveal" style="font-size:1.6rem;">Analysis History</div>'
        '<div class="hero-subtitle scroll-reveal" style="transition-delay:0.1s;">Previously completed scans for your account — loaded without re-running any agents</div>',
        unsafe_allow_html=True,
    )

    user = auth.current_user()
    if not user:
        return
    user_id = user["username"]
    records = list_history_for_user(user_id)

    if not records:
        st.markdown(
            '<div class="premium-card" style="text-align:center; color:#C3CEEA;">'
            'No past analyses yet — results are saved here automatically after you run Deep Analysis.</div>',
            unsafe_allow_html=True,
        )
        return

    for rec in records:
        score = rec.get("security_score", 100)
        score_color = "#4ADE80" if score >= 80 else ("#F59E0B" if score >= 50 else "#F43F5E")
        st.markdown(f"""
        <div class="premium-card card-flat">
            <span class="finding-meta"><code>{_safe(rec.get('language'))}</code> &middot; {_safe(rec.get('analyzed_at'))}</span>
            <div class="finding-title">{_safe(rec.get('file_name'))}</div>
            <div style="margin-top:0.4rem; color:{score_color}; font-weight:700; font-family:'JetBrains Mono', monospace;">Score: {score}/100</div>
            <div style="margin-top:0.3rem; color:#C3CEEA; font-size:0.85rem;">
                {rec.get('total_findings', 0)} findings &middot;
                Critical {rec.get('critical_count', 0)} &middot; High {rec.get('high_count', 0)} &middot;
                Medium {rec.get('medium_count', 0)} &middot; Low {rec.get('low_count', 0)}
            </div>
        </div>""", unsafe_allow_html=True)

        confirm_key = f"confirm_delete_{rec['id']}"

        col_view, col_pdf, col_delete = st.columns([1, 1, 1])
        with col_view:
            if st.button("View Analysis", key=f"view_hist_{rec['id']}", use_container_width=True):
                full = get_history_record(rec["id"], user_id)
                if full:
                    payload = full.get("analysis_payload", {})
                    st.session_state["result_history"] = payload.get("result")
                    st.session_state["deep_history"] = payload.get("deep_result")
                    st.session_state["active_subtab_history"] = "code_security"
                    st.session_state["prev_score_history"] = None
                    st.session_state["history_record_id"] = rec["id"]
                    # Store which record we should scroll to — using the
                    # record's own id (not a single shared flag/id) means
                    # every card's button targets a genuinely distinct
                    # anchor, so clicking a different card after the first
                    # one still navigates correctly.
                    st.session_state["_scroll_to_analysis_id"] = rec["id"]
                    st.rerun()
                else:
                    st.error("Could not load this record.")
        with col_pdf:
            report_path = rec.get("report_path")
            if report_path and report_belongs_to_user(report_path, user_id) and os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    st.download_button(
                        "Download PDF",
                        f.read(),
                        file_name=os.path.basename(os.path.dirname(report_path)) + "_report.pdf",
                        mime="application/pdf",
                        key=f"dl_hist_{rec['id']}",
                        use_container_width=True,
                    )
            else:
                st.caption("PDF not available for this record.")
        with col_delete:
            if st.button("Delete", key=f"delete_hist_{rec['id']}", use_container_width=True):
                st.session_state[confirm_key] = True

        if st.session_state.get(confirm_key):
            st.markdown(
                "<div class='history-confirm-banner'>Are you sure you want to delete this history? "
                "This only removes this one entry and cannot be undone.</div>",
                unsafe_allow_html=True,
            )
            confirm_col1, confirm_col2 = st.columns([1, 1])
            with confirm_col1:
                if st.button("Yes, delete", key=f"confirm_yes_{rec['id']}", type="primary", use_container_width=True):
                    deleted = delete_history_record(rec["id"], user_id)
                    st.session_state.pop(confirm_key, None)
                    if deleted:
                        if st.session_state.get("history_record_id") == rec["id"]:
                            # The deleted record was the one currently shown
                            # below — clear all of its state, not just the
                            # list card, so the loaded analysis panel (and
                            # its own View/PDF buttons) disappears entirely
                            # instead of lingering with stale data.
                            st.session_state.pop("history_record_id", None)
                            st.session_state.pop("chat_messages_history", None)
                            st.session_state.pop("chat_loaded_record_history", None)
                            st.session_state.pop("result_history", None)
                            st.session_state.pop("deep_history", None)
                            st.session_state.pop("active_subtab_history", None)
                            st.session_state.pop("prev_score_history", None)
                            st.session_state.pop("_scroll_to_analysis_id", None)
                        st.toast("History entry deleted.")
                        st.rerun()
                    else:
                        st.error("Could not delete this history entry.")
            with confirm_col2:
                if st.button("Cancel", key=f"confirm_no_{rec['id']}", use_container_width=True):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()

    if st.session_state.get("result_history"):
        loaded_id = st.session_state.get("history_record_id")
        # Unique per-record anchor — never a single shared id — so the
        # scroll target is always the right one, no matter which card's
        # "View Analysis" was clicked or how many times in a row.
        anchor_id = f"analysis-section-{loaded_id}"
        st.markdown('<hr style="border:none; border-top:1px solid rgba(167,139,250,0.16); margin:2rem 0;">', unsafe_allow_html=True)
        st.markdown(f'<div id="{anchor_id}" class="analysis-anchor"></div>', unsafe_allow_html=True)
        _section_head(f"Loaded: {st.session_state['result_history'].get('file_name') or 'Selected analysis'}")
        render_results_dashboard(st.session_state.result_history, "history")

        # Smoothly auto-scroll down to this specific record's anchor —
        # only if the pending scroll request still matches what's loaded.
        pending_id = st.session_state.pop("_scroll_to_analysis_id", None)
        if pending_id is not None and pending_id == loaded_id:
            _scroll_to_anchor(anchor_id)


# ==============================================================================
# FIXED / STICKY TOP NAVIGATION BAR
# ==============================================================================
# Sentinel sits right where the navbar would normally be. As long as it's
# visible, the navbar is still "in place" at the top of the page; once it
# scrolls out of view, the navbar has become the stuck/sticky one — that's
# the moment we play the fade+translateY entrance. Scrolling back up makes
# the sentinel reappear, which plays the same animation in reverse feel.
st.markdown('<div id="nav-sentinel"></div>', unsafe_allow_html=True)

with st.container(key="sticky_nav_container"):
    nav_title_col, nav_user_col = st.columns([1.6, 1.4], vertical_alignment="center")
    with nav_title_col:
        st.markdown('<div class="sticky-nav-brand">Smart Code Inspection Platform</div>', unsafe_allow_html=True)
    with nav_user_col:
        auth.render_user_bar()

_inject_navbar_scroll_animation()


# ==============================================================================
# HERO & SECTION NAVIGATION
# ==============================================================================
st.markdown(
    "<div class='hero-wrap scroll-stagger'><div class='hero-title'>Smart Code Inspection Platform</div>"
    "<div class='hero-subtitle'>Vulnerability Detection System &middot; multi-agent static analysis, grounded severity scoring, and LLM-powered remediation</div>"
    "<div class='badge-row'>"
    "<span class='hero-badge'>Python</span>"
    "<span class='hero-badge'>Java</span>"
    "<span class='hero-badge'>OWASP Top 10</span>"
    "<span class='hero-badge'>Gemini-Powered</span>"
    "<span class='hero-badge'>LangGraph Orchestration</span>"
    "</div></div>",
    unsafe_allow_html=True,
)

_, nav_col, _ = st.columns([1, 1, 1])
with nav_col:
    with st.container(key="hero_actions_container"):
        c1, c2 = st.columns(2)
        for col, label, view in [(c1, "Code Submission", "code_submission"), (c2, "History", "history")]:
            if col.button(label, type="primary" if st.session_state.active_view == view else "secondary", use_container_width=True):
                st.session_state.active_view = view
                st.rerun()

st.markdown('<hr style="border:none; border-top:1px solid rgba(167,139,250,0.16); margin:2rem 0;">', unsafe_allow_html=True)


# ==============================================================================
# MAIN VIEWS (CODE SUBMISSION / HISTORY)
# ==============================================================================
if st.session_state.active_view == "code_submission":
    st.markdown('<div class="hero-title scroll-reveal" style="font-size:1.6rem;">Code Submission Module</div><div class="hero-subtitle scroll-reveal" style="transition-delay:0.1s;">Submit and analyze Python or Java source code</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Paste Code", "Upload File", "GitHub Repository", "Raw URL"])

    with t1:
        code = st.text_area("Paste code", height=250, placeholder='print("Hello") or public class Main {}', label_visibility="collapsed")
        if st.button("Analyze Code", key="run_paste", type="primary"):
            st.session_state.result_paste = process_pasted_code(code)
            st.session_state.deep_paste = None
        render_results_dashboard(st.session_state.result_paste, "paste")

    with t2:
        upload = st.file_uploader("Upload a Python or Java file", type=["py", "java"])
        if st.button("Analyze Code", key="run_upload", type="primary"):
            if upload is None:
                st.error("Please choose a file before analyzing.")
            else:
                st.session_state.result_upload = process_uploaded_file(upload.read().decode("utf-8", errors="ignore"), file_name=upload.name)
                st.session_state.deep_upload = None
        render_results_dashboard(st.session_state.result_upload, "upload")

    with t3:
        repo_url = st.text_input("GitHub URL", placeholder="https://github.com/user/repo")
        if st.button("Fetch Repository Files", type="primary") and repo_url.strip():
            with st.spinner("Cloning and detecting languages..."):
                repo_result = clone_and_extract(repo_url)
            if repo_result["success"]:
                st.session_state.github_files_cached = repo_result["files"]
                st.toast(f"Found {len(repo_result['files'])} supported file(s).")
            else:
                st.session_state.github_files_cached = None
                st.error(repo_result["error"])

        if st.session_state.github_files_cached:
            labels = [
                f"{item['file_name']} [{item['language']} - {int(item.get('language_confidence', 0) * 100)}%]"
                for item in st.session_state.github_files_cached
            ]
            selected = st.selectbox("Select a detected file", labels)
            if st.button("Analyze Selected File", type="secondary"):
                index = labels.index(selected)
                item = st.session_state.github_files_cached[index]
                st.session_state.result_github = process_uploaded_file(item["code"], file_name=item["file_name"])
                st.session_state.deep_github = None
        render_results_dashboard(st.session_state.result_github, "github")

    with t4:
        raw_url = st.text_input("Raw URL", placeholder="https://raw.githubusercontent.com/...")
        if st.button("Analyze Code", key="run_url", type="primary") and raw_url.strip():
            with st.spinner("Downloading and detecting language..."):
                url_result = download_file(raw_url)
            if url_result["success"]:
                st.session_state.result_url = process_url_file(url_result["code"], file_name=url_result["file_name"])
                st.session_state.deep_url = None
            else:
                st.error(url_result["error"])
        render_results_dashboard(st.session_state.result_url, "url")
else:
    render_history_view()

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown(
    """
    <div class="app-footer">
        <b>Smart Code Inspection Platform</b> · Vulnerability Detection System
        <div class="foot-badges">
            <span class="foot-badge">Multi-Agent Pipeline</span>
            <span class="foot-badge">LangGraph</span>
            <span class="foot-badge">Gemini</span>
            <span class="foot-badge">Bandit · Semgrep · Pylint · Radon · PMD</span>
            <span class="foot-badge">OWASP Top 10 · 2025</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

