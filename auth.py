from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional
from urllib.parse import urlencode

import requests
import streamlit as st

# ==============================================================================
# CONFIG
# ==============================================================================
USERS_DB_PATH = os.path.join(os.path.dirname(__file__), "users_db.json")
PBKDF2_ITERATIONS = 260_000

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"

# Max age (seconds) for a GitHub OAuth "state" token before it's rejected.
OAUTH_STATE_MAX_AGE = 600  # 10 minutes


# ==============================================================================
# SHARED CSS
# ==============================================================================
def _inject_auth_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 0.4rem !important; }
        .stButton>button p, [data-testid="stPopover"] button p {
            white-space: nowrap !important;
        }
        [data-testid="stPopover"] button {
            width: auto !important;
            min-width: max-content !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        /* Primary buttons */
        .stButton>button[kind="primary"],
        div[data-testid="stForm"] button[kind="primary"],
        .stForm button[type="submit"] {
            background: linear-gradient(105deg, #7C3AED, #C026D3) !important;
            border: none !important;
            color: white !important;
            border-radius: 999px !important;
            font-weight: 600 !important;
            box-shadow: 0 6px 22px rgba(124,58,237,0.45) !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button[kind="primary"]:hover,
        div[data-testid="stForm"] button[kind="primary"]:hover {
            box-shadow:
                0 12px 30px rgba(124,58,237,0.45),
                0 0 18px rgba(192,38,211,0.18) !important;
            filter: brightness(1.08);
            transform: translateY(-2px) scale(1.01);
        }

        /* Secondary / outline buttons */
        .stButton>button[kind="secondary"] {
            background: rgba(35, 20, 66, 0.45) !important;
            color: #C4B5FD !important;
            border: 1px solid rgba(167,139,250,0.35) !important;
            border-radius: 999px !important;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease, color 0.25s ease !important;
        }
        .stButton>button[kind="secondary"]:hover {
            border-color: rgba(192,38,211,0.5) !important;
            color: #F8FAFC !important;
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 10px 24px rgba(124,58,237,0.22);
        }

        /* Link buttons (e.g. "Continue with GitHub") get the same gentle lift */
        .stLinkButton>a {
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
        }
        .stLinkButton>a:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 10px 24px rgba(124,58,237,0.18);
        }

        .auth-title {
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 1.2rem;
        }

        .auth-switch {
            text-align: center;
            margin-top: 1.1rem;
            color: #C3CEEA;
            font-size: 0.9rem;
        }

        /* Entrance animation */
        @keyframes authFadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .auth-animate {
            animation: authFadeUp 0.5s ease both;
        }

        /* Staggered entrance for individual auth-form elements */
        @keyframes fadeUpSmall {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .auth-stagger {
            opacity: 0;
            animation: fadeUpSmall 0.5s ease forwards;
        }
        .auth-stagger-1 { animation-delay: 0.10s; }
        .auth-stagger-2 { animation-delay: 0.18s; }
        .auth-stagger-3 { animation-delay: 0.26s; }
        .auth-stagger-4 { animation-delay: 0.34s; }
        .auth-stagger-5 { animation-delay: 0.42s; }
        .auth-stagger-6 { animation-delay: 0.50s; }
        .auth-stagger-7 { animation-delay: 0.58s; }

        /* Gentle floating animation for the primary shield/logo */
        @keyframes floatShield {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-7px); }
        }
        .floating-shield {
            display: inline-block;
            animation: floatShield 3s ease-in-out infinite;
        }

        /* Fade/slide when switching between Login and Sign Up */
        @keyframes formSwitch {
            from { opacity: 0; transform: translateX(10px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        .auth-form-content {
            animation: formSwitch 0.35s ease both;
        }

        /* Input focus glow */
        div[data-testid="stTextInput"] input {
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: rgba(167, 139, 250, 0.65) !important;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.18) !important;
        }

        /* ---------- Split layout ---------- */
        div[data-testid="stHorizontalBlock"]:has(.split-left-marker) {
            border: 1.5px solid rgba(192, 132, 252, 0.55);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 14px 46px rgba(0, 0, 0, 0.32), 0 0 0 1px rgba(124,58,237,0.12);
            align-items: stretch;
            animation: authFadeUp 0.6s ease both;
        }
        div[data-testid="stHorizontalBlock"]:has(.split-left-marker) > div {
            display: flex;
        }
        div[data-testid="stHorizontalBlock"]:has(.split-left-marker) > div > div {
            width: 100%;
        }

        .split-left {
            background:
                radial-gradient(circle at 30% 15%, rgba(124,58,237,0.5) 0%, transparent 55%),
                linear-gradient(165deg, #2B1854 0%, #150B2E 75%);
            padding: 2.2rem 2rem;
            height: 100%;
            min-height: 420px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 2rem;
            animation: authSlideInLeft 0.6s ease both;
        }
        .split-quote {
            font-size: 1.5rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.28;
        }
        .split-right {
            background: rgba(20, 12, 38, 0.55);
            padding: 1.6rem 2rem 1.8rem;
            height: 100%;
            animation: authSlideInRight 0.6s ease both;
            animation-delay: 0.08s;
        }

        .auth-tabbar-divider {
            border: none;
            border-top: 1px solid rgba(167,139,250,0.18);
            margin: 0 0 1.2rem;
        }

        @keyframes authSlideInLeft {
            from { opacity: 0; transform: translateX(-24px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes authSlideInRight {
            from { opacity: 0; transform: translateX(24px); }
            to   { opacity: 1; transform: translateX(0); }
        }

        @media (max-width: 900px) {
            div[data-testid="stHorizontalBlock"]:has(.split-left-marker) {
                flex-direction: column;
            }
            .split-left { min-height: 220px; }
        }

        /* ---------- Welcome page box ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.welcome-box-marker) {
            border: 1.5px solid rgba(192, 132, 252, 0.5);
            border-radius: 24px;
            box-shadow: 0 14px 46px rgba(0, 0, 0, 0.32), 0 0 0 1px rgba(124,58,237,0.1);
            background: rgba(20, 12, 38, 0.35);
            max-width: 900px;
            margin: 1.5rem auto 2rem;
            padding: 0.4rem 1.6rem 1.6rem;
            animation: authFadeUp 0.6s ease both;
        }
        .welcome-logo-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 0 0.2rem;
        }

        /* ---------- Ambient animated background (auth page only) ---------- */
        @keyframes gradientFloat {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50%      { transform: translate(20px, -15px) scale(1.05); }
        }
        @keyframes gradientFloatAlt {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50%      { transform: translate(-24px, 18px) scale(1.07); }
        }
        .auth-bg-blob {
            position: fixed;
            border-radius: 50%;
            filter: blur(90px);
            opacity: 0.25;
            pointer-events: none;
            z-index: 0;
        }
        .auth-bg-blob-1 {
            top: -10%;
            left: -8%;
            width: 420px;
            height: 420px;
            background: radial-gradient(circle, #7C3AED 0%, transparent 70%);
            animation: gradientFloat 22s ease-in-out infinite;
        }
        .auth-bg-blob-2 {
            bottom: -12%;
            right: -6%;
            width: 480px;
            height: 480px;
            background: radial-gradient(circle, #C026D3 0%, transparent 70%);
            animation: gradientFloatAlt 26s ease-in-out infinite;
        }
        .auth-bg-blob-3 {
            top: 40%;
            left: 45%;
            width: 360px;
            height: 360px;
            background: radial-gradient(circle, #A78BFA 0%, transparent 70%);
            animation: gradientFloat 30s ease-in-out infinite reverse;
        }
        /* Keep the actual auth UI above the decorative blobs */
        div[data-testid="stHorizontalBlock"]:has(.split-left-marker),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.welcome-box-marker) {
            position: relative;
            z-index: 1;
        }

        /* ---------- Auth form staggered entrance (tab bar / form-switch wrapper / GitHub button) ---------- */
        div[data-testid="stHorizontalBlock"]:has(.auth-tabbar-marker) {
            opacity: 0;
            animation: fadeUpSmall 0.5s ease forwards;
            animation-delay: 0.10s;
        }
        div[data-testid="stForm"] {
            opacity: 0;
            animation: fadeUpSmall 0.5s ease forwards;
            animation-delay: 0.26s;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.auth-form-content-marker) {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
            padding: 0 !important;
            margin: 0 !important;
            animation: formSwitch 0.35s ease both;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.auth-github-marker) {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
            padding: 0 !important;
            margin: 0 !important;
            opacity: 0;
            animation: fadeUpSmall 0.5s ease forwards;
            animation-delay: 0.42s;
        }

        /* ---------- How-it-works card hover/entrance ---------- */
        .how-card {
            transition:
                transform 0.3s ease,
                border-color 0.3s ease,
                box-shadow 0.3s ease;
        }
        .how-card:hover {
            transform: translateY(-6px);
            border-color: rgba(167,139,250,0.5);
            box-shadow: 0 12px 30px rgba(124,58,237,0.18);
        }

        /* ---------- Respect reduced-motion preference ---------- */
        @media (prefers-reduced-motion: reduce) {
            .auth-animate,
            .split-left,
            .split-right,
            .auth-stagger,
            .auth-stagger-1,
            .auth-stagger-2,
            .auth-stagger-3,
            .auth-stagger-4,
            .auth-stagger-5,
            .auth-stagger-6,
            .auth-stagger-7,
            .floating-shield,
            .auth-form-content,
            .auth-bg-blob-1,
            .auth-bg-blob-2,
            .auth-bg-blob-3,
            div[data-testid="stHorizontalBlock"]:has(.split-left-marker),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.welcome-box-marker) {
                animation: none !important;
                opacity: 1 !important;
                transform: none !important;
            }
            .stButton>button[kind="primary"]:hover,
            div[data-testid="stForm"] button[kind="primary"]:hover,
            .stButton>button[kind="secondary"]:hover,
            .stLinkButton>a:hover,
            .how-card:hover {
                transform: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_bg_blobs() -> None:
    """Decorative, non-interactive animated gradient blobs behind the auth UI."""
    st.markdown(
        """
        <div class="auth-bg-blob auth-bg-blob-1"></div>
        <div class="auth-bg-blob auth-bg-blob-2"></div>
        <div class="auth-bg-blob auth-bg-blob-3"></div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SAFE SECRETS ACCESS
# ==============================================================================
def _get_secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# ==============================================================================
# LOCAL USER STORE
# ==============================================================================
def _load_users() -> dict:
    if not os.path.exists(USERS_DB_PATH):
        return {}
    try:
        with open(USERS_DB_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict) -> None:
    with open(USERS_DB_PATH, "w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2)


def _hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    _, computed = _hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(computed, hash_hex)


def create_user(username: str, email: str, full_name: str, password: str) -> tuple[bool, Optional[str]]:
    username = username.strip().lower()
    email = email.strip().lower()
    if not username or not email or not full_name.strip() or not password:
        return False, "All fields are required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    users = _load_users()
    if username in users:
        return False, "That username is already taken."
    if any(u.get("email") == email for u in users.values()):
        return False, "An account with that email already exists."

    salt_hex, hash_hex = _hash_password(password)
    users[username] = {
        "username": username,
        "email": email,
        "full_name": full_name.strip(),
        "salt": salt_hex,
        "password_hash": hash_hex,
        "auth_method": "password",
        "avatar_url": None,
        "created_at": time.time(),
    }
    _save_users(users)
    return True, None


def authenticate_user(identifier: str, password: str) -> Optional[dict]:
    identifier = identifier.strip().lower()
    users = _load_users()
    user = users.get(identifier)
    if not user:
        user = next((u for u in users.values() if u.get("email") == identifier), None)
    if not user or user.get("auth_method") != "password":
        return None
    if _verify_password(password, user["salt"], user["password_hash"]):
        return user
    return None


def _find_or_create_github_user(profile: dict, email: Optional[str]) -> dict:
    login = profile.get("login", "").strip().lower()
    users = _load_users()
    key = f"gh_{login}"
    if key not in users:
        users[key] = {
            "username": key,
            "email": (email or "").lower(),
            "full_name": profile.get("name") or profile.get("login"),
            "auth_method": "github",
            "avatar_url": profile.get("avatar_url"),
            "github_login": login,
            "created_at": time.time(),
        }
        _save_users(users)
    else:
        users[key]["avatar_url"] = profile.get("avatar_url")
        users[key]["full_name"] = profile.get("name") or profile.get("login")
        _save_users(users)
    return users[key]


# ==============================================================================
# SESSION HELPERS
# ==============================================================================
def _init_auth_state() -> None:
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("auth_screen", "welcome")  # welcome | auth
    st.session_state.setdefault("auth_form", "login")        # login | signup (active pseudo-tab)


def is_authenticated() -> bool:
    return st.session_state.get("auth_user") is not None


def current_user() -> Optional[dict]:
    return st.session_state.get("auth_user")


def log_out() -> None:
    st.session_state["auth_user"] = None
    st.session_state["auth_screen"] = "welcome"


# ==============================================================================
# GITHUB OAUTH
# ==============================================================================
def _github_configured() -> bool:
    client_id = _get_secret("GITHUB_CLIENT_ID")
    redirect_uri = _get_secret("GITHUB_REDIRECT_URI")
    return bool(client_id and redirect_uri)


def _state_signing_key() -> bytes:
    # Any stable server-side secret works here. Reusing the GitHub client
    # secret avoids needing a separate config value, but you can swap this
    # for a dedicated APP_SECRET_KEY in st.secrets if you prefer.
    key = _get_secret("GITHUB_CLIENT_SECRET", "") or "fallback-key-change-me"
    return key.encode("utf-8")


def _make_oauth_state() -> str:
    """
    Build a stateless, self-verifying 'state' token: nonce + timestamp + HMAC
    signature. This avoids relying on st.session_state surviving the full
    browser redirect round-trip to GitHub and back (which Streamlit Cloud
    does not always preserve), since the token verifies itself on return.
    """
    nonce = secrets.token_urlsafe(16)
    timestamp = str(int(time.time()))
    payload = f"{nonce}.{timestamp}"
    signature = hmac.new(_state_signing_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def _verify_oauth_state(state: Optional[str]) -> bool:
    if not state or state.count(".") != 2:
        return False
    nonce, timestamp, signature = state.split(".")
    payload = f"{nonce}.{timestamp}"
    expected_signature = hmac.new(_state_signing_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        return False
    try:
        issued_at = int(timestamp)
    except ValueError:
        return False
    if time.time() - issued_at > OAUTH_STATE_MAX_AGE:
        return False
    return True


def _github_authorize_url() -> str:
    state = _make_oauth_state()
    params = {
        "client_id": _get_secret("GITHUB_CLIENT_ID", ""),
        "redirect_uri": _get_secret("GITHUB_REDIRECT_URI", ""),
        "scope": "read:user user:email",
        "state": state,
        "allow_signup": "true",
    }
    return f"{GITHUB_AUTHORIZE_URL}?{urlencode(params)}"


def _exchange_code_for_token(code: str) -> Optional[str]:
    try:
        resp = requests.post(
            GITHUB_TOKEN_URL,
            headers={"Accept": "application/json"},
            data={
                "client_id": _get_secret("GITHUB_CLIENT_ID", ""),
                "client_secret": _get_secret("GITHUB_CLIENT_SECRET", ""),
                "code": code,
                "redirect_uri": _get_secret("GITHUB_REDIRECT_URI", ""),
            },
            timeout=10,
        )
        data = resp.json()
        return data.get("access_token")
    except (requests.RequestException, ValueError):
        return None


def _fetch_github_profile(access_token: str) -> Optional[dict]:
    try:
        resp = requests.get(
            GITHUB_USER_URL,
            headers={"Authorization": f"token {access_token}", "Accept": "application/vnd.github+json"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        return resp.json()
    except requests.RequestException:
        return None


def _fetch_github_primary_email(access_token: str) -> Optional[str]:
    try:
        resp = requests.get(
            GITHUB_EMAILS_URL,
            headers={"Authorization": f"token {access_token}", "Accept": "application/vnd.github+json"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        emails = resp.json()
        primary = next((e["email"] for e in emails if e.get("primary")), None)
        return primary or (emails[0]["email"] if emails else None)
    except (requests.RequestException, KeyError, IndexError):
        return None


def handle_oauth_callback() -> None:
    query = st.query_params
    code = query.get("code")
    returned_state = query.get("state")
    if not code:
        return

    if not _verify_oauth_state(returned_state):
        st.query_params.clear()
        st.error("GitHub sign-in could not be verified (state mismatch). Please try again.")
        return

    with st.spinner("Completing GitHub sign-in..."):
        token = _exchange_code_for_token(code)
        if not token:
            st.query_params.clear()
            st.error("GitHub sign-in failed while exchanging the authorization code.")
            return
        profile = _fetch_github_profile(token)
        if not profile:
            st.query_params.clear()
            st.error("GitHub sign-in failed while fetching your profile.")
            return
        email = profile.get("email") or _fetch_github_primary_email(token)
        user = _find_or_create_github_user(profile, email)

    st.session_state["auth_user"] = user
    st.query_params.clear()
    st.rerun()


# ==============================================================================
# UI — WELCOME PAGE
# ==============================================================================
def render_welcome_page() -> None:
    _inject_auth_css()
    _render_bg_blobs()

    box = st.container(border=True)
    with box:
        st.markdown('<div class="welcome-box-marker"></div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="welcome-logo-row">
                <span style="font-size:1.4rem;">🛡️</span>
                <span style="font-size:1.05rem; font-weight:600; color:#F8FAFC; white-space:nowrap;">Smart Code Inspection Platform</span>
            </div>
            <hr style="border:none; border-top:1px solid rgba(167,139,250,0.18); margin:0.4rem 0 0;">
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 1rem 0;">
                <div class="floating-shield" style="font-size:2.2rem; margin-bottom:0.1rem; line-height:1;">🛡️</div>
                <div class="hero-title auth-stagger auth-stagger-1" style="font-size:2rem; white-space:nowrap; line-height:1.15;">SMART CODE INSPECTION PLATFORM</div>
                <div class="hero-subtitle auth-stagger auth-stagger-2" style="margin-top:0.3rem;">AI-Powered Vulnerability Detection System</div>
                <div class="auth-stagger auth-stagger-3" style="color:#C4B5FD; font-weight:600; letter-spacing:0.05em; margin-top:0.5rem; font-size:0.95rem;">
                    Analyze &bull; Detect &bull; Explain &bull; Remediate &bull; Secure
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Get Started →", key="hero_getstarted", type="primary", use_container_width=True):
                    st.session_state["auth_screen"] = "auth"
                    st.session_state["auth_form"] = "signup"
                    st.rerun()
            with c2:
                if st.button("Sign In →", key="hero_signin", use_container_width=True):
                    st.session_state["auth_screen"] = "auth"
                    st.session_state["auth_form"] = "login"
                    st.rerun()

        st.markdown(
            """
            <div class="badge-row" style="margin-top:0.7rem;">
                <span class="hero-badge">🧠 AI Analysis</span>
                <span class="hero-badge">💻 Multi-Language</span>
                <span class="hero-badge">🛡️ OWASP Top 10</span>
                <span class="hero-badge">📄 Detailed Reports</span>
                <span class="hero-badge">📕 PDF Export</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="max-width:720px; margin:2.2rem auto 0; text-align:center;">
                <div style="font-size:1.05rem; font-weight:650; color:#F1F5F9; margin-bottom:1rem;">How it works</div>
                <div style="display:flex; gap:1rem; flex-wrap:wrap; justify-content:center;">
                    <div class="how-card auth-stagger auth-stagger-1" style="flex:1; min-width:160px; background:rgba(35,20,66,0.5); border:1px solid rgba(167,139,250,0.18); border-radius:12px; padding:1rem 0.9rem;">
                        <div style="font-size:1.3rem; margin-bottom:0.35rem;">1</div>
                        <div style="font-weight:600; color:#E2E8F0; font-size:0.9rem;">Submit code</div>
                        <div style="color:#A5B4CF; font-size:0.8rem; margin-top:0.25rem;">Paste, upload, or pull from GitHub</div>
                    </div>
                    <div class="how-card auth-stagger auth-stagger-2" style="flex:1; min-width:160px; background:rgba(35,20,66,0.5); border:1px solid rgba(167,139,250,0.18); border-radius:12px; padding:1rem 0.9rem;">
                        <div style="font-size:1.3rem; margin-bottom:0.35rem;">2</div>
                        <div style="font-weight:600; color:#E2E8F0; font-size:0.9rem;">AI analyzes</div>
                        <div style="color:#A5B4CF; font-size:0.8rem; margin-top:0.25rem;">Multi-agent scan for security & quality</div>
                    </div>
                    <div class="how-card auth-stagger auth-stagger-3" style="flex:1; min-width:160px; background:rgba(35,20,66,0.5); border:1px solid rgba(167,139,250,0.18); border-radius:12px; padding:1rem 0.9rem;">
                        <div style="font-size:1.3rem; margin-bottom:0.35rem;">3</div>
                        <div style="font-weight:600; color:#E2E8F0; font-size:0.9rem;">Get report</div>
                        <div style="color:#A5B4CF; font-size:0.8rem; margin-top:0.25rem;">Findings, fixes, and PDF export</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="text-align:center; color:#7C7195; font-size:0.78rem; margin-top:2.8rem; padding-top:1.2rem; border-top:1px solid rgba(167,139,250,0.14);">
                Smart Code Inspection Platform · Privacy · Terms
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# UI — LOGIN / SIGNUP FORM BODIES
# ==============================================================================
def _github_continue_button() -> None:
    if not _github_configured():
        st.markdown(
            "<div style='text-align:center; color:#7C7195; font-size:0.84rem; margin:0.4rem 0 0.2rem;'>"
            "GitHub sign-in is coming soon.</div>",
            unsafe_allow_html=True,
        )
        return
    st.link_button("Continue with GitHub", _github_authorize_url(), use_container_width=True)


def _render_login_form() -> None:
    st.markdown('<div class="auth-title">Login</div>', unsafe_allow_html=True)

    with st.form("login_form", border=False):
        identifier = st.text_input("Email", placeholder="Enter email or username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)

    if submitted:
        user = authenticate_user(identifier, password)
        if user:
            st.session_state["auth_user"] = user
            st.rerun()
        else:
            st.error("Incorrect email/username or password.")

    st.markdown(
        "<div style='text-align:center; color:rgba(255,255,255,0.35); margin:1rem 0 0.55rem; font-size:0.82rem;'>── OR ──</div>",
        unsafe_allow_html=True,
    )
    _github_continue_button()

    st.markdown(
        "<div class='auth-switch'>Don't have an account?</div>",
        unsafe_allow_html=True,
    )
    if st.button("Register", use_container_width=True, key="switch_to_signup"):
        st.session_state["auth_form"] = "signup"
        st.rerun()


def _render_signup_form() -> None:
    st.markdown('<div class="auth-title">Register</div>', unsafe_allow_html=True)

    with st.form("signup_form", border=False):
        full_name = st.text_input("Full name", placeholder="Enter full name")
        username = st.text_input("Username", placeholder="Choose username")
        email = st.text_input("Email", placeholder="Enter email")
        password = st.text_input("Password", type="password", placeholder="Create password (min 8 characters)")
        confirm = st.text_input("Confirm password", type="password", placeholder="Confirm password")
        submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

    if submitted:
        if password != confirm:
            st.error("Passwords do not match.")
        else:
            ok, err = create_user(username, email, full_name, password)
            if ok:
                st.success("Account created — you can log in now.")
                st.session_state["auth_form"] = "login"
                st.rerun()
            else:
                st.error(err)

    st.markdown(
        "<div style='text-align:center; color:rgba(255,255,255,0.35); margin:1rem 0 0.55rem; font-size:0.82rem;'>── OR ──</div>",
        unsafe_allow_html=True,
    )
    _github_continue_button()

    st.markdown(
        "<div class='auth-switch'>Already have an account?</div>",
        unsafe_allow_html=True,
    )
    if st.button("Log In", use_container_width=True, key="switch_to_login"):
        st.session_state["auth_form"] = "login"
        st.rerun()


def _render_auth_tabbar_and_form() -> None:
    active = st.session_state.get("auth_form", "login")

    tab_col1, tab_col2 = st.columns(2)
    with tab_col1:
        if st.button(
            "Log In",
            key="tab_login",
            use_container_width=True,
            type="primary" if active == "login" else "secondary",
        ):
            st.session_state["auth_form"] = "login"
            st.rerun()
    with tab_col2:
        if st.button(
            "Sign Up",
            key="tab_signup",
            use_container_width=True,
            type="primary" if active == "signup" else "secondary",
        ):
            st.session_state["auth_form"] = "signup"
            st.rerun()

    st.markdown('<hr class="auth-tabbar-divider">', unsafe_allow_html=True)

    if active == "login":
        _render_login_form()
    else:
        _render_signup_form()


# ==============================================================================
# UI — LOGIN / SIGNUP: SPLIT SCREEN LAYOUT
# ==============================================================================
def render_auth_page() -> None:
    _inject_auth_css()
    _render_bg_blobs()

    st.markdown('<div class="auth-animate">', unsafe_allow_html=True)
    left, right = st.columns([1, 1.2])

    with left:
        st.markdown(
            """
            <div class="split-left-marker split-left">
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <span class="floating-shield" style="font-size:1.5rem;">🛡️</span>
                    <span style="font-weight:700; color:#F8FAFC; font-size:1.05rem;">Smart Code Inspection</span>
                </div>
                <div>
                    <div class="split-quote auth-stagger auth-stagger-1">Analyze code.<br/>Ship with confidence.</div>
                    <div class="auth-stagger auth-stagger-2" style="color:#C4B5FD; margin-top:0.9rem; font-size:0.92rem; max-width:320px;">
                        Multi-agent static analysis, grounded severity scoring, and
                        LLM-powered remediation — all in one place.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="split-right">', unsafe_allow_html=True)

        back_left, back_right = st.columns([2.2, 1])
        with back_right:
            if st.button("Back to website →", key="auth_back_split", use_container_width=True):
                st.session_state["auth_screen"] = "welcome"
                st.rerun()

        _render_auth_tabbar_and_form()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# UI — TOP-RIGHT USER BAR
# ==============================================================================
def render_user_bar() -> None:
    user = current_user()
    if not user:
        return
    _inject_auth_css()

    name = user.get("full_name") or user.get("username")
    left, right = st.columns([4, 1.6])
    with right:
        with st.popover(f"👤 {name} ▾"):
            st.write(f"**{name}**")
            if user.get("email"):
                st.caption(user["email"])
            if st.button("Logout", key="logout_btn", use_container_width=True):
                log_out()
                st.rerun()


# ==============================================================================
# MAIN GATE — call this at the very top of app.py, before any page content
# ==============================================================================
def require_auth() -> bool:
    _init_auth_state()
    handle_oauth_callback()

    if is_authenticated():
        return True

    screen = st.session_state.get("auth_screen", "welcome")
    if screen == "welcome":
        render_welcome_page()
    else:
        render_auth_page()
    return False
    
