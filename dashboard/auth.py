"""
Dashboard authentication — env-var login (default) or Entra ID (optional).

AUTH_MODE controls behavior:
  - env   : DASH_USER / DASH_PASS (current portfolio default)
  - entra : Microsoft Entra ID via MSAL device code flow
  - dual  : try Entra first, fall back to env login
"""

from __future__ import annotations

import hmac
import os
from typing import Optional, Tuple

import streamlit as st

AUTH_MODE = os.getenv("AUTH_MODE", "env").strip().lower()
DASH_USER = os.getenv("DASH_USER", "").strip().lower()
DASH_PASS = os.getenv("DASH_PASS", "")
DEFAULT_ROLE = os.getenv("DASH_ROLE", "EXEC").strip().upper()

ENTRA_TENANT_ID = os.getenv("ENTRA_TENANT_ID", "").strip()
ENTRA_DASHBOARD_CLIENT_ID = os.getenv("ENTRA_DASHBOARD_CLIENT_ID", "").strip()
ENTRA_API_SCOPE = os.getenv("ENTRA_API_SCOPE", "").strip()


def _constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _init_session() -> None:
    if "authed" not in st.session_state:
        st.session_state.authed = False
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.access_token = None


def _env_login_form() -> None:
    """Portfolio default — environment variable credentials."""
    if not DASH_USER or not DASH_PASS:
        st.error("Auth is not configured. Set DASH_USER and DASH_PASS, or enable Entra auth.")
        st.stop()

    st.title("Sign in")
    st.caption("Authorized access only.")

    with st.form("login"):
        email = st.text_input("Email").strip().lower()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if _constant_time_equals(email, DASH_USER) and _constant_time_equals(password, DASH_PASS):
            st.session_state.authed = True
            st.session_state.user = email
            st.session_state.role = DEFAULT_ROLE
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid email or password.")
            st.stop()


def _entra_device_code_login() -> None:
    """
    Entra login via MSAL device code flow.

    Works without redirect URI configuration — suitable for portfolio demos
    and pre-subscription development. Production should use auth code flow
    behind APIM EasyAuth or a proper OIDC redirect handler.
    """
    if not ENTRA_TENANT_ID or not ENTRA_DASHBOARD_CLIENT_ID:
        st.error(
            "Entra auth requires ENTRA_TENANT_ID and ENTRA_DASHBOARD_CLIENT_ID. "
            "Run scripts/entra/register-apps.ps1 or set config/entra/generated.env."
        )
        st.stop()

    try:
        import msal
    except ImportError:
        st.error("MSAL not installed. Run: pip install -r requirements-entra.txt")
        st.stop()

    st.title("Sign in with Microsoft")
    st.caption("Enterprise authentication via Microsoft Entra ID.")

    if st.button("Sign in with Microsoft"):
        authority = f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}"
        scopes = [ENTRA_API_SCOPE] if ENTRA_API_SCOPE else ["User.Read"]

        app = msal.PublicClientApplication(
            client_id=ENTRA_DASHBOARD_CLIENT_ID,
            authority=authority,
        )
        flow = app.initiate_device_flow(scopes=scopes)
        if "user_code" not in flow:
            st.error("Failed to create device code flow.")
            st.stop()

        st.info(flow["message"])
        result = app.acquire_token_by_device_flow(flow)

        if "access_token" in result:
            st.session_state.authed = True
            st.session_state.user = result.get("id_token_claims", {}).get(
                "preferred_username", "entra-user"
            )
            st.session_state.role = _role_from_entra_result(result)
            st.session_state.access_token = result["access_token"]
            st.success("Microsoft sign-in successful.")
            st.rerun()
        else:
            st.error(result.get("error_description", "Microsoft sign-in failed."))
            st.stop()


def _role_from_entra_result(result: dict) -> str:
    """Extract app role from ID token claims when available."""
    claims = result.get("id_token_claims") or {}
    roles = claims.get("roles") or []
    for role in ["EXEC", "OPS", "ANALYST"]:
        if role in roles:
            return role
    return DEFAULT_ROLE


def get_api_auth_headers() -> dict:
    """Return Authorization header when Entra token is available for API calls."""
    token = st.session_state.get("access_token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def require_login() -> None:
    """Unified auth gate — respects AUTH_MODE without breaking env-var default."""
    _init_session()
    if st.session_state.authed:
        return

    if AUTH_MODE == "entra":
        _entra_device_code_login()
        return

    if AUTH_MODE == "dual":
        tab_env, tab_entra = st.tabs(["Environment Login", "Microsoft Entra ID"])
        with tab_env:
            _env_login_form()
        with tab_entra:
            _entra_device_code_login()
        return

    _env_login_form()


def auth_mode_label() -> str:
    """Human-readable label for debug panel."""
    if AUTH_MODE == "entra":
        return "Entra ID (device code)"
    if AUTH_MODE == "dual":
        return "Dual (Entra + env)"
    return "Environment variables"
