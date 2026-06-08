"""
Optional Entra ID JWT validation for FastAPI.

Disabled by default (ENTRA_AUTH_ENABLED=false). When enabled, protected
routes require a valid Bearer token issued by Microsoft Entra ID.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Set

import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)

# Paths that remain public when Entra auth is enabled.
PUBLIC_PATHS: Set[str] = {
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def entra_auth_enabled() -> bool:
    """Return True when Entra JWT validation should be enforced."""
    return os.getenv("ENTRA_AUTH_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _tenant_id() -> str:
    tenant = os.getenv("ENTRA_TENANT_ID", "").strip()
    if not tenant:
        raise RuntimeError("ENTRA_TENANT_ID is required when ENTRA_AUTH_ENABLED=true")
    return tenant


def _audience() -> str:
    """Accepted token audience — defaults to api://{ENTRA_API_CLIENT_ID}."""
    explicit = os.getenv("ENTRA_API_AUDIENCE", "").strip()
    if explicit:
        return explicit
    client_id = os.getenv("ENTRA_API_CLIENT_ID", "").strip()
    if not client_id:
        raise RuntimeError(
            "ENTRA_API_CLIENT_ID or ENTRA_API_AUDIENCE is required when auth is enabled"
        )
    return f"api://{client_id}"


class _JwksCache:
    """Fetch and cache Entra JWKS keys."""

    def __init__(self) -> None:
        self._keys: Optional[Dict[str, Any]] = None
        self._fetched_at: float = 0.0
        self._ttl_seconds: int = 3600

    def get_keys(self, tenant_id: str) -> Dict[str, Any]:
        now = time.time()
        if self._keys and (now - self._fetched_at) < self._ttl_seconds:
            return self._keys

        jwks_url = (
            f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"
        )
        response = requests.get(jwks_url, timeout=15)
        response.raise_for_status()
        self._keys = response.json()
        self._fetched_at = now
        return self._keys


_jwks_cache = _JwksCache()


def _decode_token(token: str) -> Dict[str, Any]:
    """Validate JWT signature and claims against Entra JWKS."""
    try:
        import jwt
        from jwt import PyJWKClient
    except ImportError as exc:
        raise RuntimeError(
            "PyJWT is required for Entra auth. Install requirements-entra.txt"
        ) from exc

    tenant_id = _tenant_id()
    audience = _audience()
    issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
    jwks_url = f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"

    jwk_client = PyJWKClient(jwks_url)
    signing_key = jwk_client.get_signing_key_from_jwt(token)

    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=audience,
        issuer=issuer,
    )


def require_entra_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency — validates Bearer token when Entra auth is enabled.

    Returns decoded token claims, or None when auth is disabled.
    """
    if not entra_auth_enabled():
        return None

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = _decode_token(credentials.credentials)
        return claims
    except Exception as exc:
        logger.warning("Entra token validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def map_role_from_claims(claims: Optional[Dict[str, Any]]) -> str:
    """Map Entra app roles from token claims to dashboard/API role labels."""
    if not claims:
        return os.getenv("DASH_ROLE", "EXEC").strip().upper()

    roles = claims.get("roles") or []
    if not roles:
        return "ANALYST"

    priority = ["EXEC", "OPS", "ANALYST"]
    for role in priority:
        if role in roles:
            return role
    return str(roles[0]).upper()
