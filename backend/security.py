"""
security.py — password hashing (bcrypt, with legacy SHA-256 verify for
lazy migration) and HMAC-signed session tokens. No third-party deps
beyond bcrypt; no network, so it stays unit-testable.
"""

import base64
import hashlib
import hmac
import secrets
import time

import bcrypt

# Access token: short-lived so a leaked one expires fast (it's stateless and
# can't be revoked). The refresh token below covers staying logged in.
TOKEN_TTL_S = 60 * 60           # 60 minutes
# Refresh token: long-lived, revocable (stored server-side), silently mints
# new access tokens. This is the real "how long you stay logged in" window.
REFRESH_TTL_S = 14 * 24 * 3600  # 14 days


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

# bcrypt's default cost is 12 rounds. Load testing against the deployed
# Render free tier (0.1 vCPU) measured 18-29s p95 login latency at just
# 10-25 concurrent users — bcrypt is deliberately CPU-heavy, and a tenth of a
# core doesn't parallelize concurrent hashing, it serializes it. Cost 10 is
# ~4x less CPU time per hash (each step doubles/halves the work) and is
# still within bcrypt's commonly-accepted range for production use — a real
# but modest tradeoff, made explicitly here rather than silently. This only
# affects hashes created from now on: bcrypt embeds the cost used in the hash
# string itself, and checkpw reads it back out automatically, so existing
# cost-12 hashes keep verifying correctly with no migration needed.
_BCRYPT_ROUNDS = 10


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(_BCRYPT_ROUNDS)).decode()


def is_legacy_hash(hashed: str) -> bool:
    """True for pre-migration unsalted SHA-256 hex digests."""
    return not hashed.startswith("$2")


def verify_password(password: str, hashed: str) -> bool:
    if is_legacy_hash(hashed):
        # legacy unsalted SHA-256 — accepted so existing users can log in;
        # the caller re-hashes with bcrypt on success (lazy migration)
        digest = hashlib.sha256(password.encode()).hexdigest()
        return hmac.compare_digest(digest, hashed)
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Session tokens: base64(user_id:expiry:hmac_sha256(secret, user_id:expiry))
# ---------------------------------------------------------------------------

def make_token(user_id: str, secret: str, ttl_s: int = TOKEN_TTL_S) -> str:
    payload = f"{user_id}:{int(time.time()) + ttl_s}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}:{sig}".encode()).decode()


def verify_token(token: str, secret: str) -> str:
    """Return the user_id if the token is valid and unexpired, else raise ValueError."""
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        user_id, exp, sig = decoded.rsplit(":", 2)
        payload = f"{user_id}:{exp}"
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        expired = time.time() > int(exp)
    except Exception:
        raise ValueError("malformed token")
    if not hmac.compare_digest(sig, expected):
        raise ValueError("bad signature")
    if expired:
        raise ValueError("token expired")
    return user_id


# ---------------------------------------------------------------------------
# Refresh tokens: opaque random strings, stored server-side only as a hash
# ---------------------------------------------------------------------------

def make_refresh_token() -> tuple[str, str]:
    """Return (raw_token, token_hash). The raw goes to the client once; only
    the hash is persisted, so a DB leak can't be replayed as a valid token."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
