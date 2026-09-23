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


TOKEN_TTL_S = 60 * 60


REFRESH_TTL_S = 14 * 24 * 3600


_BCRYPT_ROUNDS = 10


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(_BCRYPT_ROUNDS)).decode()


def is_legacy_hash(hashed: str) -> bool:
    """True for pre-migration unsalted SHA-256 hex digests."""
    return not hashed.startswith("$2")


def verify_password(password: str, hashed: str) -> bool:
    if is_legacy_hash(hashed):


        digest = hashlib.sha256(password.encode()).hexdigest()
        return hmac.compare_digest(digest, hashed)
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        return False


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


def make_refresh_token() -> tuple[str, str]:
    """Return (raw_token, token_hash). The raw goes to the client once; only
    the hash is persisted, so a DB leak can't be replayed as a valid token."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
