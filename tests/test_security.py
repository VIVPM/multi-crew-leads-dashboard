"""Self-contained checks for backend/security.py. Run: python tests/test_security.py"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from security import (
    hash_password, verify_password, is_legacy_hash, make_token, verify_token,
)

SECRET = "test-secret"

# bcrypt round-trip
h = hash_password("hunter2!")
assert not is_legacy_hash(h)
assert verify_password("hunter2!", h)
assert not verify_password("wrong", h)

# legacy SHA-256 hashes still verify (lazy migration path)
legacy = hashlib.sha256(b"oldpassword").hexdigest()
assert is_legacy_hash(legacy)
assert verify_password("oldpassword", legacy)
assert not verify_password("wrong", legacy)

# token round-trip
tok = make_token("user-123", SECRET)
assert verify_token(tok, SECRET) == "user-123"

# tampered token rejected
try:
    verify_token(tok[:-4] + "AAAA", SECRET)
    raise AssertionError("tampered token accepted")
except ValueError:
    pass

# wrong secret rejected
try:
    verify_token(tok, "other-secret")
    raise AssertionError("wrong-secret token accepted")
except ValueError:
    pass

# expired token rejected
expired = make_token("user-123", SECRET, ttl_s=-10)
try:
    verify_token(expired, SECRET)
    raise AssertionError("expired token accepted")
except ValueError:
    pass

# garbage rejected
try:
    verify_token("not-a-token", SECRET)
    raise AssertionError("garbage token accepted")
except ValueError:
    pass

print("test_security.py: all checks passed")
