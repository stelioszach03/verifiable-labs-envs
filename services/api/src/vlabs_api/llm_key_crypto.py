"""Customer LLM API-key encryption helpers (Phase 23.B).

The customer-supplied LLM API key on a ``dataset_jobs`` row is stored
encrypted at rest via Fernet (cryptography.fernet) — symmetric AES-128
+ HMAC-SHA256 authenticated encryption. The symmetric key comes from
the ``VLABS_DATA_LLM_KEY_ENCRYPTION`` env (Fly secret in production).

Why Fernet rather than pgcrypto's ``pgp_sym_encrypt``:

- Fernet ciphertext is a single bytes blob — fits cleanly in the
  ``BYTEA`` column and round-trips through SQLAlchemy without
  per-query SQL function calls.
- pgcrypto's ``pgp_sym_encrypt`` requires the extension to be
  installed; pgserver (used in tests) doesn't ship it. Fernet runs
  entirely in Python — same code path tests + production.
- Migration path to envelope encryption (KMS / cloud HSM) later is
  drop-in: replace :class:`Fernet` with a wrapped-DEK pattern; the
  on-disk schema doesn't change.

LOCAL_FAKE_R2 mode (set via ``VLABS_LOCAL_FAKE_R2=true``) requires no
encryption key — :func:`encrypt_llm_api_key` returns the plaintext
UTF-8 bytes prefixed with a fixed marker so :func:`decrypt_llm_api_key`
knows to skip decryption. This keeps the test path simple without
exposing the marker in production paths.
"""
from __future__ import annotations

import base64
import hashlib
from typing import Final

from cryptography.fernet import Fernet, InvalidToken

from vlabs_api.config import get_settings

# Marker for LOCAL_FAKE_R2 plaintext — never appears in real Fernet output
# (which is always a urlsafe-base64 token starting with ``gAAAAA``).
_FAKE_MARKER: Final[bytes] = b"FAKE::"


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a Fernet-format key (32 bytes urlsafe-base64) from any secret.

    Uses SHA-256 of the secret as the 32-byte key material; Fernet
    expects this base64-encoded. This lets operators set
    ``VLABS_DATA_LLM_KEY_ENCRYPTION`` to any sufficiently random
    string — Fly secret rotation doesn't need to produce raw key
    material in a specific format.
    """
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_llm_api_key(plaintext: str) -> bytes:
    """Encrypt an LLM API key for storage in ``dataset_jobs.llm_api_key_encrypted``.

    Short-circuits to a marker-prefixed plaintext only when
    ``VLABS_DATA_LLM_KEY_ENCRYPTION`` is empty (tests / dev without
    secrets). Production deploys must set the secret to engage Fernet
    authenticated encryption — independent of the LOCAL_FAKE_R2 flag,
    which only governs object-storage destination.
    """
    settings = get_settings()
    if not settings.vlabs_data_llm_key_encryption:
        return _FAKE_MARKER + plaintext.encode("utf-8")
    fernet = Fernet(_derive_fernet_key(settings.vlabs_data_llm_key_encryption))
    return fernet.encrypt(plaintext.encode("utf-8"))


def decrypt_llm_api_key(ciphertext: bytes) -> str:
    """Decrypt the stored ciphertext back to the original LLM API key."""
    if ciphertext.startswith(_FAKE_MARKER):
        return ciphertext[len(_FAKE_MARKER):].decode("utf-8")
    settings = get_settings()
    if not settings.vlabs_data_llm_key_encryption:
        raise RuntimeError(
            "VLABS_DATA_LLM_KEY_ENCRYPTION not set; cannot decrypt "
            "dataset_jobs.llm_api_key_encrypted"
        )
    fernet = Fernet(_derive_fernet_key(settings.vlabs_data_llm_key_encryption))
    try:
        return fernet.decrypt(ciphertext).decode("utf-8")
    except InvalidToken as exc:
        raise RuntimeError(
            "dataset_jobs.llm_api_key_encrypted failed Fernet auth — "
            "VLABS_DATA_LLM_KEY_ENCRYPTION may have been rotated"
        ) from exc


__all__ = [
    "encrypt_llm_api_key",
    "decrypt_llm_api_key",
]
