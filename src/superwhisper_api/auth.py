"""Authentication helpers for Superwhisper-backed API calls."""
from __future__ import annotations

import json
import os
import plistlib
import sqlite3
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

CACHE_DB = Path.home() / "Library/Caches/com.superduper.superwhisper/Cache.db"
APP_PATH = "/Applications/superwhisper.app"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
ELEVENLABS_BATCH_KEY_PATTERN = "%batch-key"
V1_INFERENCE_KEY_URL = "https://api.superwhisper.com/v1/inference/key"
V2_INFERENCE_KEY_URL = "https://api.superwhisper.com/v2/inference/key"
POLL_TIMEOUT = 60
SIGNED_HEADER_NAMES = ("X-ID", "X-License", "X-Signature")
NON_FORWARDABLE_HEADER_NAMES = {
    "__hhaa__",
    "accept-encoding",
    "connection",
    "content-length",
    "host",
}


@dataclass(frozen=True)
class CachedAuth:
    """Signed base URL and headers extracted from the Superwhisper cache."""

    base_url: str
    headers: dict[str, str]


def cached_auth(cache_db: Path = CACHE_DB) -> CachedAuth:
    """Return cached signed auth by reading the Superwhisper URL cache database."""
    if not cache_db.exists():
        raise RuntimeError(f"Superwhisper cache does not exist: {cache_db}")

    with sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            select r.request_key, b.request_object
            from cfurl_cache_response r
            join cfurl_cache_blob_data b on b.entry_id = r.entry_id
            where r.request_key like 'https://api.superwhisper.com/%'
            order by
                case when r.request_key like ? then 0 else 1 end,
                r.time_stamp desc
            limit 25
            """,
            (f"%{CHAT_COMPLETIONS_PATH}",),
        ).fetchall()

    for request_url, request_object in rows:
        payload = request_object.encode() if isinstance(request_object, str) else request_object
        try:
            headers = signed_headers_from_plist(plistlib.loads(payload))
        except RuntimeError:
            continue

        parsed = urlsplit(request_url)
        return CachedAuth(base_url=f"{parsed.scheme}://{parsed.netloc}", headers=headers)

    raise RuntimeError("No cached Superwhisper signed request found.")


def signed_headers_from_plist(payload: object) -> dict[str, str]:
    """Build headers from the signed request embedded in a cached plist payload."""
    cached = find_signed_header_source(payload)
    if cached is None:
        raise RuntimeError("Cached Superwhisper request is missing signed headers.")

    for name in SIGNED_HEADER_NAMES:
        value = cached.get(name)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"Cached Superwhisper request is missing {name}.")

    headers: dict[str, str] = {}
    for name, raw_value in cached.items():
        if not isinstance(name, str) or name.lower() in NON_FORWARDABLE_HEADER_NAMES:
            continue
        value = raw_value.decode() if isinstance(raw_value, bytes) else raw_value
        if isinstance(value, str) and value:
            headers[name] = value
    return headers


def find_signed_header_source(payload: object) -> dict[str, object] | None:
    """Recursively search a plist payload for the dict containing signed headers."""
    if isinstance(payload, dict):
        if all(name in payload for name in SIGNED_HEADER_NAMES):
            return cast("dict[str, object]", payload)
        for value in payload.values():
            found = find_signed_header_source(value)
            if found is not None:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = find_signed_header_source(value)
            if found is not None:
                return found
    return None


def extract_elevenlabs_key_from_cache(cache_db: Path = CACHE_DB) -> str | None:
    """Read the latest ElevenLabs batch key from the Superwhisper URL cache."""
    return _cached_receiver_key(
        """
        select d.receiver_data
        from cfurl_cache_response r
        join cfurl_cache_receiver_data d on d.entry_id = r.entry_id
        where r.request_key like ?
        order by r.time_stamp desc
        limit 1
        """,
        (ELEVENLABS_BATCH_KEY_PATTERN,),
        cache_db,
    )


def ensure_elevenlabs_key() -> str:
    """Return a valid ElevenLabs batch key, triggering Superwhisper if necessary."""
    key = extract_elevenlabs_key_from_cache()
    if key:
        return key

    subprocess.run(["open", "-g", "-a", APP_PATH], check=True)
    time.sleep(4)

    key = extract_elevenlabs_key_from_cache()
    if key:
        return key

    dummy = _make_silent_wav()
    try:
        subprocess.run(["open", "-g", "-a", APP_PATH, str(dummy)], check=True)
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            key = extract_elevenlabs_key_from_cache()
            if key:
                return key
            time.sleep(1)
    finally:
        dummy.unlink(missing_ok=True)

    raise RuntimeError("Timed out waiting for ElevenLabs batch key.")


def extract_v1_key_from_cache(cache_db: Path = CACHE_DB) -> str | None:
    """Read the latest Ultra v1 inference key from the Superwhisper URL cache."""
    return _extract_inference_key_from_cache(V1_INFERENCE_KEY_URL, cache_db)


def extract_v2_key_from_cache(cache_db: Path = CACHE_DB) -> str | None:
    """Read the latest S1 Voice v2 inference key from the Superwhisper URL cache."""
    return _extract_inference_key_from_cache(V2_INFERENCE_KEY_URL, cache_db)


def ensure_v1_key() -> str:
    """Return a valid Ultra v1 inference key, fetching from the backend if needed."""
    return _ensure_inference_key(V1_INFERENCE_KEY_URL, extract_v1_key_from_cache, "v1")


def ensure_v2_key() -> str:
    """Return a valid S1 Voice v2 inference key, fetching from the backend if needed."""
    return _ensure_inference_key(V2_INFERENCE_KEY_URL, extract_v2_key_from_cache, "v2")


def _extract_inference_key_from_cache(key_url: str, cache_db: Path) -> str | None:
    return _cached_receiver_key(
        """
        select d.receiver_data
        from cfurl_cache_response r
        join cfurl_cache_receiver_data d on d.entry_id = r.entry_id
        where r.request_key = ?
        order by r.time_stamp desc
        limit 1
        """,
        (key_url,),
        cache_db,
    )


def _cached_receiver_key(
    query: str,
    params: tuple[str, ...],
    cache_db: Path,
) -> str | None:
    if not cache_db.exists():
        return None
    try:
        with sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True) as conn:
            row = conn.execute(query, params).fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        key = data.get("key")
        return key if isinstance(key, str) and key else None
    except Exception:
        return None


def _ensure_inference_key(
    key_url: str,
    cache_reader: Callable[[], str | None],
    version_label: str,
) -> str:
    key = cache_reader()
    if key:
        return key

    auth = cached_auth()
    response = httpx.post(
        key_url,
        headers={**auth.headers, "Content-Type": "application/json"},
        json={},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    key = data.get("key")
    if not isinstance(key, str) or not key:
        raise RuntimeError(f"{version_label}/inference/key response missing 'key' field.")

    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        cached_key = cache_reader()
        if cached_key:
            return cached_key
        time.sleep(0.5)

    return key


def _make_silent_wav() -> Path:
    descriptor, filename = tempfile.mkstemp(suffix=".wav")
    os.close(descriptor)
    tmp = Path(filename)
    with wave.open(str(tmp), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    return tmp
