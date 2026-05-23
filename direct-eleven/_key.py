from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import time
import wave
from pathlib import Path

CACHE_DB = Path.home() / "Library/Caches/com.superduper.superwhisper/Cache.db"
APP_PATH = "/Applications/superwhisper.app"
POLL_TIMEOUT = 60


def extract_key_from_cache() -> str | None:
    if not CACHE_DB.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True)
        row = conn.execute(
            """
            select d.receiver_data
            from cfurl_cache_response r
            join cfurl_cache_receiver_data d on d.entry_id = r.entry_id
            where r.request_key like '%batch-key'
            order by r.time_stamp desc
            limit 1
            """
        ).fetchone()
        conn.close()
        if row:
            data = json.loads(row[0])
            return data.get("key")
    except Exception:
        return None
    else:
        return None


def _make_silent_wav() -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    with wave.open(str(tmp), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return tmp


def ensure_key() -> str:
    key = extract_key_from_cache()
    if key:
        return key

    subprocess.run(["open", "-g", "-a", APP_PATH], check=True)
    time.sleep(4)

    key = extract_key_from_cache()
    if key:
        return key

    dummy = _make_silent_wav()
    try:
        subprocess.run(["open", "-g", "-a", APP_PATH, str(dummy)], check=True)
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            key = extract_key_from_cache()
            if key:
                return key
            time.sleep(1)
    finally:
        dummy.unlink(missing_ok=True)

    msg = "Timed out waiting for batch key."
    raise RuntimeError(msg)
