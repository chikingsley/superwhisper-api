"""Tests for the MacWhisper Global Replace apply/remove logic.

These run against an in-memory stand-in for MacWhisper's preference store (via
the ``store`` fixture), so they never touch the real Global Replace list. The
network/model suites stay fully live; this file only exercises local logic.
"""
from __future__ import annotations

import argparse
import json

import pytest

from superwhisper_api.macwhisper import replacements as R


@pytest.fixture
def store(monkeypatch):
    """Replace the real defaults-backed store with an in-memory list."""
    state: dict[str, list[dict[str, str]]] = {"items": []}
    monkeypatch.setattr(R, "_read_global_replace_items", lambda: list(state["items"]))
    monkeypatch.setattr(
        R, "_write_global_replace_items", lambda items: state.__setitem__("items", list(items))
    )
    return state


def _apply(pairs: list[dict[str, str]]) -> int:
    return R._cmd_apply(argparse.Namespace(pairs=json.dumps(pairs)))


def _remove(originals: list[str]) -> int:
    return R._cmd_remove(argparse.Namespace(originals=json.dumps(originals)))


def _originals(items: list[dict[str, str]]) -> set[str]:
    return {item["original"] for item in items}


# --- apply: single one-to-one ---------------------------------------------


def test_apply_single(store):
    assert _apply([{"original": "Deep Gram", "replacement": "Deepgram"}]) == 0
    assert len(store["items"]) == 1
    item = store["items"][0]
    assert item["original"] == "Deep Gram"
    assert item["replacement"] == "Deepgram"
    assert item["id"]  # a generated id is present


# --- apply: many variants mapping to one canonical name -------------------


def test_apply_many_variants_same_replacement(store):
    variants = ["Chi Bazor", "Chiba Zor", "Chibazor", "Chi Buzor"]
    assert _apply([{"original": v, "replacement": "Chibuzor"} for v in variants]) == 0
    assert len(store["items"]) == len(variants)
    assert _originals(store["items"]) == set(variants)
    assert all(item["replacement"] == "Chibuzor" for item in store["items"])


# --- apply: ten at once (mixed names) -------------------------------------


def test_apply_ten_distinct(store):
    pairs = [{"original": f"orig {i}", "replacement": f"repl{i}"} for i in range(10)]
    assert _apply(pairs) == 0
    assert len(store["items"]) == 10
    assert _originals(store["items"]) == {f"orig {i}" for i in range(10)}


# --- apply: existing original is updated in place (the only behavior) ------


def test_apply_updates_existing_case_insensitively(store):
    _apply([{"original": "Deep Gram", "replacement": "Deepgram"}])
    # Same original (different case) with a new replacement -> update, not add.
    assert _apply([{"original": "deep gram", "replacement": "DeepgramX"}]) == 0
    assert len(store["items"]) == 1
    assert store["items"][0]["replacement"] == "DeepgramX"


def test_apply_idempotent_when_unchanged(store):
    pair = [{"original": "Deep Gram", "replacement": "Deepgram"}]
    _apply(pair)
    _apply(pair)
    assert len(store["items"]) == 1


# --- remove: drop the ones we added ---------------------------------------


def test_remove_all_added(store):
    variants = ["Chi Bazor", "Chiba Zor", "Chibazor"]
    _apply([{"original": v, "replacement": "Chibuzor"} for v in variants])
    assert _remove(variants) == 0
    assert store["items"] == []


def test_remove_subset_and_case_insensitive(store):
    _apply(
        [
            {"original": "Deep Gram", "replacement": "Deepgram"},
            {"original": "Open Code", "replacement": "opencode"},
        ]
    )
    assert _remove(["deep gram"]) == 0  # different case still matches
    assert _originals(store["items"]) == {"Open Code"}


def test_remove_unknown_is_noop(store):
    _apply([{"original": "Deep Gram", "replacement": "Deepgram"}])
    assert _remove(["Not Present"]) == 0
    assert len(store["items"]) == 1


# --- apply then remove round-trip -----------------------------------------


def test_apply_then_remove_roundtrip(store):
    pairs = [{"original": f"orig {i}", "replacement": f"repl{i}"} for i in range(10)]
    _apply(pairs)
    assert len(store["items"]) == 10
    _remove([f"orig {i}" for i in range(10)])
    assert store["items"] == []


# --- input validation ------------------------------------------------------


def test_apply_rejects_bad_json(store):
    assert R._cmd_apply(argparse.Namespace(pairs="not json")) == 1
    assert store["items"] == []


def test_apply_rejects_wrong_shape(store):
    assert R._cmd_apply(argparse.Namespace(pairs=json.dumps([{"original": "x"}]))) == 1
    assert store["items"] == []


def test_remove_rejects_bad_json(store):
    assert R._cmd_remove(argparse.Namespace(originals="{}")) == 1
