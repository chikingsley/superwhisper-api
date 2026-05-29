---
name: macwhisper-dictionary
description: Use when the user wants to inspect, add, remove, or confirm MacWhisper Global Replace / Find and Replace dictionary entries, especially from recent dictation audio or user-provided misrecognitions.
---

# MacWhisper Dictionary

Use this for MacWhisper Global Replace dictionary work in this repo.

## Commands

Run commands from the repo root:

```bash
uv run superwhisper-macwhisper learn --latest 1
uv run superwhisper-macwhisper learn --latest 5
uv run superwhisper-macwhisper learn /absolute/path/to/audio.m4a
uv run superwhisper-macwhisper apply '[{"original":"bad text","replacement":"GoodText"}]'
uv run superwhisper-macwhisper remove '["bad text"]'
```

`learn` prints Parakeet and Scribe transcripts plus the current MacWhisper replacement list. It is read-only.

`apply` writes approved replacement pairs into MacWhisper. It expects one JSON array argument with `original` and `replacement` strings.

`remove` removes entries by their `original` strings. It expects one JSON array argument of strings.

## Workflow

If the user gives explicit replacements in chat, do not run audio unless they ask. Restate the exact pairs you plan to add and wait for confirmation unless they already explicitly said to apply.

If the user asks to inspect recent audio, run `learn --latest N`, read both transcripts, then propose exact replacements in chat. Apply only after the user approves.

Prefer the user's stated target spelling/casing over either transcript. Use transcripts to discover bad `original` forms.

Do not add spelled-out forms when the user is spelling a target for you. For example, if they say `P Y C A C H E` to explain `__pycache__`, do not add `P Y C A C H E -> __pycache__` unless they explicitly say that exact text appears in transcripts and should be replaced.

Avoid broad global replacements when the word is common unless the user approves the risk. Brand/tool names and file artifacts are usually safer than ordinary English words.

After `apply`, the CLI response is usually enough confirmation. Use a direct MacWhisper defaults readback only when debugging writer behavior, duplicate/case behavior, or a suspected failed write.

## Current Conventions

Use the active repo command name: `superwhisper-macwhisper`.

The helper is intentionally agent-driven. There is no target config file to edit for normal dictionary additions.

Common target examples:

- `__pycache__`
- `.ruff_cache`
- `.pytest_cache`
- `.DS_Store`
- `.gitignore`
- `pyproject.toml`
- `gitignored`
