# direct-eleven

Transcribe audio via ElevenLabs API directly using Superwhisper's batch key — no GUI needed.

## Setup

No Python dependencies to install. Scripts use inline metadata with `uv run --script`.

```bash
uv run direct-eleven/extract-key.py
```

If the key isn't cached, it briefly launches Superwhisper to fetch one, then you can use the key directly.

## Scripts

### `extract-key.py`

Print the current ElevenLabs batch key from Superwhisper's CFURL cache.

```bash
uv run direct-eleven/extract-key.py
# sk_4a39491e0dc57a63089510234b6835eee8e383b483dd6b6f
```

If no key is cached, it launches Superwhisper, waits for the key to appear, then prints it.

### `transcribe.py`

Unified script to transcribe a single audio file or process a batch of files in parallel.

#### Single-File Transcription

Transcribe a single audio file and output the exact JSON response.

```bash
uv run direct-eleven/transcribe.py audio.wav --model scribe_v2
uv run direct-eleven/transcribe.py speech.mp3 --language fas --pretty
```

#### Batch-File Transcription

Read a paths file, transcribe each audio file via ElevenLabs in parallel, and write JSONL results (supports resuming).

```bash
uv run direct-eleven/transcribe.py \
  --paths-file paths.txt \
  --jsonl results.jsonl \
  --fail-jsonl failures.jsonl \
  --model scribe_v2 \
  --max-workers 8
```

The paths-file format is one audio file path per line, same as the original `superwhisper-api` CLI.

## Replace the old workers

Instead of running 5 Superwhisper instances + polling SQLite, run one process:

```bash
uv run direct-eleven/transcribe.py \
  --paths-file /Volumes/.../paths.txt \
  --jsonl /Volumes/.../results.jsonl \
  --max-workers 8
```

No GUI, no memory bloat, no 2-hour recycle needed.
