# Super Whisperer CLI

Tiny CLI for batch-opening audio files in Superwhisper and routing the transcript rows that Superwhisper writes to its local database.

This does not run its own ASR model. It uses the currently selected Superwhisper mode and model.

## Usage

Run a manifest job:

```bash
uv run super-whisperer \
  --paths-file /path/to/audio-paths.txt \
  --jsonl results/scribev2.jsonl \
  --fail-jsonl results/scribev2.failures.jsonl
```

## How It Works

For each audio file, the CLI runs:

```bash
open -g -a /Applications/superwhisper.app "/path/to/audio"
```

Then it polls:

```text
~/Library/Application Support/superwhisper/database/superwhisper.sqlite
```

It waits for the next `recording.fromFile = 1` row, retries empty transcript rows up to three times, then writes a JSON object containing:

- input audio path
- Superwhisper recording id
- recording folder name
- model name
- mode name
- duration
- processing time
- transcript

## Notes

Mode switching is intentionally not implemented. Set the mode in Superwhisper first; this CLI assumes the current Superwhisper mode is the mode you want.

## Development

```bash
uv run ruff check .
uv run super-whisperer --help
```
