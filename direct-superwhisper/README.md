# direct-superwhisper

Call Superwhisper's signed language-model proxy directly, without opening the GUI.

This does not extract a raw OpenAI API key. It reuses the signed request headers
that Superwhisper caches for its own proxy calls.

## Usage

### `run-prompt.py`

Reusable command for ongoing direct requests through the Superwhisper proxy.
This is the script to extend for workflows like WER classification.

```bash
uv run direct-superwhisper/run-prompt.py --model gpt-5.4-mini "reply exactly ok"
uv run direct-superwhisper/run-prompt.py --model claude-sonnet-4-6 "reply exactly ok"
uv run direct-superwhisper/run-prompt.py \
  --model gpt-5.4-mini \
  --json-object \
  --json \
  "Return JSON with primary_issue deletion and reason missing words."
```

### `check-models.py`

Diagnostic command for checking which model IDs are actually honored. Some
unknown IDs return HTTP 200 while silently falling back to another model.

```bash
uv run direct-superwhisper/check-models.py --catalog
uv run direct-superwhisper/check-models.py gpt-5.4-mini claude-sonnet-4-6 gpt-5.5-low
```

### `client.py`

Shared proxy client code used by the commands. It reads the signed headers,
loads the cached model catalog, routes providers, parses SSE responses, and
builds request payloads.

The scripts read the cached Superwhisper request from:

```text
~/Library/Caches/com.superduper.superwhisper/Cache.db
```

Some public-looking model IDs silently fall back to `gpt-3.5-turbo-0125`.
`check-models.py` checks the returned model field so those false positives are visible.
