# Superwhisper API

Tools for working with Superwhisper: the local transcription database path and the signed model API proxy.

This does not run its own ASR model. It uses the currently selected Superwhisper mode and model.

## Model API

The package also includes a small client for Superwhisper's signed model API proxy:

```python
from superwhisper_api.text.client import SuperwhisperClient

client = SuperwhisperClient()
response = client.generate(
    "claude-sonnet-4-6",
    [{"role": "user", "content": "Return only JSON: {\"category\":\"example\"}"}],
)
print(response.text)
```

Confirmed model routes live in `src/superwhisper_api/text/models.py`.

Observed behavior:

- Superwhisper returns server-sent-event responses for all tested model routes.
- `gpt-5.2` and `gpt-5.3-chat-latest` work on `/v1/chat/completions`.
- `gpt-5.4-mini` and `gpt-5.4-nano` use the live-proven `sw-gpt-5.4-*` request IDs on `/v1/chat/completions`; direct IDs fall back to `gpt-3.5-turbo-0125`.
- `claude-sonnet-4-6` works on `/anthropic/v1/messages`.
- `claude-haiku-4-5` works on `/anthropic/v1/messages`.
- Gemini uses its own `/gemini/v1/messages` route.
- Passing a public-looking model id to the wrong route can return HTTP 200 with a fallback model.
- OpenAI-style `response_format` is accepted by the GPT route, but JSON Schema is not enforced by Superwhisper. Validate structured output locally.

## Usage

Transcribe audio through the ElevenLabs key that Superwhisper caches:

```bash
uv run superwhisper-audio audio.wav --model scribe-v2

uv run superwhisper-audio \
  --paths-file /path/to/audio-paths.txt \
  --jsonl results/scribev2.jsonl \
  --fail-jsonl results/scribev2.failures.jsonl
```

Expose Scribe v2 to MacWhisper's custom OpenAI-compatible cloud provider:

```bash
uv run superwhisper-macwhisper-proxy \
  --host 127.0.0.1 \
  --port 8766 \
  --token local-test-token
```

MacWhisper settings for the custom provider:

```text
Base URL: http://127.0.0.1:8766
Authentication Token: local-test-token
Model Name: scribe-v2
```

Run the proxy at login with launchd:

```bash
mkdir -p ~/Library/Logs/superwhisper-api
ln -sf "$PWD/launchd/com.simon.superwhisper-api.macwhisper-proxy.plist" \
  ~/Library/LaunchAgents/com.simon.superwhisper-api.macwhisper-proxy.plist
launchctl bootstrap "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.simon.superwhisper-api.macwhisper-proxy.plist
launchctl enable "gui/$(id -u)/com.simon.superwhisper-api.macwhisper-proxy"
launchctl kickstart -k "gui/$(id -u)/com.simon.superwhisper-api.macwhisper-proxy"
```

Check or unload it:

```bash
curl http://127.0.0.1:8766/health
launchctl print "gui/$(id -u)/com.simon.superwhisper-api.macwhisper-proxy"
launchctl bootout "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.simon.superwhisper-api.macwhisper-proxy.plist
```

### MacWhisper Global Replace (agent-driven)

Grow MacWhisper's Global Replace dictionary by handing context to a CLI agent.
The tool is intentionally "dumb": `learn` transcribes your recordings with both a
fast local model (`mw`) and the accurate cloud model (Scribe) and prints both
transcripts plus your current replacements as JSON. The agent reads that,
proposes `original -> replacement` pairs in chat, and applies the approved ones.

```bash
# 1. Hand the agent context for the latest recording (transcribes with both models)
uv run superwhisper-macwhisper learn
uv run superwhisper-macwhisper learn --latest 5
uv run superwhisper-macwhisper learn /path/to/audio.m4a   # or a file you point it at

# ...the agent reads both transcripts + current dictionary and proposes pairs...

# 2. Apply what you approve (a JSON array; existing originals are updated in place)
uv run superwhisper-macwhisper apply \
  '[{"original":"Deep Gram","replacement":"Deepgram"},
    {"original":"Qwen 3. 5","replacement":"qwen3.5"}]'

# Undo entries added by mistake (JSON array of originals)
uv run superwhisper-macwhisper remove '["Deep Gram","Qwen 3. 5"]'
```

Model and MacWhisper paths are constants at the top of
`src/superwhisper_api/macwhisper/replacements.py` — edit them there.

The GUI orchestration paths are archived: the original under `archive/gui-orchestration/`
and the later batch/worker rewrite under `archive/gui-batch/`. The only remaining GUI tool
is `superwhisper-new-model-inspect`, which probes Superwhisper for newly available models.

## Response Parsing

Superwhisper's model proxy returns server-sent events. The client first joins
those transport chunks into the model's final text. If you want structured data,
call `generate_json(...)`; it parses that final text as JSON and validates it
locally when you pass a schema.

## Notes

The archived GUI path is reference-only. New audio transcription should use
`superwhisper-audio`; new model work should use `SuperwhisperClient`.

## Development

```bash
uv run ruff check .
uv run superwhisper-audio --help
```

Tests call real Superwhisper/provider endpoints through the current cache and credentials:

```bash
uv run pytest
```
