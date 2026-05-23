# Super Whisperer CLI Notes

- Use normal UV project commands here: `uv run super-whisperer ...`, `uv run ruff check .`, etc.
- Do not add Hugging Face dataset code, benchmark layers, or transcription backends unless explicitly requested.
- Keep the CLI centered on the working Superwhisper path: open local audio files with Superwhisper, observe the transcript rows Superwhisper writes, and route those results to configured outputs.
- Assume the currently selected Superwhisper mode is the mode used for transcription. Do not try to switch modes programmatically until that is proven separately.
