"""MacWhisper integration: a local transcription proxy and a Global Replace helper.

Two entry points:

- ``superwhisper-macwhisper-proxy`` (proxy.py) — local OpenAI-compatible
  transcription proxy; launched at login via launchd.
- ``superwhisper-macwhisper`` (cli.py) — Global Replace helper with ``learn``
  and ``apply`` subcommands, driven by a CLI agent.
"""
