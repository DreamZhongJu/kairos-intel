#!/bin/sh
set -eu

mkdir -p /root/.agent-reach /root/.config/mcporter /root/.config/yt-dlp

# Recent YouTube pages require a JavaScript runtime for reliable extraction.
# Node is installed in the image; persist this small yt-dlp setting with the
# other Agent-Reach runtime configuration.
if ! grep -qxF -- '--js-runtimes node' /root/.config/yt-dlp/config 2>/dev/null; then
  printf '%s\n' '--js-runtimes node' >> /root/.config/yt-dlp/config
fi

# Exa is Agent-Reach's no-key semantic-search backend.  Keep the configuration
# in the mounted volume, so that container rebuilds do not lose it.
if ! mcporter config list --json 2>/dev/null | grep -q '"exa"'; then
  mcporter config add exa https://mcp.exa.ai/mcp --scope home >/dev/null 2>&1 || true
fi

exec python app.py
