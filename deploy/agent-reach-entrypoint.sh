#!/bin/sh
set -eu

mkdir -p /root/.agent-reach /root/.config/mcporter

# Exa is Agent-Reach's no-key semantic-search backend.  Keep the configuration
# in the mounted volume, so that container rebuilds do not lose it.
if ! mcporter config list --json 2>/dev/null | grep -q '"exa"'; then
  mcporter config add exa https://mcp.exa.ai/mcp --scope home >/dev/null 2>&1 || true
fi

exec python app.py
