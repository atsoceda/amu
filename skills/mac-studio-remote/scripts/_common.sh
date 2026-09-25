#!/usr/bin/env bash
# Shared settings for the mac-studio-remote scripts.
set -euo pipefail
HOST="${MACSTUDIO_HOST:-macstudio}"          # SSH alias from ~/.ssh/config
REMOTE_ROOT="${MACSTUDIO_ROOT:-amu_jobs}"    # relative to the remote home directory
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 "$HOST")
die() { echo "error: $*" >&2; exit 1; }
