#!/usr/bin/env bash
# Create ~/amu_jobs/.venv on the Mac Studio from the system python3 and install packages from PyPI.
# Installs software on the remote machine: run only after the user has approved it.
source "$(dirname "$0")/_common.sh"
[ "${1:-}" = "--approved-by-user" ] || die "installing on the Mac Studio needs the user's approval; re-run with --approved-by-user once they have agreed"
PKGS="${MACSTUDIO_PKGS:-torch transformers>=4.51 accelerate safetensors pandas numpy}"
"${SSH[@]}" "set -e; mkdir -p ~/$REMOTE_ROOT/{bundle,logs,hf_cache};
  [ -x ~/$REMOTE_ROOT/.venv/bin/python ] || /usr/bin/python3 -m venv ~/$REMOTE_ROOT/.venv;
  ~/$REMOTE_ROOT/.venv/bin/python -m pip install -q --upgrade pip;
  ~/$REMOTE_ROOT/.venv/bin/python -m pip install -q $(printf "'%s' " $PKGS);
  ~/$REMOTE_ROOT/.venv/bin/python -c 'import torch,transformers,pandas;print(\"ok torch\",torch.__version__,\"mps\",torch.backends.mps.is_available(),\"transformers\",transformers.__version__)'"
