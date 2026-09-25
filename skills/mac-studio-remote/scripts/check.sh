#!/usr/bin/env bash
# Check the split-tunnel route, SSH key login, and the remote environment. Read-only.
source "$(dirname "$0")/_common.sh"
ip=$(ssh -G "$HOST" 2>/dev/null | awk '/^hostname /{print $2}')
[ -n "$ip" ] || die "no SSH host '$HOST' in ~/.ssh/config (see references/REFERENCE.md)"
iface=$(route -n get "$ip" 2>/dev/null | awk '/interface:/{print $2}')
inet=$(route -n get 8.8.8.8 2>/dev/null | awk '/interface:/{print $2}')
echo "route to Mac Studio: ${iface:-none}   route to internet: ${inet:-none}"
case "$iface" in utun*) ;; *) die "Mac Studio is not routed through the VPN (${iface:-none}). Ask the user to connect the split-tunnel VPN profile for the Mac Studio." ;; esac
case "$inet" in utun*) echo "warning: internet traffic is also going through the VPN (full tunnel); transfers will be slow." ;; esac
if ! out=$("${SSH[@]}" 'echo ok' 2>&1); then
  echo "$out" >&2
  key=$(ssh -G "$HOST" 2>/dev/null | awk '/^identityfile /{print $2; exit}')
  grep -q "closed by authenticating" <<<"$out" && die "SSH key not in the agent. Ask the user to run: ssh-add --apple-use-keychain ${key:-<IdentityFile of $HOST>}"
  die "SSH login failed (see above)"
fi
"${SSH[@]}" "echo \"host: \$(hostname -s)  \$(sysctl -n machdep.cpu.brand_string)  \$((\$(sysctl -n hw.memsize)/1073741824)) GB  macOS \$(sw_vers -productVersion)\";
  df -h ~ | awk 'NR==2{print \"free disk: \" \$4}';
  if [ -x ~/$REMOTE_ROOT/.venv/bin/python ]; then ~/$REMOTE_ROOT/.venv/bin/python -c 'import torch,transformers;print(\"venv: torch\",torch.__version__,\"mps\",torch.backends.mps.is_available(),\"transformers\",transformers.__version__)' 2>&1 | tail -1; else echo 'venv: not created (run setup_env.sh with user approval)'; fi;
  ls ~/$REMOTE_ROOT/logs 2>/dev/null | sed 's/^/log: /' | tail -5"
