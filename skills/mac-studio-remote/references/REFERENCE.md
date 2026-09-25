# mac-studio-remote — reference

No addresses, account names, key names or passwords belong in this skill. They
live only in the user's local `~/.ssh/config` and VPN profile. Everything below
uses placeholders.

## Local setup (done once by the user)

1. **Split-tunnel VPN.** The university VPN profile is copied and two lines are
   added after `dev tun`, so only the Mac Studio is routed through the tunnel:

   ```
   route-nopull
   route <MAC-STUDIO-IP> 255.255.255.255
   ```

   `route-nopull` ignores every route and DNS setting the server pushes (the
   default profile redirects all traffic). The user imports the copy into their
   VPN client and connects. Agents must not change VPN or network settings.
2. **SSH key.** `ssh-keygen -t ed25519 -f ~/.ssh/<key-name>`, then
   `ssh-copy-id -i ~/.ssh/<key-name>.pub <user>@<MAC-STUDIO-IP>` (asks for the
   account password once). If the key has a passphrase, unlock it into the agent
   and Keychain with `ssh-add --apple-use-keychain ~/.ssh/<key-name>`.
3. **Host alias** in `~/.ssh/config` (the scripts use `macstudio`; override with
   `MACSTUDIO_HOST`):

   ```
   Host macstudio
     HostName <MAC-STUDIO-IP>
     User <account short name, lowercase>
     IdentityFile ~/.ssh/<key-name>
     UseKeychain yes
     AddKeysToAgent yes
   ```

## Remote layout

Everything lives under `~/amu_jobs/` on the Mac Studio (override with
`MACSTUDIO_ROOT`):

| Path | Contents |
|---|---|
| `.venv/` | Python venv from the system `python3` (created by `setup_env.sh`) |
| `hf_cache/` | `HF_HOME`: model downloads |
| `bundle/` | Allowlisted scripts and small data, same relative paths as the repo |
| `logs/<name>.log`, `logs/<name>.pid` | Job output and process id (`run.sh`) |

## Scripts

| Script | Does | Side effects |
|---|---|---|
| `check.sh` | Verifies the Mac Studio route is on the VPN interface and the internet route is not; tests key login; prints remote hardware, free disk, venv status, recent logs | None |
| `setup_env.sh --approved-by-user` | Creates the venv; installs `torch`, `transformers>=4.51`, `accelerate`, `safetensors`, `pandas`, `numpy` from PyPI (override with `MACSTUDIO_PKGS`) | Installs software remotely: needs user approval |
| `push_bundle.sh <allowlist>` | Expands the allowlist, refuses forbidden paths (git, manuscript, `docs/`, `AGENTS.md`, submissions, credentials, VPN profiles, keys), prints every file and the total, refuses bundles over 100 MB (`ALLOW_LARGE=1` overrides), rsyncs to `bundle/` | Writes remote files |
| `run.sh <name> <cmd...>` | Starts `<cmd>` under `nohup` in `bundle/` with the venv on `PATH` and `HF_HOME` set; refuses to start a second copy of a running job | Starts a remote process |
| `status.sh <name> [lines]` | Running or not, plus the last log lines (warnings filtered) | None |
| `fetch.sh [--allow-large] <path>...` | Copies `bundle/<path>` back to the same repo path; refuses files over 50 MB | Writes local files |

## Practical notes

- Estimate each job's duration and set a timer before polling; every poll is a
  slow round trip through the VPN.
- AMU experiment stages checkpoint per layer or per item, so after a dropped
  connection or a crash, re-running the same `run.sh` command resumes.
- Large Hugging Face files: run `skills/hf-range-streaming` on the Mac Studio
  (put `skills/hf-range-streaming/scripts/hf_stream.py` in the allowlist). It
  resolves each file once and sends range requests straight to the CDN, avoiding
  Hugging Face's resolver rate limit.
- Fetch only small outputs (summaries, row files). Keep feature caches and
  captured activations remote.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `check.sh`: Mac Studio route not on `utun*` | VPN not connected, or full profile in use | Ask the user to connect the split-tunnel profile |
| Internet route also on `utun*` | Full-tunnel profile | Ask the user to switch to the split-tunnel profile |
| `Connection closed by authenticating user` | Key has a passphrase and is not in the agent | Ask the user to run `ssh-add --apple-use-keychain <IdentityFile>` |
| `Permission denied (publickey…)` although the key was copied | Key not accepted by the server | The user can run a one-off debug server on the Mac Studio (`sudo /usr/sbin/sshd -d -p 2222`) while you connect once with `ssh -p 2222 macstudio true`; its output names the cause |
| `venv missing` from `run.sh` | Environment not set up | Ask the user to approve `setup_env.sh` |
