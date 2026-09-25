---
name: mac-studio-remote
description: >-
  Runs AMU experiment scripts on the remote Mac Studio (a large-memory Apple Silicon machine) over SSH through a split-tunnel
  VPN: checks connectivity, sends an allowlisted bundle of scripts and small data (never the codebase), launches
  detached jobs, polls status, and fetches small results back. Use when an experiment needs more memory than the
  local M1 (for example Qwen3-8B or 14B), or when the user asks to use the Mac Studio.
license: MIT
compatibility: >-
  Local macOS with an SSH host alias "macstudio" in ~/.ssh/config, its key unlocked in the SSH agent, and the
  user's split-tunnel VPN connected. Remote: macOS with system python3; rsync on both sides.
metadata:
  version: "1.0"
---

# Mac Studio remote jobs

Use the Mac Studio as a compute box: scripts and small data go over, jobs run
detached there, the Mac Studio downloads models itself, and only small results
come back. Details, measured facts and troubleshooting are in
[`references/REFERENCE.md`](references/REFERENCE.md).

## Rules (from the user)

1. **Never copy the codebase.** The Mac Studio is considered insecure. Send only
   scripts and data listed in an allowlist; never the git history, the
   manuscript (`paper.qmd`, `manuscript/`), `AGENTS.md`, `docs/`, other
   experiments' results, or credentials. `scripts/push_bundle.sh` enforces this.
2. **The link from this laptop is slow** (VPN abroad). Keep transfers small; let
   the Mac Studio download models and transcoder data over its own connection.
3. **Ask the user before installing or downloading anything on the Mac Studio.**
4. **The user controls the VPN and all network settings.** Never change them.
   Agents cannot type passwords or passphrases.

## When you are done

- Results are fetched into the repo on this laptop and committed there.
- Nothing outside `~/amu_jobs/` was created on the Mac Studio.
- You reported what ran, where the logs are, and how long it took.

## Procedure

1. **Check the connection.** It verifies the split-tunnel route, the SSH key,
   and the remote environment:

   ```bash
   skills/mac-studio-remote/scripts/check.sh
   ```

   If it reports that the VPN route is missing, ask the user to connect the
   split-tunnel VPN profile for the Mac Studio. If SSH fails with "Connection closed by
   authenticating user", the key is not in the agent: ask the user to run
   `ssh-add --apple-use-keychain <the IdentityFile of the macstudio alias>`.
2. **Set up the environment once, with user approval.** It creates
   `~/amu_jobs/.venv` from the system Python and installs the packages from
   PyPI:

   ```bash
   skills/mac-studio-remote/scripts/setup_env.sh --approved-by-user
   ```

3. **Write an allowlist** of repository-relative paths the job needs, one per
   line, following [`assets/couplets_allowlist.txt`](assets/couplets_allowlist.txt).
   Then send it (it prints every file and the total size, and refuses forbidden
   paths):

   ```bash
   skills/mac-studio-remote/scripts/push_bundle.sh skills/mac-studio-remote/assets/couplets_allowlist.txt
   ```

4. **Launch a detached job.** It runs in `~/amu_jobs/bundle` with the venv
   activated and `HF_HOME=~/amu_jobs/hf_cache`, and logs to
   `~/amu_jobs/logs/<name>.log`:

   ```bash
   skills/mac-studio-remote/scripts/run.sh couplets-8b-step2 python -u experiments/couplet_routes/step2_state_edit.py Qwen3-8B
   ```

5. **Poll sparingly.** Estimate the run time first and set a timer; polling
   costs a slow round trip:

   ```bash
   skills/mac-studio-remote/scripts/status.sh couplets-8b-step2
   ```

   To be notified when a job ends, run the wait-and-fetch helper in the
   background on the laptop. It polls every 5 minutes (`POLL=<seconds>` to
   change), then fetches the listed result paths and exits, non-zero if the job
   failed:

   ```bash
   skills/mac-studio-remote/scripts/wait_fetch.sh couplets-8b-step2 experiments/couplet_routes/results/Qwen3-8B
   ```

   For a long multi-step job, `wait_file.sh <job> <marker file> <paths>...` waits
   for one step's output file instead, so results arrive step by step.

6. **Fetch small results back** into the repo (repository-relative paths;
   files over 50 MB are refused unless `--allow-large`):

   ```bash
   skills/mac-studio-remote/scripts/fetch.sh experiments/couplet_routes/results/Qwen3-8B
   ```

   Then review, document in the experiment README, and commit as usual.

## Edge cases

- Jobs survive disconnects: they run under `nohup`, and every AMU stage
  checkpoints, so re-running the same `run.sh` command resumes.
- Intermediates such as feature cards and captured activations stay on the Mac
  Studio. Fetch only what the experiment README reports.
- For large Hugging Face files, use `skills/hf-range-streaming` on the Mac Studio
  (include its script in the allowlist). Never route downloads through this
  laptop.
- Cleanup (`rm -rf ~/amu_jobs` on the Mac Studio) is a permanent deletion: ask
  the user first.
