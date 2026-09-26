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
  version: "1.3"
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
3. **Downloads and installs:** as of 2026-09-26 the user granted full use of the
   Mac Studio (downloads, installs, all memory and compute) without asking each
   time. Another user's long-running process (DeepLabCut, `ioannaporfyri`) shares
   the machine: never touch it.
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

## Lessons from running many jobs (2026-09-26)

- **Schedule by declared memory, not free memory.** Free-memory gates let two 32B
  models and two smaller ones load together (about 170 GB of weights on 128 GB) and
  the machine thrashed (one couplet in 12 minutes). Use the scheduler
  `experiments/couplet_routes/jobs/runner.sh` (queue `~/amu_jobs/queue.txt`, one
  line per job: `<name> <GB> <command>`, budget 108 GB) and the watcher
  `scripts/wait_runner.sh`. bf16 weights plus overhead: 1.7B 6, 4B 12, 8B 20, 14B 34,
  32B 72-76, Gemma 12B 28, Gemma 27B 60 GB. Register processes started outside the
  runner in `~/amu_jobs/running/<name>.{pid,mem}`.
- **The GPU saturates before memory does.** With batch-size-1 scripts the GPU shows
  100% busy; more parallel jobs do not help. Batch inside scripts instead (see the
  `state-edit-experiments` skill).
- **Remote shell quirks.** The login shell is zsh and `bash` is 3.2 (no associative
  arrays). Multi-line or heavily quoted commands break inside `run.sh`: put the job in
  a script file in the bundle (for example `experiments/couplet_routes/jobs/*.sh`) and
  run it by path. The remote Python is 3.9.
- **The bundle filter rejects any path containing `token`** (a credential guard):
  name scripts accordingly. Pipe `push_bundle.sh` output only with `set -o pipefail`,
  otherwise a refused push looks like success and the job runs without its script.
- **Newer architectures need the second environment.** The main venv is Python 3.9 with
  transformers 4.57 (no `qwen3_5`, Qwen3.5's hybrid Gated DeltaNet). `~/amu_jobs/.venv-next`
  (Python 3.12 via `~/.local/bin/uv`, transformers 5.x) runs them; call its python
  explicitly in the job script (see `experiments/recurrent_carry/run_pilot.sh`). On
  Apple GPUs the recurrent layers use slow reference kernels (flash-linear-attention is
  CUDA-only).
- **Correct a running job's declared memory** by editing
  `~/amu_jobs/running/<name>.mem` when the first guess was too high; the runner re-reads
  it every cycle.
- **Never allowlist a directory that holds results.** Pushing it copies local
  (older) result files over newer remote ones; on 2026-09-26 this silently destroyed a
  rerun. `push_bundle.sh` now skips `results/` inside listed directories; list scripts
  individually, and list a result file by name only when a job needs it as input.
- **Test on a model name no experiment uses** (for example `Qwen3-0.6B`): a smoke
  test under a real model name overwrites that model's remote results, and the next
  sync copies them back over committed files.
- **Watchers must run as tracked background tasks** (the harness's background mode),
  not as detached `( … &)` subshells, or nobody is notified when they finish.
- **Keep both machines awake for long runs:** `caffeinate -i -s -t <seconds>` locally
  (as a tracked background task) and as a job on the Mac Studio.
- **Check a resumed chain's inputs exist** before resuming from a later step (a
  32B-plain chain was resumed at step 3/4 although its first steps had never run).

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
