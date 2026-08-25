# Paper experiment plan (PT + affine stack)

> **Historical (frozen 2026-08-07).** This was the execution contract for
> E1–E6 (wrapper-like vs compiled-trajectory interventions). Those runs
> are complete. The current paper claim is generated-token mediation;
> source of truth is [`paper.qmd`](../paper.qmd) and
> [`manuscript/sections/`](../manuscript/sections/). Do not treat this
> file as a live pipeline.

Last updated: 2026-08-07  
Model stack: `google/gemma-3-270m` + Gemma Scope 2 PT CLT `clt/width_262k_l0_medium_affine`  
Env: `/Users/anthony/miniconda3/bin/python`

**Stack decision:** Primary runs are **pretrained + affine**, not IT. Latent Planning Appendix J found base models slightly better on a/an than IT; the PT clincher already produced a clean Outcome-B signal. IT + non-affine was abandoned as the primary stack (Hub IT `medium_affine` layer-0 is empty; IT is a weaker instrument for this task).

This was the execution contract for completing the E1–E6 suite.

---

## Scientific fork (outcome-robust)

| Outcome | Claim |
| --- | --- |
| **A — Modular / content-specific planning** | Sparse features can move the article while locking future content identity above controls. |
| **B — Compiled trajectories** | Features that look like “planning” select coherent multi-token packages; article and content move together. |

Prior PT clincher favored **B**. Either ending is publishable if E1–E4 + E6 complete.

---

## Already done (do not re-prove as foundation)

1. Ophthalmologist LoF pair repair on source prompt (PT).
2. Fixed-pair generalization → class switching, not content-preserving transfer (PT).
3. Planning GoF clincher → trajectory path (~95% class shift, ~5% content preserve) (PT).

Cite Latent Planning Appendix J for choosing PT over IT on a/an.

---

## Shared protocol (all new experiments)

- **Model / transcoder / backend / dtype:** from each `config.json` (PT affine defaults).
- **Prompts:** Latent-Planning-style occupation completions unless noted.
- **Selection set:** 8 expected-`an` occupations (same sentences as clincher), disjoint from test.
- **Held-out test:** 20 occupations including twin pairs (clincher list), unless an experiment subsets.
- **Amplify default:** \(5\times\) prompt-specific activation unless dose-sweeping.
- **Controls:** activation-matched random active features; same op and factor.
- **Primary scores (always report):**
  - \(\Delta(\texttt{an}-\texttt{a})\)
  - article changed?
  - content preserved? (listed / baseline noun or first token)
  - class shifted? (consonant↔vowel package)
  - twin match?
  - wrapper-like vs trajectory-like rates
  - illicit mismatch rate (`an`+consonant or `a`+vowel) where relevant
- **Deliverables per experiment:** `config.json`, `run.py`, `README.md`, `results/{summary.json,report.md,...}`

---

## E1 — Selection-criterion ablation *(required; first)*

**Question:** Does *how* we pick features determine whether interventions look like wrappers or trajectories?

**Feature sets (freeze after selection; same count \(k=4\) when possible):**

| ID | Rule |
| --- | --- |
| **S1 Dual-effect** | \(+\mathrm{attr}(\texttt{an})\) and \(+\mathrm{attr}(\text{future token})\) |
| **S2 Article-only** | \(+\mathrm{attr}(\texttt{an})\), near-zero \(\lvert\mathrm{attr}(\text{future})\rvert\) |
| **S3 Content-only** | \(+\mathrm{attr}(\text{future})\), near-zero \(\lvert\mathrm{attr}(\texttt{an})\rvert\) |
| **S4 Competing / a-favoring** | \(+\mathrm{attr}(\texttt{a}-\texttt{an})\), low \(\lvert\mathrm{attr}(\text{future})\rvert\) |

**Do:** Build article + future graphs on 8 selection prompts → nominate recurring features per rule → amplify each frozen set at \(5\times\) on 20 held-out prompts vs random controls.

**Conditional:** If recurrence cannot fill \(k=4\), lower recurrence / take top-k available and record fallback.

**Expected (prior):** S1 trajectory-like; no high wrapper-like rate.

**Dir:** `experiments/selection_criterion_ablation/`

---

## E2 — Dose–response *(required)*

**Question:** Is there a dose window for content-preserving article move, or does gain scale as package switch?

**Do:** Amplify **S1** (and any E1 set with wrapper-like ≥ 0.25 **or** strong article effect) at `{1.5×, 2×, 3×, 5×, 8×}` on the same 20 prompts + controls. Same scores as E1.

**Conditional:** Sweep extra sets only per rule above; do not dose-sweep null sets.

**Expected:** Monotone class-switch with dose; no wrapper window.

**Dir:** `experiments/planning_dose_response/`

---

## E3 — Forced content-lock / dual intervention *(required; improved logic lives here)*

**This is where we selectively intervene on preparation vs content and score both axes jointly.**

| Condition | Intervention |
| --- | --- |
| **C0** | Baseline |
| **C1** | Article-push only (best article-moving set from E1, usually S1) |
| **C2** | Content-lock only (amplify content-supporting / S3-style features for baseline noun) |
| **C3** | **Dual: C1 + C2 simultaneously** |
| **C4** | Dual opposite / illicit attempt: push toward `an` while locking consonant baseline *(conditional)* |
| **C5** | Matched random controls (single and dual) |

**Decision:**
- **A:** C3 or C4 flips article **with** content preserve above controls and above C1 alone.
- **B:** C1 class-switches; C3 re-bundles / lock wins / push wins as a package; illicit mismatch ≈ 0.

**Conditional:** Full C4 on all 20 only if C3 shows partial dissociation (content preserve ≥ 0.2 with article move, or article stuck under lock). Else C4 = short confirmation on ≥5 twin prompts.

**Dir:** `experiments/forced_content_lock/`

---

## E4 — Causal tetrad on twin families *(required)*

**Do:** On 2 twin families (e.g. pilot/aviator, lawyer/attorney — refine from E1/E2 behavior):

1. LoF (zero best set)
2. GoF (amplify)
3. Rescue (LoF then restore / opposite GoF)
4. Specificity (matched random)

Score package membership (baseline vs twin vs other), not article alone.

**Conditional:** If twins unstable across doses, use the E2 dose where twin match peaked.

**Dir:** `experiments/trajectory_causal_tetrad/`

---

## E5 — Domain transfer *(required slim; last)*

**Do:** Freeze winning set(s) from E1/E3; test on ≥15 non-occupation `a`/`an` nouns (animals / instruments / nationalities). Cite occupation held-out from earlier; do not rerun unless protocol changes.

**Conditional:** If transfer is nonspecific (control-like), stop and report locality. No third domain unless transfer is clean. Optional `is`/`are` only if time remains after E1–E4 + E6.

**Dir:** `experiments/domain_transfer_aan/`

---

## E6 — Reclassify ophthalmologist *(required; after E3 protocol exists)*

**Do:** On source mismatch prompt only, apply E3-style conditions (LoF pair if recoverable, content-lock, dual, GoF S1). Classify: true wrapper repair vs package coincidence.

**Dir:** `experiments/ophthalmologist_it_reclassify/` (directory name historical; runs on **PT**)

---

## Execution order and branches

```text
E1 → always
E2 → always on S1; also on E1 sets with wrapper≥0.25 or strong article effect
E3 → always (C0–C3, C5); C4 full iff C3 partial dissociation
E4 → always (2 twin families; dose from E2 if needed)
E6 → always after E3 protocol
E5 → slim non-occupation last; secondary task only if time
```

---

## Where the improved logic is

| Idea | Location |
| --- | --- |
| Multiple selection rules (not only “suppress bad `a`”) | **E1** |
| Separate article vs content features | **E1 S2/S3**; **E3 C1 vs C2** |
| Intervene on both axes at once | **E3 C3 / C4** ← primary |
| Content preservation as success criterion | Scoring in **all**; decision in **E3** |
| Causality beyond one-shot GoF | **E4** |
| Ophthalmologist not the foundation | **E6** after E3 |

---

## Paper endings

- **A:** Content-specific planning is dissociable; dual-effect selection confounds planning with package features.
- **B:** Sparse features implement compiled trajectories; article metrics alone misread latent planning.

Meta-result either way: **article movement under sparse intervention is not evidence of content-preserving latent planning** unless content identity is locked and reported.
