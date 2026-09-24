# Handoff: ICLR review and mentoring decisions (2026-09-24)

Context carried over from a separate cloud chat session. That session saw only the v35 PDF
(`neurips-planning-trajectories-Gemma-3-270M-v35-submitted.pdf`) and the texts pasted by the author,
not this repository or its artifacts. Treat it as reviewer/mentor input, not as instructions that
override AGENTS.md.

## 1. Independent ICLR-style review of v35 (summary)

- Rating: 4 (marginally below acceptance); confidence 4.
- Main reasons:
  1. The six-cell decomposition is standard mediation analysis (indirect-effect contrast plus a
     controlled direct effect at the treated mediator); the identity and Eq. 1 are algebraic. Value
     rests on the empirical findings.
  2. The findings sit in one narrow setting: two small models, one binary grammatical mediator
     (a/an), synthetic prompts, 20 held-out S1 prompts, 14 families.
- Corrections accepted after consolidation:
  - The S1 result is *not* guaranteed by grammar. The 0.028-TV remainder, replay, rescue and
    boundary results are real evidence. The valid objection is the joint article+noun selection
    rule.
  - Conditioning the routing effect on article-policy movement would remove the mechanism being
    measured. Do not treat it as a control.
- Other weaknesses: heavy coined terminology; the aggregate R^2 = 0.840 gain fit is unstable
  (crossed CI [-0.293, 0.986], no transfer to 1B); related work should add path patching / IOI,
  CoT faithfulness via token forcing (e.g., Lanham et al.), and formal mediation terms (NDE/CDE);
  NeurIPS format and checklist need converting for ICLR; no anonymized code yet.

## 2. Consolidated plan (agreed with the implementation agent)

1. Essential: apply the six-cell assay to an independently selected future-token intervention
   (no article criterion in selection).
2. Essential for the broad main-track claim: reproduce mediator-relative routing with one
   meaningful non-article mediator, starting with a route-blind feasibility screen and a stop rule.
3. Supporting: demote the gain fit to a limited diagnostic; build an anonymized verification
   package.

Submission gate: keep main-track framing only if both 1 and 2 pass. Otherwise write a focused
methodological case study and remove the general claims about "organization of causal state" and
direct CoT/tool-use implications.

## 3. Provenance facts disclosed by the implementation agent (must appear in the paper)

- The S1 5x dose was chosen after inspecting dose behavior on the same 20 occupations later used
  for the six-cell analysis.
- The S3 set (future-attributed, near-zero article attribution) changed content on only 1/20
  prompts at 5x.
- Ranked by future attribution alone, the 32-feature calibration set has no greedy article switches
  and TV <= 0.030 in the top 8; the first large effect (L13/F10231, 0.893 TV) is at rank 12 and also
  switches the article. Future attribution vs free-noun TV: Spearman rho = -0.122, p = 0.503.
- The matched-triad study was frozen before its own screen but *after* the independent-family route
  result had been seen.
- The Figure 4 recomputation was written by an AI agent. That makes it an independent code path,
  not an independent human audit.
- Lexical-gap overlap between cross- and within-class arms is poor (only 3 cross arms fall inside
  the within-class range).

## 4. Mentor decisions

### D1: Priority 1 selection rule
Use development-only free-generation future-noun efficacy. It is biased toward article movers, but
that population is exactly the effects people would call "planning."

Protocol:
- Candidate pool: all pre-article features in the attribution graphs; no article score anywhere.
- Three-way occupation split: development / selection / untouched route test set.
- Fix the dose on development prompts.
- Freeze the top K (10-16) plus a pre-specified random mid-rank sample, and keep the failures.
- On the test set, report six cells, mediator support, and replay/rescue for every frozen handle.

Pre-registered readings:
- Mostly public: a confirmatory version of the S1 claim.
- Any private or hybrid handle: a new natural sparse private carrier.
- No large effects: report it and rely on the case-study framing.

If the pinned Hanna-Ameisen commit contains their feature-selection code, also apply their exact
rule.

### D2: second mediator at 1B or below
No published task is known to show causal planning at this scale (Ma & Rui find the causal rhyme
handoff only in Gemma 3 27B). Ranked candidates:
1. German der/die/das or French le/la gender. Still grammatical, but lexical and arbitrary with
   respect to meaning rather than phonological, and three-valued in German. Defeats "special to
   English a/an." Cheap screen.
2. Multiple-choice answer letter with options listed in context (risk: a coding-rule objection).
3. A written bridge entity in two-hop recall (likely fails the behavioral gate).

Each candidate gets a route-blind screen first: behavior, mediator mass >= 0.8, number of
admissible families, and a stop rule set in advance.

### D3: larger compute
Yes, if aiming for the main track. The strongest single result would be the assay applied to the
published Gemma 3 27B rhyme handoff (Ma & Rui). Check whether NDIF remote NNsight hosts a suitable
model; otherwise rent one 80 GB GPU briefly (full-residual patching only, no CLT needed).

### D4: minimum result that changes a reviewer's verdict (4 -> 6)
Either:
- (A) the assay applied to a published planning intervention using its original selection rule; or
- (B) the Priority 1 result plus mediator-relative routing reproduced with a non-article mediator.

Without either, target TMLR or a mechanistic-interpretability workshop.

## 5. Order of work
1. Attempt ledger over the ~55 experiment directories (classify each as attempt, control, repair or
   audit), plus a history-free anonymized export.
2. Fresh held-out S1 replication at the frozen 5x dose, on occupations nobody has inspected.
3. Priority 1 using the D1 protocol (restore the 270M CLT cache; archive the 1B cache if disk
   requires it).
4. Gender-mediator feasibility screen, then the full matched assay only if it passes.
5. In parallel: check access to larger compute for the 27B rhyme handoff.
6. Frozen-row audit of at least 3 headline numbers by a human or a separately prompted agent.

## 6. Open questions for the implementation agent
1. Does the pinned Hanna-Ameisen commit contain a runnable feature-selection rule for Gemma 3
   270M/1B, or only prompts?
2. How many candidate features exist at the pre-article position across the attribution graphs?
   Is there enough for a K-of-N selection not dominated by the 32 already-assayed features?
