# Pre-registered analysis plan

**Study:** Do AI-tuned and AI-generated SAT solvers generalise beyond the instances they were
developed on? A paired, exposure-stratified evaluation with semantics-preserving perturbations.

**Status:** to be registered on OSF *before* the main measurement starts. Pilot data (8 instances,
used only to test the tooling and to motivate the separate perturbation types) are excluded from
every confirmatory analysis. Deviations from this plan will be listed in the paper with reasons.

## 1. Research questions and hypotheses

Notation: for a unit (base B, AI variant A) and an instance *i*, *d_i* = PAR-2(B, i) − PAR-2(A, i).
The paired effect size is the matched-pairs rank-biserial correlation *r* of the *d_i*; positive
values favour the AI variant. *Seen* = instances the AI variant's developers could have used
(`aesat_train`, `comp2425`; see docs/exposure.md). *Unseen* = `clean2026`.

- **RQ1 (generalisation gap).** Is the AI variant's advantage over its base smaller on unseen
  than on seen instances?
  - H1: for each primary unit, *r*(seen) − *r*(unseen) > 0.
- **RQ2 (presentation robustness).** Does the AI variant lose more than its base under
  variable/clause/literal reordering (`s1-rcl`), and under polarity flipping (`s1-f`)?
  - H2a/H2b: the AI-variant slowdown, log(T(A, perturbed) / T(A, original)), exceeds the base's,
    paired by instance. Tested separately for rcl and f.
- **RQ3 (memorisation).** LymphoSAT (trained on 80 % of every GBD family) has a larger advantage
  over its fallback on GBD instances from before 2026 than on 2026 instances, and that advantage
  shrinks under reordering.
  - H3 is H1 + H2a applied to the unit (kissat-4.0.2, lymphosat). Reported separately because the
    exposure definition differs.
- **RQ4 (correctness).** Do AI-derived solvers return wrong answers (models that falsify a clause,
  or answers contradicting a verified result) more often than human-designed ones?
  - Descriptive. Every WRONG answer is reported with its instance and log.
- **RQ5 (aggregate).** Across all primary units, is the advantage smaller on unseen instances?
  - Random-effects meta-analysis of the per-unit differences in *r*.

## 2. Design (fixed in study/design.toml)

- **Units:** 5 primary (AI variant vs declared base), 1 secondary and 1 reference. Each block runs
  the two solvers of a unit concurrently on one host, one physical core each, with the core
  assignment randomised. Blocks are assigned to hosts at random (`packing_seed`).
- **Instances:** clean2026 (n = 240, ≤ 12 per family); aesat_train (60); comp2425 (120, ≤ 6 per
  family); legacy (54, 6 per family from the 9 industrial families of the author's thesis).
  Selection is by keyed hash rank (`selection_seed`) from a GBD snapshot whose SHA-256 is recorded.
  2026 instances isomorphic (equal GBD isohash) to any earlier instance are excluded.
- **Variants:** original; `s1-rcl` (rename + shuffle clauses and literals); `s1-f` (polarity flip).
  Variants apply to primary units on a fixed 25 % subsample of clean2026, aesat_train and comp2425.
  Each perturbed file is checked: isohash equal, gbdhash different.
- **Limits:** 5000 s CPU time, 7000 MB resident memory, measured by the harness (VmHWM), not by
  ulimit. Solvers run unmodified via their competition `run.sh`, proofs generated and discarded.
- **Hosts:** GitHub-hosted runners, 2 physical cores per job. A calibration run (kissat-sc2026 on a
  fixed instance, both cores) brackets every job.

## 3. Outcomes

- **Primary:** PAR-2 per run: CPU time if solved, 10 000 s otherwise. Unsolved includes TIMEOUT,
  MEMOUT, CRASH, ERROR and WRONG.
- **Secondary:** solved status (paired, McNemar); restricted mean CPU time to 5000 s
  (Kaplan–Meier, descriptive only, because censoring is informative); CPU-time ratio on instances
  solved by both.

## 4. Exclusions (decided before seeing outcomes)

1. **Host CPU model:** the confirmatory analysis uses blocks run on the majority CPU model.
   Blocks on other models are re-run if possible; otherwise they are reported in a sensitivity
   analysis.
2. **Host speed:** a job whose calibration time (mean of the two cores, start and end) deviates
   more than 25 % from the median over all jobs, or whose start/end calibrations differ more than
   25 %, is re-run. If that is impossible, its blocks are excluded and reported.
3. **Harness failures (ERROR):** re-run. Solver failures (CRASH, MEMOUT, WRONG) are outcomes,
   never exclusions.
4. **Instances:** none excluded after selection. Instances that no solver solves stay in the
   analysis; they contribute ties.

## 5. Statistical analysis

- **Per unit × instance set:** Wilcoxon signed-rank on *d_i* (zero differences: Pratt), *r* with
  95 % bootstrap CI (10 000 resamples of instances, stratified by family), and McNemar on solved
  status.
- **H1:** for each primary unit, Δ = *r*(seen) − *r*(unseen), with a 95 % stratified-bootstrap CI.
  One-sided test of Δ > 0 by bootstrap.
- **H2:** Wilcoxon signed-rank on log(T_A,pert / T_A,orig) − log(T_B,pert / T_B,orig), using CPU
  time capped at 5000 s, over instances solved in the original by at least one of the two.
- **Multiplicity:** Holm across the 5 primary units, separately for H1, H2a and H2b
  (α = 0.05, family-wise).
- **RQ5:** DerSimonian–Laird random-effects pooling of the per-unit Δ, reporting τ² and I².
- **Power:** with n_unseen = 240 and n_seen = 180, the minimum detectable paired standardised effect
  per set at α = 0.05 and power 0.8 is d ≈ 0.18 and d ≈ 0.21. Non-significant results are
  reported as inconclusive when the observed effect is below these thresholds.
- **Robustness checks (reported, not confirmatory):** PAR-10; ranks instead of PAR-2;
  excluding SAT or UNSAT instances; per-family estimates; leave-one-family-out stability of H1.

## 6. Reproducibility

Solver sources pinned by SHA-256 and built offline in a container whose image digest is
recorded; GBD snapshot hash; every seed above; harness code and all raw run records released
(Zenodo DOI) with the paper. All numbers in the paper are generated from the released records.
