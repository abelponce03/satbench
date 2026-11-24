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

## 6. Amendment 1 (2026-11-24, before main data collection)

Made after the tooling pilot (10 jobs; excluded from every analysis) and before any
main-study block ran. It is purely additive: every block of the original plan is kept unchanged
(`study/blocks.csv` contains all 4260 original block ids), and §1–§5 still define the
confirmatory analyses. Added: 361 blocks (4621 in total), new per-run measurements, and the
pre-specified analyses below.

### 6.1 New measurements (recorded for every run)

- **Retired user-space instructions and cycles** of the whole process tree, via
  `perf_event_open` attached before the solver starts (start gate). In tests on the same binary
  and input, instruction counts varied by 0.00017 % across repetitions (CPU time: 0.23 %).
  Instruction counts depend on the binary and the input, not on the host's clock or
  microarchitecture.
- **CPU time per executable**, sampled. It separates preprocessing (e.g. Satsuma) from search, and
  shows which LymphoSAT sub-solver handled each instance.
- Excerpts of solver comment lines, the end of stderr, and GBD base features of every file run.
- Per job: a **concurrent calibration** (both cores at once) next to the solo calibrations, which
  measures interference between the two members of a block.
- **Replicates:** 2 % of blocks run twice on different hosts (test–retest).
- **Encoding seeds:** `s2-rcl` and `s3-rcl` on a fixed quarter of the perturbed instances, so each
  of those instances has three order-perturbed encodings.

### 6.2 Additional pre-specified analyses (secondary unless stated)

1. **Host-invariant outcome (co-primary for H1).**
   - Every host's speed is estimated from its calibration runs as instructions per CPU-second (IPS).
   - A common instruction budget is set: B = 0.95 × 5000 s × min(IPS) over retained hosts.
   - A run is *solved within B* if it is SAT/UNSAT and used at most B instructions. Its
     instruction-PAR-2 is its instruction count if solved, 2B otherwise.
   - Censoring is then identical on every host by construction. H1 is tested with this outcome
     exactly as in §5, and both versions are reported.
2. **Encoding-lottery variance.**
   - Model: log CPU time = instance effect + encoding effect, fitted on the instances with
     original + s1/s2/s3-rcl.
   - Reported: the intraclass correlation and the within-instance (encoding) standard deviation.
   - Also reported: the share of instance-level win/loss outcomes of each unit that flip between
     encodings, which is the "encoding noise floor" of a paired solver comparison.
3. **Test–retest reliability:** ICC and Bland–Altman limits for log CPU time and for log
   instructions on replicate pairs.
4. **Interference:** concurrent-to-solo calibration time ratio per job. Sensitivity of H1 to
   excluding jobs whose ratio exceeds 1.15.
5. **Time-horizon curves:** r(T′) for T′ on a log grid from 10 s to 5000 s (PAR-2 recomputed at
   every horizon from the same runs). Does the AI advantage depend on the time limit the solvers
   were tuned for?
6. **Distribution-shift gradient (directional hypothesis H6).** The AI variant's advantage
   decreases with an instance's distance from its development data.
   - Distance: Euclidean distance, in standardised log-transformed GBD base features, to the
     nearest instance of that variant's development set (docs/exposure.md).
   - Test: Spearman correlation between the per-instance paired advantage and the distance,
     one-sided, Holm across the primary units.
7. **Mechanism.** CPU share of preprocessing vs search per unit, and LymphoSAT's dispatch rate to
   targeted sub-solvers on seen vs unseen families and under perturbation. Does dispatch explain
   its advantage, and does reordering defeat its matchers?
8. **Correctness audit.** Any UNSAT claim on an instance for which some run produced a verified
   model is counted as wrong, together with the WRONG answers of §1 RQ4.
9. **External validity.** On instances and solvers shared with the SAT Competition 2026 results,
   Spearman correlation of log CPU times and agreement on solved status.
10. **Censored mixed model (robustness).** A Bayesian accelerated-failure-time model for log CPU
    time, with right-censoring at 5000 s:
    - Fixed effects: solver × exposure set.
    - Random effects: instance and encoding.
    - Host covariate: speed (IPS).
    - Reported as speed-up factors with 95 % credible intervals.
11. **Size of the AI modification.** Lines changed between each AI variant and its base, from the
    archives, against the unit's generalisation gap (descriptive; n = 5).

## 7. Reproducibility

Solver sources pinned by SHA-256 and built offline in a container whose image digest is
recorded; GBD snapshot hash; every seed above; harness code and all raw run records released
(Zenodo DOI) with the paper. All numbers in the paper are generated from the released records.
