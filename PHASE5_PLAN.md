# Phase 5 plan (analysis and ablations)

Status labels: **pre-registered** = predictions are written in the config before the run; **no GPU** = CPU analysis of runs that already exist.
Costs assume ~22 min per 7500-step Adam/SGD run on the RTX 2050 (BN-adapt ~10 min); Colab T4 timings differ.

## Status 2026-11-25
Done (user-reported, see README `Phase 5`): factorial (3 seeds, P1-P4 pass, my effect(250) note wrong), T = 350, T = 100 / 150 (all pre-registered checks pass except the T = 350 peak location), Phase 1 W+b spot-check (intermediate states are not exact fixed points).
Next (GPU, laptop): confusion reruns (~45 min), lambda threshold (~45 min), severity contrast (~2.3 h), T extension on severity 3 (~66 min). Seeds 3-5 replication only if needed (new stream order only). Candidate cells that need predictions written first: (lam 1, T 150), lam = 1 T-profile at T = 550.

## 0. Housekeeping (no GPU)
1. Commit the configs with predictions BEFORE running (`stream_factorial.yaml`, `stream_t350.yaml`, `stream_t_extension.yaml`, `stream_severity_contrast.yaml`, `stream_reset_control.yaml`); push the result CSVs.
2. Per-segment eta* test (no GPU): `scripts/segment_vs_curvature.py` on c = 2.0 vs c = 0.5 (seed 0) with the Lanczos curvature CSV. Tells whether the extra degradation at large step sits in the segments whose local eta* is below the step.
3. Finish the novelty check (REFERENCES.md): Google Scholar and arXiv listing over the last 12 months; read in full Lim et al. (ICLR 2026), Niu et al. (SAR / SAR2), RDumb; open SHOT before citing it.

## 1. Tier 1: needed for the central claims
| # | Experiment | Cost | Status | Question |
|---|---|---|---|---|
| 1 | lam x T factorial: add (lam 1, T 1000) and (lam 1, T 250), 3 seeds (`stream_factorial.yaml`) | 6 runs, 2.2 h | pre-registered | does the marginal term still matter when resets exist, and does it extend the usable window? |
| 2 | Confusion matrices: re-run `tent_im_sgd_lam1_c1p1` and `tent_im_lam1` seed 0 under `experiment.name=confusion_rerun` (new folder; also a determinism check: errors must reproduce) | 2 runs, 45 min | prediction to write | is 'degradation with a balanced marginal' a systematic wrong mapping (error 0.916 above chance)? |

## 2. Tier 2: sharpen the mechanism
| # | Experiment | Cost | Status |
|---|---|---|---|
| 3 | lam threshold: lam in {0.01, 0.03} at severity 5, seed 0 (toy: collapse returns below lam_c; does it in the real network?) | 2 runs, 45 min | config + prediction to write |
| 4 | T = 350 identifiable design (`stream_t350.yaml`), 3 seeds | 66 min | pre-registered |
| 5 | T grid below 250: T = 50, 100, 150 on the selection stream (`stream_t_extension.yaml`) | 66 min | pre-registered |
| 6 | Severity contrast: sev-3 seeds 0-2 for bn_adapt and T2000 (+ sev-5 T2000 seeds 1-2) (`stream_severity_contrast.yaml`) | ~2.3 h | pre-registered |

## 3. Tier 3: ablations from the project spec (need design decisions first)
- Adam lr {3e-4, 3e-3} for TENT and TENT+lam1; batch size {100, 400}: one factor at a time, 3 seeds, severity 5.
- Class-imbalanced stream (the known weak spot of a batch-marginal term): needs a new stream option (decide the imbalance protocol first).
- Extra source checkpoints (2 more seeds of `train_source.py`): the current seed std is stream order only. Needs the training time on your GPU.
- Dose-response completions: c = 0.25 for 20 cycles (+ 20-cycle references, ~1.3 h); seeds 1-2 for c = 0.9 and 1.1 (~1.5 h). Only if the eta* statement stays in the paper.

## 4. Toy versus real (no GPU, `scripts/compare_toy_real.py` to be written)
(a) collapse trajectory: H(p_bar) / top_frac of the TENT runs against the toy's rich-get-richer gap flow (shape and time scale, qualitative); (b) lam threshold from item 3 against the toy's lam_c;
(c) toy eta* scale against the measured BN-parameter curvature (the two live in different parameterisations: only qualitative).

## 5. Figures to produce from saved runs
F1 error per cycle, all methods; F2 failure modes (collapse vs degradation); F3 dose-response; F4 reset profile and T; F5 factorial; F6 toy vs real.

## Decisions needed from you
1. GPU budget (laptop only, or Colab T4 for the long items). 2. Imbalance protocol and whether to train extra checkpoints. 3. Title / claim wording with your advisor (the data do not support "stable updates via eta < eta*").
