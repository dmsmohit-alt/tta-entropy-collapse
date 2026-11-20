# When Does Entropy Minimisation at Test Time Collapse?

A dynamical analysis of test-time entropy minimisation (TENT) and a stable
marginal-entropy correction for continual shift.

**Status: Phases 0-3 done (scaffold, toy model, theory/stability, real-network baselines on CIFAR-10-C). Phase 4 (TENT + marginal entropy, RDumb-style reset, SGD arm): experiments run on the user's machine, analysis in progress, NOT yet approved.**
Every number below comes from a saved run (config + CSV in `results/`); anything else is `TBD`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Configs live in `configs/`. Override any value from the command line with
dotted keys, e.g. `sweep.n_delta=9`.

Phase 1 (toy model, CPU, ~6 min with the default config):

```bash
python scripts/run_theory.py --config configs/toy_basin.yaml     # prints the run dir
python scripts/make_plots.py --run-dir results/toy_basin/<hash>  # figures/ inside it
# quick smoke run: ... toy.n_per_class=50 sweep.n_delta=9 sweep.sep_grid=[2,3]
```

Phase 2 (stability / eta*, CPU, about 10 min; infomax basin check about 2 min):

```bash
python scripts/run_stability.py --config configs/toy_stability.yaml
python scripts/make_plots_stability.py --run-dir results/toy_stability/<hash>
python scripts/run_infomax_basin.py --config configs/toy_infomax_basin.yaml
# quick run: ... stability.sep_grid=[2.0,3.0] stability.lam_list=[0.0,1.0] population.n_xi=20000
```

## Results

### Phase 1: toy linear-softmax model (run `toy_basin/489889f9`, 3 data seeds)

Setup: K=5, d=10, 200 points/class, sigma=1, orthogonal class means of norm `sep`,
Bayes-optimal init plus an extra bias `delta` on class 0, full-batch GD on mean
entropy (eta=2, 2000 steps). Collapse := final H(p_bar) < 0.1 log K. Margin :=
sep^2/sigma^2 (logit gap of the Bayes classifier at a class mean).

Bias-only adaptation, collapse threshold delta* (mean +/- std over 3 seeds; std is
population std; threshold found by 13-point scan + 10 bisection steps):

| sep | margin | delta* | delta*/margin |
|---|---|---|---|
| 1 | 1 | collapses already at delta=0 (3/3 seeds) | n/a |
| 2 | 4 | collapses already at delta=0 (3/3 seeds) | n/a |
| 3 | 9 | 7.61 +/- 0.50 | 0.85 |
| 4 | 16 | 16.34 +/- 0.15 | 1.02 |
| 5 | 25 | 25.85 +/- 0.34 | 1.03 |

Figures: `results/toy_basin/489889f9/figures/` (`fig1_trajectories`, `fig2_basin_1d`,
`fig3_basin_2d`, `fig4_threshold_vs_margin`).

Status of each statement (see THEORY.md in Phase 2 for derivations):
- **[observed]** For sep >= 3 there is a single threshold along the delta grid
  (collapse indicator flips once; unchanged when steps x4 or eta/2 at sep=3).
- **[observed]** Collapsed runs end with H(p_bar) <= 0.008 and top-class fraction >= 0.999
  (max / min over all collapsed runs in the sweep). In collapsing runs the bias gap
  b_0 - mean(b_others) keeps growing (fig1): rich-get-richer.
- **[observed]** delta* is close to the margin for sep >= 4; at sep=3 it is lower
  (0.85). No formula for delta* has been derived yet.
- **[observed]** With W also adapted (sep=3), no seed collapsed until delta=10.8, the
  last grid point (all 3 collapsed there); at intermediate delta the final H(p_bar)
  is between 0 and log K (fig2). Whether those intermediate states are true fixed
  points was only spot-checked informally, not logged: **TBD**.
- **Hypothesis (untested):** for sep <= 2 the balanced point is itself unstable under
  entropy minimisation. Phase 2 (Jacobian) will test this.
- Caveats: only 3 seeds; non-collapsed runs often hit the step budget before
  grad-norm < 1e-7 (only 21% of non-collapsed runs flagged `converged`), though
  their H(p_bar) is flat; collapsed runs never meet the tolerance (gradient decays ~1/t).

### Phase 2: derivations and stability (see THEORY.md for every derivation and its status label)

Runs: `toy_stability/5dd255d1` (84 rows = 3 data seeds x 7 separations x 4 lambdas, N=1000, K=5),
`toy_infomax_basin/a28b7df4`. Bias-only adaptation, toy model only.

- **Derived and verified:** bias flow `b_dot = E[p(log p + H)] - lam E[p(log pbar - <log pbar>_p)]`;
  at the balanced point the Hessian is `alpha_J (I - 11^T/K)` with `alpha_J = alpha_L + lam K s^2`;
  `eta* = 2/alpha_J`. Analytic Hessian vs finite differences: max error 3.4e-11.
  PyTorch-autograd Hessian check: `tests/test_dynamics.py::test_hessian_matches_torch_autograd_if_available` is not skipped on the user's machine (88 passed, 0 skipped), i.e. it ran and passed; the error size was not logged: **TBD**.
- **eta\*, theory vs measurement:** `eta* = 2 / lambda_max(Hessian)` vs bisection on nonlinear GD:
  max relative error 0.19% over 57 stable rows (sample-level Hessians). The population `eta*` is higher than
  the N=1000 sample `eta*` (ratio 0.79 +/- 0.14). `eta*` values here (10 to 10^4) belong to this toy objective and are
  **not** a learning rate for real networks (the real-network curvature was measured in Phase 4, see below).
- **Curvature of plain entropy minimisation (K=5, population, Monte Carlo):** `alpha_L` is negative for sep <= 2
  (-0.099, -0.035, -0.0010 at sep 1, 1.5, 2) and positive from sep 2.5 (+0.0086); sign change near sep 2.05 [observed].
  This agrees with Phase 1, where sep 1 and 2 collapsed already at `delta = 0`.
- **Threshold:** the exact 1-D reduced population flow gives `g_c` = 8.18, 15.88, 25.01 at sep 3, 4, 5, versus
  Phase 1 simulated 7.61 +/- 0.50, 16.34 +/- 0.15, 25.85 +/- 0.34 (differences larger than one std: finite-sample, untested).
- **Toy check of the fix** (3 seeds, bias-only, eta 2, 3000 steps): for sep 3 and 4, `lam` in {0.5, 1, 2} gave 0/3 collapses
  at every initial bias up to 2 x margin, while `lam = 0` collapsed 3/3 from 1.2 x margin. [observed; toy only]
- **Caveats:** finite-sample noise at large separation (sep 5: population curvature +0.0002 but sample curvature negative);
  the 1-D reduction does not hold for 2 of 9 sample seeds; only K=5; no claim about deep networks yet.

### Phase 3: real-network baseline (Source / BN-adapt / TENT on CIFAR-10-C)

The stream runs were executed on the user's machine (RTX 2050); the author's sandbox has no GPU/torch/internet. The table below
is copied from the user's `aggregate_runs.py` output (`results/stream_tent/aggregate.csv`, which the author has not opened).

```bash
pip install -r requirements.txt                        # torch with CUDA already installed; adds torchvision
python scripts/download_cifar10c.py                    # ~2.9 GB, md5-checked, -> data_cache/CIFAR-10-C
python scripts/train_source.py --epochs 30             # standard ResNet-18 -> checkpoints/resnet18_cifar10_seed0.pt
python scripts/check_setup.py --config configs/stream_tent.yaml        # data layout + source-error sanity table
python scripts/run_stream.py --config configs/stream_smoke.yaml        # 20-step plumbing run (not a result)
python scripts/run_stream.py --config configs/stream_tent.yaml         # 3 methods x 3 seeds, 10 cycles
python scripts/aggregate_runs.py --exp-dir results/stream_tent         # mean +/- std (sample std, ddof=1)
pytest -q                                                              # includes tests/test_tta.py (needs torch)
```

Per run: `results/stream_tent/<method>/<confighash>_seed<N>/{config.yaml, metrics.csv, summary.json, run.log}`.
`metrics.csv` has one row per batch: `online_err, mean_H, H_bar_batch, H_bar_ema, top_frac, top_class, collapsed_flag`.
Collapse rule (assumption, in `src/metrics/collapse.py`): `top_frac > 0.9` for 5 consecutive batches; we report the
first batch of the first such streak (`time_to_collapse_start`) and the batch where it was detected.

| Experiment | Result |
|---|---|
| Setting for all rows | ResNet-18 (own checkpoint), CIFAR-10-C severity 5, 15 corruptions x 10 cycles, batch 200, 7500 steps, `nchw` layout, seeds 0,1,2, mean +/- sample std |
| Source overall error | 0.4473 +/- 0.0000 (collapsed 0/3) |
| BN-adapt overall error | 0.2256 +/- 0.0001 (collapsed 0/3, max top-class fraction 0.195) |
| TENT overall error | 0.6895 +/- 0.0724; last-cycle error 0.8592 +/- 0.0360 (accuracy 14.1%) |
| TENT collapse (top_frac > 0.9 for 5 batches) | 3/3 seeds; start step 4584 +/- 1455; final H(p_bar) EMA 0.21 +/- 0.21; 0 diverged |

**Finding to confirm (input memory layout).** `permute(0,3,1,2)` of an NHWC array gives a channels_last-strided tensor, which
makes the whole network run in NHWC. On the author's RTX 2050 this made batch-statistics BatchNorm (BNAdapt, TENT) several times
slower than contiguous NCHW input (see `results/bench_bn_layout/model.csv`). The default is now `compute.input_layout: nchw`
(also how the source model was trained). Layout changes floating-point rounding only, and:
- Source and BNAdapt: practically unchanged (a handful of flipped predictions out of 150,000 on one cycle).
- **TENT is layout-sensitive:** on one cycle (750 steps, one seed, identical batches) the two layouts gave visibly different error and
  different adapted BN parameters. Magnitude: see the user-reported numbers below (JSON in `results/check_layout_equivalence/`, not seen by the author). We do not know which layout is "more correct"; neither is ground truth.
  Consequence: single-run differences between TENT variants of a size comparable to this effect are not trustworthy; compare
  methods with >= 3 seeds, in ONE fixed layout, and report layout sensitivity as a limitation.
  Possible explanation, untested: entropy-minimisation dynamics amplify tiny perturbations (cf. the unstable balanced point in THEORY.md).
- User-reported layout results (console/CSV not seen by the author; TENT, seed 0, 750 steps, identical batches): as_is vs nchw max|dlogit| 27.7,
  11321/150000 prediction diffs, error 0.23693 vs 0.26089, max|dparam| 9.155e-2. **Control** (nchw vs nchw rerun): 0 prediction diffs, identical error
  (so run-to-run nondeterminism is not the cause; the layout rounding difference is). Per-segment error (as_is / nchw): segments 0-5 differ by <= 0.0014;
  seg 6 0.1768/0.1959; seg 9 0.2252/0.2706; seg 14 0.2911/0.3604. I.e. trajectories stay together for ~6 segments, then drift apart (one seed only).
- Layout study (TENT, 1 cycle, 3 seeds; user-reported): as_is 0.2579 +/- 0.0452 (848 s per run), nchw 0.2712 +/- 0.0312 (150 s per run); n_collapsed 0 in both.
  The difference of means (0.013) is well inside the seed std, so with n = 3 a layout effect is **not detectable** here; that is absence of evidence, not
  evidence of absence. Paired per seed (user-reported, nchw - as_is on identical batches): +0.0240, +0.0195, -0.0036; mean +0.0133, std of differences 0.0148, n = 3.
  Two of three seeds favour as_is; with n = 3 this is not distinguishable from zero.
- Does the collapse timing depend on layout? **Unknown.** TENT collapse start varies by +/- 1455 steps across seeds (nchw), so one as_is run cannot resolve a timing difference.

Notes: the stream order depends on the seed only, so all methods see identical batches. The source model is a single
trained checkpoint, so seed std reflects stream order, not source-training variance (a limitation).

Caveats for the Phase 3 numbers: collapse is defined by our own rule (assumption); one source checkpoint, so std is stream-order
only; TENT is layout-sensitive (above), and the layout study (3 seeds, 1 cycle) found no detectable layout effect at n = 3 (see above); the user reports `pytest -q`: 88 passed, 0 skipped (2026-11-18, latest report; the count grew 81 -> 83 -> 88 as tests were added; not seen by the author).

### Phase 4: TENT + marginal entropy, RDumb-style reset, SGD arm - code only, NO results yet

Methods (all via `tta.<name>` in the configs; `kind: tent` with options):
- `lam > 0`: loss = E[H(p)] - lam * H(p_bar) on the batch ("TENT + marginal entropy", SHOT-style information maximisation). `lam = 0` is bit-identical to Phase 3's TENT (test).
- `reset_every: T`: RDumb-style baseline = reset the BN parameters *and the Adam state* every T steps, before processing batch t (t % T == 0, t > 0).
  Verified from the RDumb paper page (arXiv 2306.05401): fixed period T = 1000 steps, resets to the pre-trained weights, ETA-style loss, SGD lr 2.5e-4, batch 64, ImageNet-scale;
  it gives **no CIFAR-10-C settings**, so ours is "RDumb-style", not a reproduction.
- SGD arm: plain gradient descent (momentum 0) with `eta = c * 2/lambda_max`, lambda_max measured by `scripts/measure_curvature.py`. The theory's eta* = 2/lambda is for GD, not Adam, so the Adam main comparison is only *qualitatively* linked to it.

Decisions (user-approved): lam = 1 and T = 1000 are a-priori primaries; sweeps lam in {0.1, 0.5, 1, 2} and T in {250, 500, 1000, 2000} are reported in full;
selection (if any) is made on a severity-3 stream with a different seed (100), with a pre-registered rule written in `configs/stream_select_sev3.yaml`; Adam lr 1e-3 for the main comparison.

```bash
pytest -q tests/test_tta.py tests/test_curvature.py                         # new torch tests (NOT run by the author)
python scripts/run_stream.py --config configs/stream_smoke4.yaml            # 20 steps; rdumb_T10 summary must show n_resets = 1
python scripts/run_stream.py --config configs/stream_phase4.yaml            # tent_im_lam1, rdumb_T1000 x 3 seeds (~2.2 h on RTX 2050)
python scripts/run_stream.py --config configs/stream_select_sev3.yaml       # selection stream (~3.3 h; subset allowed)
python scripts/measure_curvature.py --config configs/stream_select_sev3.yaml --lams 0 1 --n-segments 5
python scripts/run_stream.py --config configs/stream_sgd.yaml tta.tent_sgd.lr=<eta> tta.tent_im_sgd_lam1.lr=<eta>
python scripts/run_stream.py --config configs/stream_sweep_sev5.yaml        # full sweep (~6.6 h; use Colab or experiment.seeds=[0])
python scripts/aggregate_runs.py --exp-dir results/stream_tent
```

| Experiment (severity 5, 3 seeds unless noted) | Result |
|---|---|
| TENT + marginal entropy (lam = 1, a priori), overall error / collapse | 0.2632 (collapsed 0/3) [from the user's aggregate output; per-cycle values in the row below; std and `top_frac` columns not seen by the author: TBD] |
| RDumb-style (T = 1000, a priori), overall error / collapse | 0.2862 (collapsed 0/3) [same source, same TBDs] |
| Reference rows (Phase 3, same stream) | Source 0.4473, BN-adapt 0.2256, TENT 0.6895 (collapse 3/3) |
| Observation | Both collapse-preventing variants are *worse* than BN-adapt alone (0.2256) in overall error; whether this is a slow per-cycle drift, a cost of the entropy objective itself, or lam/T not tuned is **unknown** (needs err_by_cycle and the sweeps) |
| Severity-3 selection stream (seed 100, 10 cycles, **one seed**, overall error; user-reported) | Source 0.2627, BN-adapt 0.1575, TENT 0.4615; lam 0.1 / 0.5 / 1 / 2: 0.1820 / 0.1887 / 0.1878 / 0.1947; RDumb-style T = 250 / 500 / 1000 / 2000: 0.1417 / 0.1479 / 0.1587 / 0.1883; n_collapsed 0 for all |
| Selection rule applied (rule in `configs/stream_select_sev3.yaml`, written before the runs) | **lam**: best alternative lam = 0.1 is better than lam = 1 by 0.0058 < 0.01, so lam = 1 is kept. **T**: T = 500 (better than T = 1000 by 0.0108) and T = 250 (by 0.0170) both exceed 0.01. The rule did **not** say which one to take when several qualify (a gap in the rule, found only now). Reading used: the qualifying candidate with the lowest overall error, **T = 250**. This tie-break was fixed after seeing the selection results; to limit the damage T = 500 is reported with the same status (see the table below). |
| Severity-5 sweep, **seed 0 only** (overall error; user-reported) | RDumb-style T = 250 / 500 / 1000 / 2000: 0.2009 / 0.2235 / 0.2832 / 0.4088 (T = 1000 3-seed mean: 0.2862). lam = 0.1 / 0.5 / 1 / 2: 0.2753 / 0.2447 / 0.2727 / 0.2928. BN-adapt 0.2256. Single-seed differences below ~0.02 are not reliable (lam = 1 is 0.2727 at seed 0 vs 0.2632 over 3 seeds). |
| Observations (hypothesis level) | Error rises monotonically with T on both streams (sev-3 and sev-5, seed 0), and the smallest T tried (250) is the best: the grid has not found the minimum. On sev-3 every lam arm is worse than BN-adapt, and only T <= 500 beats it. On sev-5 seed 0 the best lam is 0.5, on sev-3 it is 0.1: the lam ranking is not stable across streams, which is why lam was not tuned. Resets in all four T are aligned with corruption boundaries (see the control below). |
| RDumb-style, sev-5, 3 seeds each (user-reported) | **T = 250 (selected)**: overall 0.2033 +/- 0.0023, last-cycle 0.2073, collapse 0/3, degradation onset vs BN-adapt: none in all 3 seeds; paired difference to BN-adapt (overall): -0.0246, -0.0223, -0.0202 (better in 3 of 3; std of differences 0.0022). **T = 500**: 0.2300 +/- 0.0057; paired -0.0020, +0.0071, +0.0080 (better in 1 of 3). **T = 1000** (a priori): 0.2862. T = 2000: 0.4088 (seed 0 only). The selection on sev-3 (T = 250) and the tie-break hedge (T = 500) therefore end in the same ranking on sev-5: T = 250 < 500 < 1000 < 2000. |
| Reset-phase control `rdumb_T250_off25` (resets in the middle of a segment; seed 0 only; predictions P1/P2 written in `configs/stream_reset_control.yaml` before the run) | overall 0.2041, last-cycle 0.2014, no degradation onset, versus 0.2009 for aligned T = 250 at seed 0 (difference 0.003, about one seed std of T = 250). Looks like P2 (short horizon, not boundary alignment), single seed, so only a statement about effects of ~0.02 or larger. Seeds 1-2 not run (the pre-registered trigger was a difference above ~0.01). |
| Wrapper sanity check (user-reported) | cycle-0 error of `rdumb_T1000` equals `tent` (0.271, sev-5 3-seed mean); on sev-3 tent = T1000 = T2000 = 0.151, as required (no reset within the first 750 steps) |
| Error vs steps since reset (`scripts/error_vs_steps_since_reset.py`) | TBD |
| Measured lambda_max / eta*_GD of the BN-parameter Hessian (source weights) | Power-iteration numbers (user's run, 5 segments, severity 3) are **superseded**: that estimator is unreliable when |lambda_min| >> lambda_max. Lanczos run (user-reported, 15 segments, severity 3, batch 200, source weights; `results/curvature/.../curvature.csv` not seen by the author): lambda_max 2.5 to 7.0, lambda_min -6.4 to -17.2, nearly identical for lam=0 and lam=1; min eta*_GD = 0.2873 (lam=0), 0.2857 (lam=1), both from the contrast segment; impulse noise lambda_max 3.80 / 4.01 (so the earlier 0.027 -> 3.88 jump was an estimator artifact). Contrast cross-check (ncv 120, tol 1e-5, other start vector; user-reported): lambda_max 6.960 / 6.999, lambda_min -7.78 / -7.70, same as before |
| SGD arm, eta = 0.5 x 0.2857 = 0.1429 (plain SGD, momentum 0), `tent_sgd` (lam=0) | overall error 0.7572, collapsed 3/3, start 4617 +/- 1645 [user's aggregate output, not seen by the author] |
| SGD arm, `tent_im_sgd_lam1` | overall 0.3652; last-cycle 0.5654 +/- 0.0677; collapsed 0/3; top-class fraction 0.13; final H(p_bar) EMA 2.30 (= log 10: marginal essentially uniform). **Last-cycle error is above Source (0.447) although the marginal is balanced**: a balanced marginal does not imply correct predictions. Cause unknown; per-cycle error (`aggregate_per_cycle.csv`): TBD |
| Dose-response, SGD + lam=1 (user-reported; c = eta / 0.2857; c = 0.5 is the 3-seed mean, c = 0.9 / 1.1 / 2.0 are **seed 0 only**) | per-cycle error, c=0.5 (eta 0.1429): 0.196 0.224 0.248 0.285 0.319 0.378 0.426 0.483 0.527 0.565 (overall 0.3652). c=0.9 (0.2571): 0.211 0.285 0.330 0.394 0.464 0.508 0.552 0.607 0.704 0.783 (overall 0.4838). c=1.1 (0.3143): 0.228 0.372 0.448 0.529 0.681 0.705 0.738 0.781 0.846 0.916 (overall 0.6244). c=2.0 (0.5714): 0.401 0.639 0.718 0.740 0.742 0.742 0.755 0.778 0.796 0.797 (overall 0.7108; diverged_step null). Marginal balanced in all: max top_frac 0.185-0.19, final H(p_bar) EMA 2.296-2.2985, 0 collapses. |
| Reading of the dose-response (hypothesis level) | Cycle-0 error rises smoothly and convexly with c (slopes per unit c: 0.037, 0.085, 0.192; cycle-0 seed noise ~0.003): **no kink at c = 1 there**. Against t = eta x steps, c=0.9 stays within 0.03 of c=0.5 up to t ~ 500 and is lower later (up to 0.05, inside c=0.5's seed std 0.068); c=1.1 sits 0.03-0.09 above c=0.5 and ~0.08-0.11 above c=0.9 from t ~ 375. Compatible with an eta*-linked onset between c = 0.9 and 1.1, but not demonstrated: 4 points, one seed, a saturating metric (c=2.0 plateaus at 0.74-0.80), and the theory's eta* is per-segment (0.29 to 0.8), i.e. it predicts a smeared, not sharp, onset. Per-segment test (`scripts/segment_vs_curvature.py`): TBD. c = 1.1 last-cycle error 0.916 is above chance (0.90): predictions anti-correlated with labels; confusion matrices (new `confusion_by_cycle.npy`): TBD |
| Degradation onset (delta = 0.01), vs BN-adapt / vs Source (user-reported) | tent (Adam) 0 / 2; tent_im_lam1 (Adam) 3 (per seed 2,4,3) / never (final excess vs Source -0.12); tent_im_sgd_lam1 c=0.5: 2 (2,2,2) / 7 (9,7,7); c=0.9: 1 / 4; c=1.1: 1 / 3; c=2.0: 0 / 1; tent_sgd: 0 / 1; rdumb_T1000: 0 / never (final excess vs BN-adapt +0.028) |
| Paired cycle-0 difference, lam=1 arm minus BN-adapt (user-reported, 3 seeds) | tent_im_lam1: -0.0306, -0.0257, -0.0286 (mean -0.0283, std 0.0025). tent_im_sgd_lam1: -0.0324, -0.0286, -0.0272 (mean -0.0294, std 0.0027). Better in 3 of 3 seeds |
| Per-cycle error, seed means (user-reported; only the values the user listed, full table in `aggregate_per_cycle.csv`: TBD) | BN-adapt flat 0.226; TENT (Adam) 0.271 -> 0.859; TENT+lam1 (Adam) 0.198, 0.214, 0.231, ..., 0.327; TENT+lam1 (SGD, lr 0.1429) 0.196, 0.224, 0.248, ..., 0.565; RDumb T=1000 saw-tooth, 0.236 to 0.361; `tent_sgd` 0.316 -> 0.870 |

**Two different failure modes (terminology used from here on).**
1. *Marginal collapse*: one class dominates (`top_frac > 0.9` for 5 batches; `src/metrics/collapse.py`). This is the failure the toy theory describes.
2. *Degradation*: error rises above a reference while the marginal can stay balanced. Defined by `scripts/degradation_metrics.py`: first cycle where the
   seed-mean error exceeds the BN-adapt error (primary) or Source error (secondary) by more than delta = 0.01 (absolute); also `sustained`, `n_degraded`,
   per-seed onsets paired with the same seed's reference. delta = 0.01 is far above BN-adapt's seed noise (~1e-4) and below all effects seen so far; it was
   **fixed after seeing the Phase 4d per-cycle means, so it is not blind for those runs**, only for later ones (c = 0.9 / 1.1 / 0.25, selection, sweeps).
   Sensitivity delta in {0, 0.02, 0.05} is in `degradation.csv`.

Observations so far (seed means, 3 seeds unless noted; interpretation is hypothesis, not a result):
- The marginal term (lam = 1) prevents marginal collapse in every run reported (0/3 for Adam and SGD; also at eta = 2 x eta*_min, seed 0).
- It does not prevent degradation: every lam = 1 arm drifts upward over cycles, and the SGD arm ends above Source (0.565 vs 0.447).
- In cycle 0 the lam = 1 arms (0.198 Adam, 0.196 SGD) are below BN-adapt (0.226). Paired per-seed differences are in the table above (3 of 3 seeds, mean about -0.03).
- Why a balanced marginal can be wrong (hypothesis): the objective E[H(p)] - lam H(p_bar) never sees labels; any confident AND balanced labelling is a minimiser, correct or not.

```bash
python scripts/degradation_metrics.py --exp-dir results/stream_tent          # -> degradation.csv
python scripts/compare_runs_paired.py --exp-dir results/stream_tent --method-a bn_adapt --method-b tent_im_lam1 --cycle 0
```

Limitations of the curvature estimate: one batch per segment, at the source weights only, power iteration (see `--n-iter`, `converged_*` columns); the Hessian along the adapted trajectory may differ.
The batch-marginal term assumes roughly class-balanced batches; on a class-imbalanced stream it may push the wrong way (ablation planned in Phase 5).

## Limitations (to be expanded in Phase 7)

- A linear-softmax result may not fully transfer to deep networks.
- Collapse has been reported empirically by prior work; the contribution here
  must be the sharper dynamical explanation plus the stability condition.
- Novelty versus the last 12 months of arXiv / Google Scholar is **not yet checked**.
