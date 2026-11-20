# Learning Notes

Running notes on *why* each design decision was made. Updated every phase.

## Phase 0

**Why seeds, saved configs, and CSV logs?**
A result is only science if it can be reproduced. Every run saves
`config.yaml`, `metrics.csv`, `summary.json` in its own folder, so any number in
the README can be traced to a file.

**Why compute entropy from `log_softmax` instead of `log(softmax(x))`?**
If one logit is far larger than the others, softmax gives exactly `0.0` for the
rest and `log(0) = -inf`, which then gives `0 * -inf = nan`. `log_softmax`
computes log-probabilities directly, so they stay finite.

**Why `logsumexp` for the marginal?**
`p_bar = mean_i p_i`, so `log p_bar = logsumexp_i(log p_i) - log N`. Same
stability reason: we never leave log-space until the end.

**The identity `E[H(p)] = H(p_bar) - I(X;Y)`.**
Here `I` is *defined* as `H(p_bar) - E[H(p)]`; the real content is that it is
`>= 0` (Jensen: entropy is concave, so the entropy of the average is at least
the average of the entropies). Consequence: lowering `E[H(p)]` can be done two
ways: raise `I` (predictions become confident *and* diverse: good) or lower
`H(p_bar)` (everything goes to one class: collapse). Plain entropy
minimisation does not distinguish them. The `-lambda * H(p_bar)` term does.

**Sanity picture (K classes):**
- uniform predictions: `H(p_bar) = log K`, `E[H(p)] = log K`, `I = 0`
- confident and diverse: `H(p_bar) = log K`, `E[H(p)] ~ 0`, `I = log K` (ideal)
- confident and collapsed: `H(p_bar) ~ 0`, `E[H(p)] ~ 0`, `I ~ 0` (the failure)

**Why `from __future__ import annotations` and type hints?**
Readable signatures, and tools can catch mistakes early.

## Phase 1

**Entropy gradient.** For one sample with p = softmax(z): dH/dz_k = -p_k (log p_k + H).
So the bias update under GD is b_dot_k = mean_i p_ik (log p_ik + H_i). This is
checked against finite differences in `tests/test_toy_model.py` (and against torch
autograd if torch is installed). Phase 2 turns this into the rich-get-richer argument.

**Why start from the Bayes classifier?** It is accurate and has a balanced marginal,
so any collapse is caused by the injected bias `delta`, not by a bad classifier.

**Why numpy and not torch here?** The model is tiny; analytic gradients are faster
on a CPU, and give an independent check for the later autograd computation.

**Why report "converged" separately from "collapsed"?** Near collapse the gradient
decays slowly (~1/t), so a gradient-norm stopping rule would never fire. We classify
by final H(p_bar) and report the flag honestly instead.

**A plotting bug worth remembering.** Floats that "should" be equal (delta/margin on
different grids) differ in the last bits; pivoting on them produced fake blank cells.
Round before grouping. Always look at the figure, not only the numbers.

**What this phase does NOT show.** It shows *numerically* where the boundary is for
one family of toy problems. It does not derive the threshold; that is Phase 2.

## Phase 2

**Gauge directions.** Adding the same constant to every bias changes nothing (softmax is shift invariant),
so the Hessian always has a zero eigenvalue along the all-ones vector. That is not an instability. We analyse
stability on the subspace orthogonal to it.

**Symmetry does the work.** At the balanced point all classes are interchangeable, so the Hessian must look like
`alpha * (I - 11^T/K)`: one number controls everything. That is why `eta* = 2/alpha` and why a 1-D flow for the
gap `b_0 - b_others` is *exact* for the population.

**Why `eta < 2/alpha`?** Plain GD on a quadratic with curvature `alpha` multiplies the error by `1 - eta*alpha`
each step; its magnitude is below 1 only if `0 < eta*alpha < 2`.

**Why check against the *nonlinear* GD?** Computing `eta*` from the Hessian and then "checking" it with the same
Hessian proves nothing. We perturb the fixed point slightly and run real GD; that uses only the gradient.

**Two bugs this phase, both worth remembering.**
1. *Noise-floor measurement.* The measured growth factor was about 1.000000 for steps that should decay. The perturbation had
   shrunk to the solver's noise level (1e-12) and stopped moving. Fix: stop when it falls below a floor far above
   the noise, and compare to `1 - 1e-6`, never `1.0`. Before the fix three cases showed errors of 17-63%; after, 0.19% max.
2. *Newton escaping.* When the balanced point is unstable, Newton from `b=0` jumped to a far-away flat (collapsed) stationary
   point, where the Hessian is about 0 and looks "marginally stable". We now cap the step, require a small gradient and
   `|gap| < 2`, and also log the curvature at exactly `b=0`.
Also: a hand-checked table caught three wrong numbers in a draft of THEORY.md. Always recompute quoted numbers from the CSVs.

**What this phase does not prove.** Bias-only, K=5, orthogonal means, Monte-Carlo evaluated constants. The threshold
`g_c` and the sign change of `alpha_L` near margin 4.2 have no closed form here.

## Phase 3

**What TENT actually does.** (1) BatchNorm normalises with the *current test batch's* mean/variance instead of the stored
training statistics (we set `running_mean = None`); (2) only the BN scale/shift (gamma, beta) get gradients; everything else is
frozen; (3) one Adam step per batch on the mean prediction entropy. ~10k parameters out of 11M for ResNet-18.

**Why return the pre-update logits?** Prediction for batch t is made before the step on batch t. Otherwise the model
would "see" its own adaptation on exactly the images it is scored on.

**Why BNAdapt (extra baseline)?** Using batch statistics alone is already an adaptation. Comparing Source, BNAdapt and TENT
separates "free" gains from the gains (and the collapse) caused by entropy gradient steps.

**Why seeds shuffle the stream.** Source and TENT are deterministic given the data, so without shuffling, 3 seeds would give 3 identical
numbers. Seeds change the order of images inside each corruption segment (and the same seed gives every method the same batches).

**Why drop_last?** BN statistics from a tiny final batch are noisy. With batch 200 and 10,000 images nothing is dropped.

**Why train our own ResNet-18?** RobustBench's CIFAR-10 ResNet-18 uses PRIME augmentation (already robust to corruptions), so it is not
the usual "standard model under shift" setting. WRN-28-10 `Standard` is the benchmark model but heavy for 4 GB VRAM.

**Collapse definition.** `top_frac > 0.9` for 5 consecutive batches. `top_frac` is the share of the batch predicted as its most common class.
It is an assumption; the log keeps `top_frac` for every batch, so other thresholds can be applied afterwards without re-running.

**What was verified how.** Numpy parts (metrics, collapse detector, loader, stream) have tests that ran. Torch parts (ResNet,
BN selection, Source/BNAdapt/TENT) have tests that were written but not executed by the author; run `pytest tests/test_tta.py`.

## Phase 3 follow-up: memory layout (channels_last) and TENT sensitivity

**Provisional numbers pasted from the author's console on 2026-11-16 (the script did not save them then; NOT yet README-grade,
to be regenerated by the updated script which writes a JSON):** RTX 2050, 750 steps, one seed, identical batches, as_is vs nchw.
Source 12/150000 prediction diffs, err 0.44737 vs 0.44733. BNAdapt 11/150000, err 0.22617 vs 0.22615. TENT: max|dlogit| 27.7,
11321/150000 diffs (7.5%), err 0.23693 vs 0.26089, BN param max|d| 9.2e-2. Speed with nchw input: BNAdapt 77 ms vs 409 ms, TENT step 175 ms vs 1112 ms.

**What happened.** `x.permute(0,3,1,2)` is a view with channels_last strides; `.float()` preserves strides; convolutions then run
in channels_last, and in this setup batch-statistics BN was several times slower there. `.contiguous()` fixed the speed.
Lesson: look at `tensor.stride()` / `is_contiguous(memory_format=...)` whenever a GPU op is mysteriously slow.

**Layout is not a property of the trained model.** The weights compute the same function in either layout; only rounding differs.
`train_source.py` trained in NCHW (data `.permute(...).contiguous()`, no channels_last anywhere), and `check_setup`/`profile_step`
used the channels_last path until they were switched to the shared `to_tensor(..., layout)`.

**Why TENT moved but Source did not.** Source has no feedback: rounding differences stay at 1e-5 scale. TENT feeds every batch's
logits back into the parameters that produce the next batch's logits, 750 times, so small differences can grow. This is a hypothesis
for the size of the effect, not something we measured; the control run (`--control`: nchw vs nchw rerun) and the 3-seed study are
what tell us whether layout, general nondeterminism, or stream order dominates.

**What 'errs equal' meant.** In the first version of the script the comparison was exact (`np.array_equal` on per-batch errors,
zero tolerance), so it could print False even when mean errors agree to 4 decimals. The new version reports mean error difference,
number of batches whose error differs, and the largest per-batch difference instead of one boolean.

**Layout study commands (TENT, one cycle, 3 seeds, both layouts; ~7 min for nchw, ~40 min for as_is):**
```
python scripts/run_stream.py --config configs/stream_tent.yaml methods=[tent] stream.num_cycles=1 experiment.name=layout_study compute.input_layout=nchw
python scripts/run_stream.py --config configs/stream_tent.yaml methods=[tent] stream.num_cycles=1 experiment.name=layout_study compute.input_layout=as_is
python scripts/aggregate_runs.py --exp-dir results/layout_study        # one row per layout (column 'layout')
```
Read it as: per-seed differences between layouts vs. the seed-to-seed std within a layout.

## Phase 4

**One class, three behaviours.** TENT+marginal, RDumb-style and the SGD arm are all `Tent` with different options (`lam`, `reset_every`,
`optimizer`), so a difference between them is caused by that option and nothing else. `lam = 0` is a test-checked identity with Phase 3.

**Why reset the Adam state too.** If only the weights are reset, stale Adam moments from the collapsing trajectory kick the first steps
after the reset. Our choice (user-approved), tested in `test_periodic_reset_also_resets_optimizer_state`.

**Why reset *before* the batch.** Batch t (t % T == 0) is then predicted by source weights, so no prediction is made by a model that
has had T updates in the wrong direction after the boundary.

**Why a severity-3 selection stream and a pre-registered rule.** Choosing lam or T on the stream we report would give optimistic numbers.
A different severity and seed keeps the report stream untouched; writing the rule before running prevents picking the rule after seeing results.

**Measuring curvature.** Hessian-vector products by double backward (`autograd.grad` of the gradient), power iteration for the largest-magnitude
eigenvalue, then power iteration on H - s*I for the other extreme. The algorithm was checked against `eigvalsh` in a numpy replica; the torch
version has tests that have not yet been run. eta* = 2/lambda_max is for plain GD only.

**A bug in my own estimator (found from a user's odd-looking result).** Shifted power iteration returned lam_max ~ 0.02 in most segments and a
jump 0.027 -> 3.88 on impulse noise. A numpy replica with a spectrum like the measured one (lambda_min = -17, bulk ~ 0) shows the cause: after the
shift the bulk sits at +17 and the true top eigenvalue is a ~0.02% bump, the iteration stops after 2 steps on the bulk, reports ~0 (true 3.88 -> reported 0.003)
and still says converged. Lesson: 'converged' only means two successive iterates agreed, not that the right eigenvalue was found; test an estimator on a
synthetic spectrum that looks like the real one. Fix: Lanczos (`src/utils/lanczos.py`, tested on synthetic spectra, no torch needed).

**What the numbers do NOT say.** Phase 3 shows TENT collapse in 3/3 seeds in our setup (nchw, our collapse rule). It does not yet show *why*;
the toy-vs-real comparison is Phase 5.

**Collapse is not the only way to fail (Phase 4d).** With lam = 1 the marginal stays at log K (top_frac ~ 0.13) yet error climbs cycle after cycle, and at
eta = 2 x eta*_min it reaches 0.71 overall. `E[H] - lam H(p_bar)` is label-free; confident balanced but wrong labellings are as good as right ones. So we track
two failure modes with two metrics (top-class collapse, error-based degradation). Writing the degradation rule down: a rule fixed AFTER seeing numbers is
honest only if you say so (we did, in README and the script docstring) and is only blind for later runs.

**Reading eta* in the real network.** The theory's eta* is about the local stability of a stationary point under plain GD. The c = 2.0 run is confounded: it
has 4x larger steps everywhere, and only the sharpest segments (lam_max up to ~7, local eta* ~ 0.29) are above their own eta*. Tests that separate the
explanations: (a) a knee between c = 0.9 and c = 1.1 would point to eta*, a smooth rise would point to a generic step-size effect; (b) per segment, degradation
at c = 2.0 vs c = 0.5 against that segment's lam_max (curvature measured on severity 3, stream on severity 5: a caveat).

**Gradient-flow time.** For small SGD steps the trajectory depends on eta x steps; halving eta needs twice the steps to reach the same point. Hence c = 0.25
needs 20 cycles to cover what c = 0.5 does in 10. Deviations from that collapse tell drift from minibatch noise (noise per unit time scales with eta).

**What the dose-response can and cannot say about eta*.** The measured local eta* spans 0.29 (contrast, lam_max ~ 7) to ~0.8 (lam_max ~ 2.5), so
'eta > eta*' switches on segment by segment between c = 1 and c ~ 2.8: the theory predicts a smeared onset, not a sharp threshold. A smooth global
curve therefore does not refute it, and a bump near c = 1 would not prove it. The discriminating test is per segment (`segment_vs_curvature.py`), with
its own confounds (order of segments, severity-3 curvature vs severity-5 stream). Cycle 0 is the cleanest data (seed noise ~0.003): there the excess
error grows smoothly with c (slopes 0.037 -> 0.085 -> 0.192 per unit c, ratio ~2.2 each).

**Worse than chance.** Last-cycle error 0.916 at c = 1.1 is above the 0.90 of random guessing: the predictions are anti-correlated with the labels
(a systematic wrong mapping), not noise. A label-free objective can reach that. Confusion matrices per cycle are now saved (`confusion_by_cycle.npy`,
[cycle, true, predicted]) for NEW runs to look at it directly.

**A cheap sanity check worth running.** rdumb_T1000 has no reset before step 1000, so its cycle-0 error must equal tent's cycle-0 error seed by seed
(same method, same batches, deterministic nchw). If it does not, the wrapper has a bug.

**A gap in my own pre-registered rule (Phase 4 selection).** The rule said "keep lam / T unless another value is better by more than 0.01". Two T values
qualified (500 and 250) and the rule did not say which. I took the lowest-error qualifying candidate (T = 250) AFTER seeing the numbers and said so in the
README; both candidates are reported with the same status. Lesson: write the rule as a function that returns exactly one answer for every possible outcome
(here: "among qualifying candidates take the lowest error"), and test it on made-up numbers before running.

**The best value was at the edge of the grid.** T = 250 is the smallest T tried and the error rose monotonically with T on both streams, so the true optimum may be
smaller. A grid whose best point is at its edge has not located the optimum; do not describe 250 as "optimal". (T = 1 would be BN-adapt exactly, because the prediction is
made before the update; so the curve over T must come back to the BN-adapt level at small T.)

**Resets at known boundaries.** All T in the grid are multiples of 50 (batches per corruption), so every reset lands right before a new corruption. Resetting at a
domain change is the easiest case. `reset_offset` moves the reset into the middle of a segment (control config). A prediction was written before running it.

**A free identity check.** At the first batch after a reset the model is exactly BN-adapt (source affine parameters, batch statistics, prediction before the update),
so error(method) - error(BN-adapt) on that batch must be 0. The analysis script prints the largest deviation as a test of the wrapper.

**What the Phase 4 data support (and what they do not).** Supported (our setup only: one ResNet-18 checkpoint, CIFAR-10-C severity 5, 15 corruptions x 10 cycles,
Adam lr 1e-3, batch 200): TENT collapses (3/3); the marginal term removes marginal collapse (0/3) but not error degradation; frequent resets (T = 250) beat BN-adapt
(3/3, about 0.022) and T >= 1000 do not. Not supported: that eta < eta* gives stable updates (SGD with c = 0.5 still degrades; the per-segment test is pending), and that the
marginal term is useful by itself. A title promising "stable updates" is therefore ahead of the evidence.

**Why check related work before choosing the claim.** The reset result echoes RDumb, and "reset when collapse is likely" is already a 2026 paper (Lim et al.); the diversity
term was compared in Niu et al. A claim that rests on those would not be new. What might be new is the dynamical toy analysis and the two-failure-mode measurement, which
is why REFERENCES.md keeps the novelty check open instead of declaring it.

**A design trap I should have seen: position in the reset window is confounded with corruption.** The stream repeats 15 corruptions of 50 steps (cycle = 750 steps).
For T = 250 each corruption always sits at the same place in the reset window, so 'error vs steps since reset' mixes the effect of s with the effect of WHICH corruption
is there. Counting connected components of the corruption-by-s-bin graph gives 5 blocks for T in {250, 500, 1000, 2000} (every T is a multiple of 250 and so is
750), so no regression can separate them; 1 block (separable) for T = 50, 100, 350 and for T = 250 with offset 25. The check needs no GPU time: run `--design-check`
BEFORE designing a run. Practical consequence: use the profile for the mechanism only descriptively, use overall error for decisions.

**Break-even is a brittle statistic.** 'First bin from which every later bin is positive' changes with a single noisy bin; sev-3 gave 750 for T1000 and 975 for T2000 on the
same seed and stream (the quantity moved by 225 steps between two runs of one design). Prefer a trailing rolling mean (as in `error_vs_step_noreset.py`) or the
running mean of excess over a window (its minimum is where a reset period would minimise the mean excess if the profile transferred).

## Open questions to verify numerically (later phases, not claimed yet)

- Does the toy stability picture (negative curvature at low margin, `lam > lam_c` fixes it) show up in a real network? (Phases 3-5)
- What is the right curvature / learning-rate scale for BatchNorm parameters in a real network? (Phase 4)
