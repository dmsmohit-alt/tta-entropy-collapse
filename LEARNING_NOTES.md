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

## Open questions to verify numerically (later phases, not claimed yet)

- Does the toy stability picture (negative curvature at low margin, `lam > lam_c` fixes it) show up in a real network? (Phases 3-5)
- What is the right curvature / learning-rate scale for BatchNorm parameters in a real network? (Phase 4)
