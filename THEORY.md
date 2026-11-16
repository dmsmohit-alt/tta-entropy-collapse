# THEORY: bias dynamics of entropy minimisation in a linear-softmax model

Status labels used on every claim:

- **[derived]** follows from the math written here (and, where noted, cross-checked numerically)
- **[verified]** a derived formula that was checked against an independent numerical computation (run id given)
- **[observed]** only seen in simulations; no proof
- **[assumption]** a modelling choice or a convention we made
- **[TBD]** not done yet / needs a run on another machine

All numbers below come from saved runs: `results/toy_basin/489889f9` (Phase 1),
`results/toy_stability/5dd255d1` (Phase 2), `results/toy_infomax_basin/a28b7df4` (Phase 2, toy infomax).
Citations are intentionally not included yet (see section 9).

---

## 1. Setup

- Model: `p(y|x) = softmax(Wx + b)`, `K` classes. **[assumption]** Only the bias `b` is adapted
  in sections 2-7 (W frozen); Phase 1 also ran adapting both (observation only, section 7).
- Data: balanced mixture, `x | y=j ~ N(mu_j, sigma^2 I)`, **[assumption]** `mu_j` orthogonal with equal norm `sep`
  (needs `d >= K`), and `W` = the Bayes (LDA) classifier `W_j = mu_j / sigma^2`.
- Margin `m := sep^2 / sigma^2`.
- Objective (information-maximisation, as in SHOT):
  `J(b) = E_x[H(p)] - lam * H(p_bar)`, `p_bar = E_x[p]`. `lam = 0` is plain entropy minimisation (TENT loss).
- Dynamics: gradient flow `b_dot = -grad J(b)`; discrete GD `b <- b - eta * grad J(b)`.
- Notation: `l_k = log p_k`, `H = -sum_k p_k l_k`, `P = I - 11^T/K` (projector onto 1-perp).

## 2. Entropy gradient and the bias flow  **[derived]**

For one sample, `p = softmax(z)`: `dp_k/dz_j = p_k (delta_kj - p_j)`, so

    dH/dz_j = - sum_k (dp_k/dz_j)(l_k + 1) = -p_j (l_j + H)        (the "+1" drops because sum_k dp_k = 0)

Since `z = Wx + b`, `dz/db = I`, therefore for `lam = 0`:

    b_dot_j = E_x[ p_j (l_j + H) ]                                   (1)

(Checked against finite differences, `tests/test_toy_model.py`, and `tests/test_dynamics.py`.)

**What (1) says (sign argument, [derived]).** Because `H >= -log p_max` (min-entropy bound),
`l_top + H >= 0` for the top class of every sample. So every sample pushes the bias of *its own
currently most-likely class up* (weighted by `p_top`), and pushes up any class whose
probability exceeds `e^{-H}`; classes with `p_j < e^{-H}` are pushed down. This is the per-sample
"rich get richer" effect.
**What it does not say:** (1) alone does not prove collapse. Samples from the other classes push
their own classes up too; whether the net drift on the *gap* between class 0 and the rest is positive
depends on the full distribution. That is what section 4 computes.

## 3. The marginal term  **[derived]**

`H(p_bar) = -sum_k pbar_k log pbar_k`, `dpbar_k/db_j = E[p_k(delta_kj - p_j)]`. Again `sum_k dpbar_k = 0`, so

    d H(p_bar)/d b_j = - E[ p_j ( log pbar_j - sum_k p_k log pbar_k ) ]

and for `J = E[H] - lam H(p_bar)`:

    b_dot_j = E[ p_j (l_j + H) ]  -  lam * E[ p_j ( log pbar_j - <log pbar>_p ) ]     (2)

with `<f>_p = sum_k p_k f_k`. The second term pushes *down* the bias of any class that is already over-represented
in `p_bar` (`log pbar_j` above its p-weighted average): "rich get poorer", opposing (1).
(Verified: `bias_flow_from_logp` equals minus the finite-difference gradient of `J`, error ~1e-10.)

## 4. Gauge direction and the exact 1-D reduction

**[derived] Gauge.** `softmax(z + c1) = softmax(z)`, so `J(b + c1) = J(b)`: the Hessian always has a zero eigenvalue along `1`
and `sum_j b_dot_j = 0`. That direction is neutral by construction, not an instability. All stability statements are on `1^perp`.

**[derived] Population logits.** With `W_j = mu_j/sigma^2` and orthogonal equal-norm means:
`z_k = mu_k . x / sigma^2 = m * 1[k = y] + sqrt(m) * xi_k`, with `xi_k = mu_k.eps / (sigma sep)` iid `N(0,1)`.
So the population problem depends only on `(K, m, b)`; `d` and `sigma` drop out except through `m`.
(Verified against sampled logits: class means within 0.15 of `m*e_y`, covariance within 0.4 of `m I`, 20000 points per class.)

**[derived] Reduction.** Take `b = (g, 0, ..., 0)`. The population law of the logits is invariant under permuting
classes `1..K-1`, so `b_dot_1 = ... = b_dot_{K-1}` and the subspace is invariant. With `g := b_0 - b_others`:

    g_dot = b_dot_0 - b_dot_1  =: F_lam(g)                                            (3)

This is an exact 1-D flow for the population. The basin boundary is its first upward zero crossing `g_c`
(`g_0 < g_c` returns to balance, `g_0 > g_c` grows). **[assumption]** `F_lam` is evaluated by Monte Carlo
(`Population`, 100k xi-draws x K classes, common random numbers) and `g_c` is found with a root finder;
no closed form is known to us.

## 5. The balanced point: stationarity, Hessian, step-size limit

**[derived] Stationary.** At `b = 0` the population is invariant under all permutations of the classes, so `b_dot` is
proportional to `1`; with the gauge, `b_dot = 0`. (On a finite sample this holds only approximately: section 6.)

**[derived] Hessian structure.** The Hessian of `J` at `b = 0` commutes with all permutations and kills `1`, so it equals
`alpha * P`: one eigenvalue `alpha` on `1^perp` (multiplicity `K-1`).

- Entropy part. Per sample, Hessian of `H` w.r.t. `z`: `A_kj = -delta_kj p_k (l_k+H+1) + p_k p_j (l_k + l_j + 2H + 1)`.
  `tr A = -1 + sum_k p_k^2 (2 l_k + 2H + 1)`, hence
  `alpha_L = E[ tr A ] / (K-1)`.
- Marginal part. `S := E[diag(p) - pp^T] = s P` with `s = E[1 - sum p^2]/(K-1)`. At `b=0`, `pbar = 1/K`, the
  second-derivative term proportional to `(log pbar_k + 1)` vanishes (it is a constant times `sum_k T_kij`, and `sum_k T_kij = 0`),
  leaving `Hess H(p_bar) = -K S^T S = -K s^2 P`.
- Therefore

      alpha_J = alpha_L + lam * K * s^2                                  (4)

  and the balanced point is a strict local minimum of `J` on `1^perp` iff `alpha_J > 0`.

**[derived] Critical lambda.** `lam_c = max(0, -alpha_L / (K s^2))`: any `lam > lam_c` makes the balanced point locally stable.

**[derived] Step-size limit.** GD map Jacobian on `1^perp` is `I - eta*Hess`, eigenvalues `1 - eta*alpha_i`.
Stable iff all `alpha_i > 0` and `eta < eta* = 2 / alpha_max`. Population: `eta* = 2 / alpha_J`.
**[assumption]** This is local linear stability of full-batch GD on the *mean loss*, bias-only. The numerical value
of `eta*` is in units of this objective and parametrisation and is NOT comparable to learning rates of BatchNorm
parameters in a real network (that needs its own curvature estimate: Phase 4).

### Numbers (population, Monte Carlo, K=5; MC std of `alpha_L` is about 1e-4 at sep 2: 6 extra seeds gave -0.00106 +/- 0.00009)

| sep | margin m | alpha_L (lam=0) | s | lam_c | eta* at lam=1 (pop) |
|---|---|---|---|---|---|
| 1.0 | 1 | -0.0990 | 0.160 | 0.78 | 69.7 |
| 1.5 | 2.25 | -0.0345 | 0.117 | 0.50 | 58.3 |
| 2.0 | 4 | -0.00096 | 0.075 | 0.03-0.04 | 73.8 |
| 2.5 | 6.25 | +0.0086 | 0.042 | 0 | 115.8 |
| 3.0 | 9 | +0.0074 | 0.020 | 0 | 210.1 |
| 4.0 | 16 | +0.0018 | 0.0032 | 0 | 1056 |
| 5.0 | 25 | +0.0002 | 0.0003 | 0 | 9747 |

(Source: `population.csv`, `mc_seed` 0/1.)

- **[observed]** `alpha_L` changes sign between sep 2.0 and 2.1 (margin about 4.2 for K=5; 4 mc seeds each, std ~1e-4).
  For margins below that, the balanced point of plain entropy minimisation is unstable, consistent with Phase 1:
  sep 1 and 2 collapsed at `delta = 0` (3/3 seeds). No closed form for the critical margin is derived.
- **[observed]** For large separation both `alpha_L` and `s` shrink fast (predictions are almost one-hot), so the
  *local* effect of `lam` on `alpha_J` is negligible for sep >= 4 within `lam <= 2`, and `eta*` becomes huge. The
  stabilising effect of the marginal term then comes from the *global* flow (section 7), not from the local curvature.

## 6. Numerical verification (run `toy_stability/5dd255d1`, 3 data seeds x 7 separations x 4 lambdas = 84 rows, N=1000 each)

| claim | check | result |
|---|---|---|
| Hessian formula (valid at any b) | analytic vs central finite differences at the stationary point | max abs error 3.4e-11 (all 84 rows) |
| Hessian formula | analytic vs PyTorch autograd (`torch_hessian`) | **[TBD]** not run in the author's sandbox (no torch there); `tests/test_dynamics.py::test_hessian_matches_torch_autograd_if_available` runs on a machine with torch |
| `eta* = 2/lambda_max(Hess)` | bisection on a *nonlinear* GD perturbation experiment (independent of the Jacobian) | 57 near-balanced stable rows: max relative error 0.19%, 0 failures to bracket |
| spectral radius of `I - eta Hess` | at `eta = 0.5, 0.9, 1.1, 1.5 x eta*` | 0.34 (median), 0.80, 1.20, 2.00 as predicted (this is arithmetic on the eigenvalues; the independent check is the row above) |
| `alpha_J > 0` iff stable | population prediction vs plain GD from `b=0`, 3000 steps | 84/84 rows agree, **but** for sep 5 the curvature is ~2e-4, so 3000 steps cannot reveal an instability: that part of the agreement is uninformative |

**Finite-sample caveats ([observed]).**
- With N=1000 the sample Hessian differs from the population one. The sample `eta*` is lower than the population one:
  ratio mean 0.79, std 0.14 over the 57 stable rows (decreasing with sep; 0.61 at sep 4).
- For sep 5 the population `alpha_L = +0.0002`, but the sample Hessian at `b=0` has a negative eigenvalue (about -0.0002 on average):
  finite-sample noise exceeds the curvature. So at large separation the "stable balanced point" is only meaningful for very large N.
- Newton from `b=0` can leave the balanced region when the balanced point is unstable (it jumped to flat collapsed stationary points
  with gap -9 to -51 in an earlier version of the code). The final code caps the step and records `near_balanced`
  (`|gap| < 2` and `|grad| < 1e-9`, **[assumption]** thresholds chosen by us). Of 84 rows, 76 are near-balanced
  (57 stable, 19 unstable) and 8 have no converged nearby stationary point; for those the curvature at exactly `b=0`
  (`eig_*_at0` in `stability.csv`) is the quantity to read.

## 7. Basin boundary (collapse threshold)

Population root `g_c` of the reduced flow (3), `lam = 0`, versus the Phase 1 simulation (bias-only GD from the Bayes classifier plus `delta` on class 0):

| sep | margin | population `g_c` (MC) | Phase 1 simulated `delta*`, mean +/- std over 3 data seeds |
|---|---|---|---|
| 2.5 | 6.25 | 4.65 | not simulated |
| 3 | 9 | 8.18 | 7.61 +/- 0.50 |
| 4 | 16 | 15.88 | 16.34 +/- 0.15 |
| 5 | 25 | 25.01 | 25.85 +/- 0.34 |

- **[verified]** The population reduced flow reproduces the Phase 1 thresholds to within the seed-to-seed spread
  (differences +0.56, -0.46, -0.84 for sep 3, 4, 5; Phase 1 std 0.50, 0.15, 0.34: in all three cases the gap is larger than one std,
  which we attribute, without having tested it, to finite-sample effects at N=1000).
- **[verified]** Sample level: the zero crossing of the *finite-sample* 1-D flow matches the per-seed simulated `delta*` within 0.12 for 5 of 9 seeds with sep >= 3
  and within 0.6 for 7 of 9; for 2 of 9 seeds (sep 3 seed 0, sep 4 seed 0) the 1-D symmetric reduction does not hold for the sample
  (flow already positive near `g = 0`, asymmetry among the non-biased classes), see `sample_roots.csv`.
- **[observed]** `g_c / m` tends to 1 for large margin (0.91, 0.99, 1.00 at sep 3, 4, 5 in the population; Phase 1 simulation: 0.85, 1.02, 1.03). We have no derivation of why.
- **[observed]** With `lam >= 0.5` the reduced population flow has no positive-gap zero crossing in the scanned range `g in [0, 1.5 m]`
  for sep >= 2 (`g_c_status = no_root` in `population.csv`): along this axis there is no collapse basin up to 1.5 margins.
- **[observed] Toy check of the fix** (`toy_infomax_basin/a28b7df4`, GD from the biased start, eta 2, 3000 steps, 3 seeds,
  bias-only; initial bias `delta = rel x margin`):

  | sep | lam | rel=0.5 | 0.9 | 1.2 | 1.5 | 2.0 |
  |---|---|---|---|---|---|---|
  | 3 | 0 | 0/3 collapse | 2/3 | 3/3 | 3/3 | 3/3 |
  | 3 | 0.5, 1, 2 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
  | 4 | 0 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 |
  | 4 | 0.5, 1, 2 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |

  Final accuracy with `lam >= 0.5` equals the unbiased Bayes-classifier value (0.947 at sep 3, 0.990 at sep 4) for every start.
  Phase 1, adapting `W` and `b` (sep 3, `lam = 0`): collapse delayed to `delta` near 10.8 (> margin 9); intermediate final states have `0 < H(p_bar) < log K`;
  whether those are true fixed points is **[TBD]**.

## 8. What is derived, observed, assumed (summary)

| item | status |
|---|---|
| Gradient of H, flows (1), (2), gauge zero mode | derived; flow/gradient/Hessian verified vs finite differences |
| Top-class sign argument for "rich get richer" | derived (sign only; not a collapse proof) |
| Population logit law, exact 1-D reduction (3) | derived; logit statistics verified on samples |
| `Hess = alpha P`, `alpha_J = alpha_L + lam K s^2`, `lam_c`, `eta* = 2/alpha` | derived; `eta*` verified by nonlinear GD (0.19% max error) |
| Numerical values of `alpha_L(m)`, `s(m)`, `g_c(m)` | derived formulas, Monte-Carlo evaluated (MC error ~1e-4 on `alpha_L`) |
| Sign change of `alpha_L` near margin 4.2 (K=5) | observed |
| `g_c ~ m` for large `m`; closed form for `g_c` | observed / not derived |
| No collapse basin for `lam >= 0.5` (toy) | observed (bias-only, K=5, orthogonal means, 3 seeds) |
| Phase 1 sweeps and thresholds | observed (see README) |
| Collapse definition (`H(p_bar) < 0.1 log K`), `near_balanced` thresholds, orthogonal equal-norm means, Bayes init, full-batch GD | assumptions |

## 9. Limitations and things to verify before claiming anything

- Everything above is for a bias-only linear-softmax model with symmetric, well-separated Gaussian classes. Transfer to deep networks is a hypothesis to be tested in Phases 3-5, not a result.
- `eta*` values here (10 to 10^4) are properties of this toy objective. No claim is made about the learning rate for BatchNorm parameters in a real network.
- Only K=5 and 3 data seeds. The sign-change margin (about 4.2) is for K=5 only.
- Related work (TENT, SHOT's information-maximisation loss, RDumb, and papers reporting TTA collapse) is **not** cited here:
  author names, venues and years must be verified before any citation, and a novelty check against the last 12 months of arXiv/Scholar is still pending.
