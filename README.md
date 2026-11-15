# When Does Entropy Minimisation at Test Time Collapse?

A dynamical analysis of test-time entropy minimisation (TENT) and a stable
marginal-entropy correction for continual shift.

**Status: Phases 0-1 done (scaffold + toy model).** Real-network experiments not run yet.
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

### Not yet run

| Experiment | Result |
|---|---|
| eta* (theory vs numerical) | TBD (Phase 2) |
| CIFAR-10-C time-to-collapse (Source / TENT / TENT+marginal / RDumb) | TBD (Phases 3-4) |

## Limitations (to be expanded in Phase 7)

- A linear-softmax result may not fully transfer to deep networks.
- Collapse has been reported empirically by prior work; the contribution here
  must be the sharper dynamical explanation plus the stability condition.
- Novelty versus the last 12 months of arXiv / Google Scholar is **not yet checked**.
