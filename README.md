# When Does Entropy Minimisation at Test Time Collapse?

A dynamical analysis of test-time entropy minimisation (TENT) and a stable
marginal-entropy correction for continual shift.

**Status: Phase 0 (scaffold) only.** No experiments have been run yet.
All results below are `TBD` until produced by a saved run (config + log).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Configs live in `configs/`. Override any value from the command line with
dotted keys, e.g. `stream.batch_size=64` (from Phase 1 onward, via the scripts).

## Results

| Experiment | Result |
|---|---|
| Toy-model basin boundary | TBD |
| eta* (theory vs numerical) | TBD |
| CIFAR-10-C time-to-collapse (Source / TENT / TENT+marginal / RDumb) | TBD |

## Limitations (to be expanded in Phase 7)

- A linear-softmax result may not fully transfer to deep networks.
- Collapse has been reported empirically by prior work; the contribution here
  must be the sharper dynamical explanation plus the stability condition.
- Novelty versus the last 12 months of arXiv / Google Scholar is **not yet checked**.
