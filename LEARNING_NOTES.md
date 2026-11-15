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

## Open questions to verify numerically (later phases, not claimed yet)

- Is the collapse threshold a sharp boundary in initial bias? (Phase 1)
- What is eta* and does the analytic value match the numerical one? (Phase 2)
