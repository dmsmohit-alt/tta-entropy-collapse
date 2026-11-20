# References and verification status

"Verified" = the author (Claude) opened the page named below and read what it states (an abstract page unless noted); it does NOT mean the full paper
was read or that its results were reproduced. Anything not in this table is not cited. Novelty check against the last 12 months: **not complete** (see bottom).

| Work | What we use it for | Status |
|---|---|---|
| Wang, Shelhamer, Liu, Olshausen, Darrell. TENT: Fully Test-Time Adaptation by Entropy Minimization. ICLR 2021. arXiv:2006.10726 | the method; updates BN gamma/beta, batch statistics; text says BS = 128 and LR = 0.001 for non-ImageNet datasets, Adam | verified from the arXiv PDF text (the CIFAR assignment is implied by "other datasets") |
| Press, Schneider, Kuemmerer, Bethge. RDumb: A simple approach that questions our progress in continual test-time adaptation. arXiv:2306.05401 | periodic reset every T = 1000 steps (ImageNet-scale, batch 64, ETA-style loss, SGD lr 2.5e-4); CCC benchmark; claims most TTA methods eventually collapse | verified from the arXiv abstract page and HTML; venue not stated on the pages read (a search listing mentioned NeurIPS 2023: unverified) |
| Lim, Hwang, Lee. When and Where to Reset Matters for Long-Term Test-Time Adaptation. ICLR 2026 | adaptive/selective reset triggered by collapse risk (collapse = few classes predicted); argues periodic resets are not tied to collapse risk | abstract only (mlanthology page); whether it analyses a marginal-entropy term is not stated there |
| Niu et al. Adapt in the Wild: Test-Time Entropy Minimization with Sharpness and Feature Regularization. arXiv:2509.04977 | SAR / SAR2; links collapse to noisy-sample gradients and feature redundancy/inequity; compares its inequity regulariser with the diversity term of IM loss | abstract/page summary only |
| Chen et al. Neural Collapse in Test-Time Adaptation. arXiv:2512.10421 | geometric (sample-wise feature-classifier misalignment) explanation of TTA degradation, NCTTA method | abstract only |
| Mishra. RDumb++: Drift-Aware Continual Test-Time Adaptation. ICLR 2026 workshop poster | drift-triggered resets on CCC | page summary only; single-author workshop paper |
| Liang et al. SHOT (information-maximisation loss with a diversity term) | the `E[H] - lam H(p_bar)` loss is described in the code as "SHOT-style" | **UNVERIFIED**: the fetch was refused; cite only after opening the paper |

## Novelty check (not finished)
Four searches (RDumb-type resets, collapse analyses, diversity regularisation, dynamics of entropy minimisation) turned up the works above. None of their
abstracts mentions a basin map of the collapse, a local-stability step-size bound (eta*), or a separation of marginal collapse from error degradation measured
by a pre-specified error rule. That is NOT evidence of novelty: absence from four searches and abstract-level reading is weak. Still to do: Google Scholar and
arXiv listing search over the last 12 months, and reading the full text of the closest papers (Lim et al., Niu et al., RDumb).
