# Brazilian replication — status and findings

A multi-market rebuild of the volatility horse race, testing whether the
findings of Khan (2026) — [SSRN 6663418](https://ssrn.com/abstract=6663418) —
hold on Brazilian data.

## Status

| Market | What it is | Status |
|---|---|---|
| `us` | `^GSPC` + `^VIX`, 2004–2025, 30 features | **Done.** Corrected-pipeline baseline in `results/us/` |
| `br_long` | `^BVSP`, 2004–2025, no implied vol (Arm A) | Blocked on step 0 |
| `br_iv` | `^BVSP` + IVol-BR, ~2012–2025, 30 features (Arm B) | Blocked on step 0 |

**Step 0 is a data-availability spike that needs network access.** It was not
possible in the environment this was built in: Yahoo returns HTTP 429 for every
ticker, including `^GSPC`, and `cef.fgv.br` is unreachable. Run
`python code/00_probe_sources.py` on a networked machine and fill in
`data/br_long/PROVENANCE.md` and `data/br_iv/PROVENANCE.md`. Until then, the
Brazilian sample dates in `code/config.py` are provisional and IVol-BR's
redistribution terms are an open licensing question.

Everything that does not need the network is built, run and verified.

## Design

Both Brazilian arms exist because Brazil has no implied-volatility index before
roughly 2011, and in the US study `vix_close` is the single most-split feature.
So Brazil forces a choice between history and features — and the two arms turn
that constraint into the measurement.

- **Arm A (`br_long`)** — 2004–2025, drop the four implied-vol features. Keeps
  2008, the 2014–16 recession, Joesley Day and COVID in training.
- **Arm B (`br_iv`)** — ~2012–2025, full feature parity with the US study.

They share the same validation window and the same test window, so their QLIKE
numbers are computed over **identical evaluation days**. The A-vs-B difference
is therefore a clean estimate of what the implied-vol block is worth, not an
artifact of different samples.

US and Brazil run through byte-identical model code. Everything that differs
between markets lives in `code/config.py` as data. Forking `code/` into a
`code_br/` would have confounded every Brazil-vs-US difference with
implementation drift.

## Running it

```bash
pip install -r requirements.txt

python code/00_probe_sources.py                      # step 0 (needs network)

python code/01_collect_data.py            --market br_long
python code/03_run_core_models.py         --market br_long
python code/04_subperiod_and_importance.py --market br_long
python code/05_dm_tests.py                --market br_long
python code/05b_mcs_spa.py                --market br_long
python code/06_audit.py                   --market br_long --regenerate
```

Swap `br_long` for `br_iv` or `us`. Artifacts land in `data/<market>/` and
`results/<market>/`. The legacy root-level `data/` and `results/` are the
original US snapshot and are left untouched.

The `us` market reads the committed root-level panel, since a fresh Yahoo pull
was not possible here:

```bash
python code/03_run_core_models.py --market us --input data/combined.parquet --out results/us
```

---

## Finding: the published US result does not survive correcting the code

Before touching Brazilian data, the corrected pipeline was re-run on the US
panel, so that Brazil-vs-US would be a comparison of like with like rather than
against published numbers produced by different code. That re-run is itself the
most important result so far.

Two defects were corrected (details in `code/03_run_core_models.py`):

1. **The validation set was nested inside the training set.** Early stopping
   never fired — all fifteen LightGBM fits ran the full 500 rounds. Corrected,
   early stopping fires in **15/15** fits at a median of **38** rounds. Both
   tree models were badly overfit.
2. **GARCH reported an in-sample filtered volatility, not a forecast.**
   Parameters were estimated on data through the end of each batch and then
   scored on that same batch.

### Full-sample QLIKE (lower is better)

| Model | Published | Corrected | Δ |
|---|---:|---:|---:|
| GJR-GARCH | 0.3447 | **0.3135** | −0.0312 |
| XGBoost | 0.3553 | 0.3351 | −0.0203 |
| LightGBM | 0.3632 | 0.3356 | −0.0276 |
| Ensemble | **0.3431** | 0.3401 | −0.0030 |
| GARCH | 0.3806 | 0.3572 | −0.0234 |
| EGARCH | 0.3748 | 0.3573 | −0.0175 |
| HAR-RV | 0.4198 | 0.4221 | +0.0023 |

**The ensemble goes from 1st to 4th.** The paper's headline — that an
equal-weight ensemble of LightGBM + HAR-RV + GARCH narrowly leads the field —
is an artifact of the two defects. Once the individual models are fixed, the
ensemble's two weak members (HAR-RV and plain GARCH) drag it below the models
it was meant to beat. HAR-RV is the only model that does not improve, which is
the expected control: neither defect touched it.

### The subperiod reversal largely dissolves

| Model | 2022 published | 2022 corrected | 2023–25 published | 2023–25 corrected |
|---|---:|---:|---:|---:|
| EGARCH | **0.2346** | **0.2292** | 0.4230 | 0.4014 |
| GJR-GARCH | 0.2597 | 0.2553 | 0.3740 | **0.3335** |
| Ensemble | 0.2616 | 0.2593 | 0.3712 | 0.3680 |
| GARCH | 0.2579 | 0.2595 | 0.4228 | 0.3908 |
| XGBoost | 0.3148 | 0.2902 | **0.3693** | 0.3505 |
| LightGBM | 0.3429 | 0.2975 | 0.3702 | 0.3488 |
| HAR-RV | 0.3123 | 0.3139 | 0.4568 | 0.4593 |

The paper's central claim is a regime-dependent reversal: the GARCH family wins
the high-volatility 2022 subperiod, tree models win the calmer 2023–25
subperiod. Corrected, **GJR-GARCH wins 2023–25 as well** (0.3335 against
XGBoost's 0.3505). GARCH still wins 2022. The trees improve a lot, but GJR
improves more. What survives is "GARCH wins", not "the winner flips with the
regime".

This matters for the Brazil study: the hypothesis it was going to test is not
the one the published paper states.

### Joint tests

The paper's §6.3 shipped MCS/SPA/Reality Check numbers attributed to a
`vol-eval 0.1.0` package that is not on PyPI, with no producing script in the
repository. `code/05b_mcs_spa.py` replaces it using `arch.bootstrap`, which
implements all three. On the corrected US results (2000 stationary-bootstrap
replications, QLIKE):

- **90% MCS** eliminates GARCH and HAR-RV; five models survive.
- **95% MCS** eliminates only HAR-RV.
- **SPA with GJR-GARCH as benchmark**: p_consistent = 0.87 — not dominated by
  anything in the panel.
- Pairwise DM: GJR-GARCH beats GARCH, EGARCH and HAR-RV significantly
  (p < 0.01), but its edge over the trees and the ensemble is **not**
  significant (p = 0.36–0.40).

So GJR-GARCH leads on point estimates and cannot be beaten, but the panel still
cannot separate it from the tree models. The paper's "no model significantly
dominates" conclusion survives even though its ranking does not.

### Caveat: Fix 2 bundles two changes

Removing the GARCH look-ahead also forced a horizon repair. The original scored
a **one-day** conditional volatility against a **five-day** forward target;
once parameters are frozen, the natural object is a genuine multi-step
forecast, so the corrected code predicts σ̂₅ = √(mean(σ̂²_{t+1..t+5})) × √252.

These two changes push in opposite directions — removing look-ahead should hurt
GARCH, matching the horizon should help it — and the net effect is a large
improvement. **They are not separately identified in the current results.**
Splitting them would need a third configuration; say the word and it is a small
addition.

---

## Reproducibility changes

The earlier audit of this repository found that the shipped artifacts passed
their own audit while a fresh run of the same pipeline did not. Addressed:

- **`requirements.txt` with `==` pins.** Unpinned, XGBoost's QLIKE moved 0.3553
  → 0.3487 between library versions — about four times the paper's headline
  Ensemble-vs-GJR gap.
- **Explicit seeds** for LightGBM and XGBoost, instead of relying on library
  defaults that happen to be deterministic.
- **`06_audit.py --regenerate`** re-runs the pipeline into a temp directory and
  diffs against the committed artifacts. The original audit only compared paper
  text against committed files and never re-ran a model, so it structurally
  could not catch regeneration drift. Currently **150 checks pass**, with every
  model reproducing bit-identically.
- **Expected values moved out of code** into `results/<market>/expected.json`,
  so one generic audit serves all three markets. The published US claims are
  preserved in `results/expected_paper.json`:
  ```bash
  python code/06_audit.py --market us --results-dir results \
      --expected results/expected_paper.json    # 120 checks, still passes
  ```
- **Look-ahead assertions at runtime.** Every GARCH fit window must end before
  its batch starts; train/val/test ordering and the embargo are checked for
  every tree batch. `fit_metadata.json` records the windows so the audit can
  re-verify them independently.
- **Data-quality gates** in `01_collect_data.py`: volume reliability, implied-vol
  scale, forward-fill fraction, dropna losses, minimum test rows. It refuses to
  write a degenerate panel rather than producing one quietly — which matters far
  more for `^BVSP` than it ever did for `^GSPC`.
- **Retry with backoff** and `--use-cache` on downloads. The original failed a
  rate-limited pull with `IndexError: index 0 is out of bounds` raised from a
  print statement.
- **`dropna` scoped** to the columns the study uses, rather than dropping any
  row with a NaN anywhere — the original discarded ~22 usable rows from the end
  of every sample because of an unused 22-day target.

A `--legacy-quirks` flag restores the original behaviour. It exists only for
the regression test that proves this refactor is behaviour-preserving: under it
LightGBM reproduces the committed forecasts bit-identically, HAR-RV to machine
epsilon, and the GARCH family to ~1e-8. It is not an analysis arm, and the audit fails any run
whose `fit_metadata.json` records it.

## Files

```
code/config.py                  MarketConfig + registry; the only place markets differ
code/00_probe_sources.py        step 0 data-availability spike (needs network)
code/audit_core.py              market-agnostic audit primitives
code/05b_mcs_spa.py             MCS / SPA / Reality Check via arch.bootstrap
requirements.txt                pinned environment
results/us/                     corrected US baseline
results/expected_paper.json     published claims, for auditing the legacy snapshot
data/br_long/PROVENANCE.md      Arm A — step-0 checklist
data/br_iv/PROVENANCE.md        Arm B — step-0 checklist + licensing blocker
```
