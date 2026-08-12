# Brazilian replication

Tests whether the findings of Khan (2026), [SSRN 6663418](https://ssrn.com/abstract=6663418),
hold on Brazilian data. US and Brazil run through identical model code —
everything market-specific is data in [`code/config.py`](code/config.py).

## Markets

| Market | Series | Sample | Test window | Features |
|---|---|---|---|---|
| `us` | `^GSPC` + `^VIX` | 2004–2025 | 2022-01 → 2025-11 | 30 |
| `us_2018` | `^GSPC` + `^VIX` | 2004–2025 | 2018-01 → 2022-04 | 30 |
| `br_long` | `^BVSP` | 2004–2022 | 2018-01 → 2022-04 | 26 |
| `br_iv` | `^BVSP` + **IVol-BR** | 2011-08 → 2022-04 | 2018-01 → 2022-04 | 30 |

`br_iv` is the main Brazilian arm. **IVol-BR** is built from IBOVESPA options and
published by [NEFIN, FEA-USP](https://nefin.com.br/data/volatility-index/) — same
underlying, currency and exchange calendar as `^BVSP`. Committed snapshot:
2011-08-01 → 2022-04-29, 2,415 rows (2,318 usable, 4.0% blank), percentage points,
median 23.1. NEFIN makes the data freely available; the pipeline re-fetches it from
their URL, so no manual download step.

**The test window is set by where IVol-BR ends, not by preference.** Only 71 IVol-BR
observations fall after 2022-01-01, so the original 2022–2025 window is unusable.
Evaluation moves to 2018-01 → 2022-04 (~1,090 days, close to the US study's 980).
`br_long` and `br_iv` share that window exactly, so their QLIKE numbers cover
identical days and the pair isolates what implied vol is worth. `us_2018` exists so
Brazil-vs-US compares the same period rather than different regimes.

## Status

`us` and `us_2018` are done and audited. The Brazilian arms need `^BVSP`, which the
build environment could not fetch (Yahoo HTTP 429 on every ticker; Stooq behind a
JavaScript bot-wall). Everything else is built and tested — run this on any machine
with network access:

```bash
pip install -r requirements.txt
python code/01_collect_data.py             --market br_iv
python code/03_run_core_models.py          --market br_iv
python code/04_subperiod_and_importance.py --market br_iv
python code/05_dm_tests.py                 --market br_iv
python code/05b_mcs_spa.py                 --market br_iv
python code/06_audit.py                    --market br_iv --regenerate
```

Swap `br_iv` for `br_long` and repeat. Artifacts land in `data/<market>/` and
`results/<market>/`; root-level `data/` and `results/` are the original US snapshot,
untouched. `python -m pytest tests/` runs the suite.

If Yahoo is still rate-limiting, skip it with a CSV export — `Date,Open,High,Low,
Close,Volume`, ISO or Brazilian `dd/mm/yyyy`, Portuguese headers accepted:

```bash
python code/01_collect_data.py --market br_long --equity-csv ~/Downloads/bvsp.csv
```

One open question remains: whether `^BVSP` volume is usable. Two features depend on
it, and `01_collect_data.py` disables them automatically above a 1% bad-row rate —
see [`data/br_long/PROVENANCE.md`](data/br_long/PROVENANCE.md).

## Two corrected defects

1. **Validation was nested inside training** — `train = df[:batch_start]` while
   `val = df["2019":"2021"]`. Early stopping never fired; both tree models ran the
   full 500 rounds every batch. Now a walk-forward validation block precedes each
   batch with an embargo for the 5-day target. Early stopping fires in every fit.
2. **GARCH reported an in-sample filter, not a forecast** — parameters were fit
   through the *end* of each batch and scored on that batch. Now fit strictly
   beforehand, held fixed, producing a genuine 5-day forecast.

## Findings so far

Full-sample QLIKE, lower is better. The US column is the corrected re-run, which is
what Brazil will be compared against — not the published numbers.

| Model | Published | `us` corrected | `us_2018` |
|---|---:|---:|---:|
| GJR-GARCH | 0.3447 | **0.3135** | 0.5625 |
| XGBoost | 0.3553 | 0.3351 | **0.4976** |
| LightGBM | 0.3632 | 0.3356 | 0.4990 |
| Ensemble | **0.3431** | 0.3401 | 0.5327 |
| GARCH | 0.3806 | 0.3572 | 0.5430 |
| EGARCH | 0.3748 | 0.3573 | 0.5723 |
| HAR-RV | 0.4198 | 0.4221 | 0.7606 |

**The published ranking does not survive correcting the code.** The ensemble falls
from 1st to 4th — its two weak members drag it down once the individual models are
fixed. HAR-RV is the only model that does not improve, the expected control since
neither defect touched it.

**The subperiod reversal dissolves, then reappears with the opposite sign.** On
2022–2025, corrected GJR-GARCH wins *both* subperiods, so "the winner flips with the
regime" does not hold. On 2018–2022 the trees win outright and GJR-GARCH is 5th.
Which family wins depends on the evaluation window, not on the volatility regime
within it. Worth settling before drawing conclusions from the Brazilian arms.

Joint tests agree the field is not separable: the 95% MCS eliminates only HAR-RV on
both windows, and GJR-GARCH's edge over the trees on `us` is insignificant
(DM p = 0.36–0.40).

**Caveat.** Fix 2 bundles two changes — removing look-ahead (should hurt GARCH) and
repairing a 1-day-vs-5-day horizon mismatch (should help it). They are not separately
identified; splitting them needs a third configuration.

## Notes on the Brazilian setting

- **`sqrt(252)` stays.** 252 *dias úteis* is the standard Brazilian convention, not a
  US leftover.
- **B3 circuit breakers** (−10%/−15%, six triggers in March 2020) truncate
  close-to-close realized volatility exactly in the high-vol regime.
- **`^BVSP` volume is unreliable** on Yahoo. Two features depend on it;
  `01_collect_data.py` disables them above a 1% bad-row rate and says so.
- **IVol-BR over `^VXEWZ`**: VXEWZ prices a USD claim (bundling FX vol) and follows
  the US calendar, which would force forward-filling on every US holiday that is a
  B3 trading day.

## Reproducibility

`requirements.txt` pins every dependency — unpinned, XGBoost's QLIKE moved 0.3553 →
0.3487, four times the paper's headline margin. Seeds are explicit.
`06_audit.py --regenerate` re-runs the pipeline and diffs against committed artifacts,
which the original audit could not do. Expected values live in
`results/<market>/expected.json`; the published claims are preserved in
`results/expected_paper.json` and still pass against the legacy snapshot.
`05b_mcs_spa.py` implements MCS/SPA/Reality Check via `arch.bootstrap`, replacing
results attributed to a package that is not on PyPI. `tests/` guards the two fixes by
asserting on window boundaries.
