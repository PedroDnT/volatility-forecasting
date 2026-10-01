# Replication audit: volatility forecasting horse race (Khan, 2026)

This fork re-runs [Khan (2026), SSRN 6663418](https://ssrn.com/abstract=6663418) and corrects two protocol
errors in the model code. The original paper, code and data are by Akram Khan and are kept below, unchanged.
The audit, the corrected pipeline and the Brazilian extension are by
[Pedro Del Nero Todescan](https://github.com/PedroDnT).

## What was wrong

1. **Validation nested inside training.** The tree models validated on 2019–2021, a window that was also in
   their training data. Early stopping never fired, so LightGBM and XGBoost trained the full 500 rounds on
   every batch. Fix: a walk-forward validation block before each batch, with an embargo covering the 5-day
   target. Early stopping now fires in 15 of 15 fits, at a median of 38 rounds.
2. **GARCH scored in-sample.** GARCH parameters were fit through the end of each test batch, and the in-sample
   filtered volatility was then scored on that same batch: a 1-day filter scored against a 5-day target.
   Fix: parameters estimated strictly before each batch and held fixed while the filter runs forward,
   producing a genuine 5-day forecast.

## What changes

S&P 500, test window Jan 2022 to Nov 2025 (980 days), QLIKE, lower is better.

| Model | Published | Corrected |
|---|---:|---:|
| GJR-GARCH | 0.3447 | **0.3135** |
| XGBoost | 0.3553 | 0.3351 |
| LightGBM | 0.3632 | 0.3356 |
| Ensemble | **0.3431** | 0.3401 |
| GARCH(1,1) | 0.3806 | 0.3572 |
| EGARCH | 0.3748 | 0.3573 |
| HAR-RV | 0.4198 | 0.4221 |

- **The published ranking does not hold.** The ensemble falls from 1st to 4th and GJR-GARCH ranks first.
- **The top of the table is not separable.** GJR-GARCH's edge over the trees is not significant
  (Diebold-Mariano p = 0.36 vs LightGBM, 0.41 vs XGBoost), and the 95% Model Confidence Set eliminates only HAR-RV.
- **The calendar reversal does not hold either.** The paper reports GARCH models winning 2022 and trees winning
  2023–2025. Corrected, a GARCH-family model wins both: EGARCH in 2022, GJR-GARCH in 2023–2025. Split by
  realized-volatility regime instead, GARCH-family models lead in high volatility and the trees in lower volatility.
- **HAR-RV is the control.** Neither fix touches it, and its score barely moves.

**Caveat.** Fix 2 bundles two effects: removing look-ahead (which should hurt GARCH) and repairing the
1-day vs 5-day horizon mismatch (which should help it). They are not separately identified.

## Reproduce

```bash
pip install -r requirements.txt   # pinned; unpinned XGBoost drifts by ~4x the paper's headline margin
python code/01_collect_data.py --market us
python code/03_run_core_models.py --market us
python code/04_subperiod_and_importance.py --market us
python code/05_dm_tests.py --market us
python code/05b_mcs_spa.py --market us
python code/06_audit.py --market us --regenerate   # reruns the pipeline and diffs against committed results
```

Corrected results live in `results/us/`. The published numbers are preserved in `results/expected_paper.json`,
and the original snapshot under `data/` and `results/` at the root is untouched. The Brazilian extension
(Ibovespa with the NEFIN IVol-BR implied-volatility index) is described in [README_BR.md](README_BR.md).

---

*Original README by Akram Khan follows.*
