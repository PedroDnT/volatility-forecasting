# Arm B — Ibovespa with IVol-BR

## Source

**IVol-BR**, the Brazilian implied-volatility index built from IBOVESPA options,
published by **NEFIN** (Núcleo de Economia Financeira), FEA-USP.

- Page: https://nefin.com.br/data/volatility-index/
- File: https://nefin.com.br/resources/volatility_index/IVol-BR.csv
- Retrieved: 2026-08-08 → `ivolbr_raw.csv` (44,743 bytes)
- Terms: NEFIN states it makes its data "freely available for academics and
  practitioners", with no stated redistribution restriction, so the snapshot is
  committed here. Cite NEFIN and the methodology paper (Astorino, Chague,
  Giovannetti & Silva, *Variance Premium and Implied Volatility in a
  Low-Liquidity Option Market*) in any published work.

The pipeline re-fetches from that URL on each run and falls back to the committed
snapshot if the fetch fails, so there is no manual download step.

## Coverage

| | |
|---|---|
| Range | 2011-08-01 → 2022-04-29 |
| Rows | 2,415 (2,318 usable) |
| Blank `ivolbr` | 97 (4.0%) |
| Units | percentage points — median 23.09, min 5.30, max 118.52 (COVID) |
| Per year | 204–244 observations |

Format is `year,month,day,ivolbr` — separate integer date columns, not a date
column. `_parse_iv_table()` in `code/01_collect_data.py` handles both shapes.

## The end date drives the design

**IVol-BR stops on 2022-04-29.** Only 71 observations fall after 2022-01-01, so
the original 2022–2025 test window is unusable — it would score against a
forward-filled constant for 95% of the evaluation.

Evaluation therefore runs **2018-01-01 → 2022-04-29** (~1,090 trading days,
against the US study's 980), with training from 2011-08 and the walk-forward
validation block taken from the 252 rows before each batch. `br_long` and
`us_2018` use the identical window so all three are comparable.

If NEFIN later extends the series, update `BR_TEST_END` in `code/config.py` and
re-run; nothing else is date-coupled.

## Data-quality notes

- **Forward-fill ceiling is 12%** for this market (`max_iv_ffill`), against 5%
  elsewhere. IVol-BR carries 4% blanks *and* skips some B3 sessions, so a
  ^VIX-calibrated bound would reject a perfectly good panel. The measured value
  is reported by `01_collect_data.py` and recorded in the run report — check it
  rather than assuming.
- **Scale gate**: `iv_lag1`/`iv_lag5` divide by 100, assuming percentage points.
  The median of 23.09 confirms this; the gate rejects anything outside 1–200.

## Why not `^VXEWZ`

`^VXEWZ` is fetchable straight from Yahoo, which is tempting, but it prices
implied volatility of `EWZ` — a USD-denominated, US-listed Brazil ETF — so it
bundles Ibovespa volatility with BRL/USD exchange-rate volatility. It also
follows the US calendar, which would force forward-filling on every US holiday
that is a B3 trading day, and drop Carnival and Corpus Christi. IVol-BR shares
`^BVSP`'s underlying, currency and calendar.

If the USD-investor question is worth answering later, the coherent pairing is
`EWZ` + `^VXEWZ` — same underlying on both sides — as a separate market entry,
not as a substitute here.
