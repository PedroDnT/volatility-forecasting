# Arm B — Ibovespa with IVol-BR implied volatility

**Status: NOT YET POPULATED.** This directory is empty pending step 0, and
this arm has a licensing question that must be resolved before any data file
is committed here.

## What this arm is

`^BVSP` plus **IVol-BR**, giving full 30-feature parity with the US study, at
the cost of training history: IVol-BR does not reach back to 2004.

IVol-BR is the Brazilian implied-volatility index constructed from IBOVESPA
options by the Centro de Estudos em Finanças at FGV EESP (Astorino, Chague,
Giovannetti & Silva). It is the genuine BRL-denominated VIX analogue — same
underlying as the equity series, same currency, same exchange, same trading
calendar.

## Why IVol-BR and not `^VXEWZ`

`^VXEWZ` (CBOE's Brazil ETF Volatility Index) is tempting because yfinance can
fetch it with no manual step. It was rejected on two grounds:

1. **It prices the wrong thing.** VXEWZ is implied volatility of `EWZ`, a
   USD-denominated, US-listed Brazil ETF. It therefore bundles Ibovespa
   volatility with BRL/USD exchange-rate volatility. Feeding it as "Brazilian
   implied vol" would quietly change the research question.
2. **It follows the wrong calendar.** VXEWZ trades on the US calendar. B3
   closes for Carnival and Corpus Christi; the NYSE closes for Thanksgiving
   and Independence Day. Merging a US-calendar IV series onto a B3-calendar
   equity series forces forward-filling on every US holiday that is a Brazilian
   trading day. IVol-BR follows B3, so the merge is clean.

If you later want the USD-investor question answered, the coherent pairing is
`EWZ` + `^VXEWZ` — same underlying on both sides — as a separate market entry,
not as a substitute here.

## BLOCKER: redistribution terms

**Check FGV's terms before committing any file to this directory.**

IVol-BR is distributed as a file download, not through an API, so it cannot be
pulled by a script the way `^VIX` can. That makes committing a snapshot the
obvious move — but only if FGV's terms permit redistribution. Do not assume
they do.

If redistribution is **permitted**: place the download at
`data/br_iv/ivolbr_raw.csv` (the path `code/config.py` expects) and record the
retrieval date, source URL and licence below.

If redistribution is **not permitted**: commit only the derived feature
columns (`iv_lag1`, `iv_lag5`, `iv_change`) plus fetch instructions, and record
that decision here. Do not commit the raw level series.

`code/01_collect_data.py` accepts any CSV with a recognisable date column and
a numeric value column, so no reformatting is needed.

## To populate

```bash
python code/00_probe_sources.py --ivolbr /path/to/download.csv   # needs network
cp /path/to/download.csv data/br_iv/ivolbr_raw.csv
python code/01_collect_data.py --market br_iv
```

## Step 0 — record findings here

- [ ] IVol-BR first date and last date (expected start ~August 2011)
- [ ] Source URL, retrieval date, licence / terms of use
- [ ] Redistribution decision: raw file committed, or derived columns only
- [ ] Calendar agreement with `^BVSP` — count of `^BVSP` days with no IVol-BR
      observation, which is how often the feature goes stale via forward-fill

### Then set the sample start

`code/config.py` currently has `br_iv.start = "2012-01-01"`, which is
**provisional**. Set it from IVol-BR's actual first date, rounded up to the
next January so the arm starts on a clean year boundary.

Note the knock-on effect: the walk-forward split needs
`val_window + 2 * embargo` rows of history before the first test batch, and
`code/03_run_core_models.py` refuses to run rather than silently shrinking the
training set. With `val_window = 252` that is roughly 262 trading days before
2022-01-01, which a 2012 start comfortably clears.

## Scale check

`code/01_collect_data.py` divides the implied-vol level by 100 to reach
decimals for `iv_lag1` and `iv_lag5`, assuming a VIX-style percentage-point
quote. It validates this: if the series median falls outside 1–200 it refuses
to build the panel rather than producing silently wrong features. If IVol-BR
turns out to be quoted as a decimal, adjust `IV_SCALE` in that file.

## Forward-fill limit

The merge is capped at 5% forward-filled rows. Above that,
`01_collect_data.py` fails rather than building a panel whose implied-vol
features are mostly stale. Since both series follow B3, exceeding this would
indicate a real problem with the download — investigate rather than raising
the limit.
