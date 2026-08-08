# Arm A — Ibovespa, long sample, no implied volatility

**Status: NOT YET POPULATED.** Blocked only on Yahoo access for `^BVSP`.

## What this arm is

`^BVSP` (Ibovespa) daily OHLCV from Yahoo Finance, January 2004 to April 2022,
with the same feature construction as the US study minus the four
implied-volatility features.

It shares `br_iv`'s evaluation window (2018-01-01 → 2022-04-29) exactly, so the
two differ only in training start and the presence of implied vol — which is
what makes the pair a clean measurement of what implied vol is worth. The end
date is inherited from IVol-BR; see `data/br_iv/PROVENANCE.md`.

The trade is deliberate. This arm buys extra training history at the cost of
the implied-vol block:

| Episode | Why it matters |
|---|---|
| 2008 GFC | The only pre-sample crisis; lost entirely if the sample starts in 2011 |
| 2011 Euro crisis | |
| 2013 taper tantrum | Sharp EM-specific drawdown |
| 2014–16 recession + Lava Jato | A prolonged domestic stress regime with no US analogue |
| May 2017 "Joesley Day" | One-day −8.8%, two circuit breakers |
| 2018 truckers' strike, election | Sits inside the test window |
| March 2020 COVID | Six B3 circuit breakers; sits inside the test window |
| 2021–22 fiscal stress, election | Sits inside the test window |

`br_iv` makes the opposite trade: implied-vol features, but training only from
August 2011.

## To populate

```bash
python code/00_probe_sources.py                 # needs network
python code/01_collect_data.py --market br_long
```

Blocked only on Yahoo access — nothing else is outstanding for this arm.

## Open question — record findings here

Run `code/00_probe_sources.py` on a machine with network access and fill in:

- [ ] `^BVSP` first date, last date, row count for 2004-01-01 .. 2022-04-29
- [ ] `^BVSP` volume zero/NaN rate, and the years where it is worst
- [ ] Median trading days per calendar year

### Open question: is `^BVSP` volume usable?

Two model features depend on it — `volume_change` and `volume_ma_ratio`.
(`log_volume` is constructed but never reaches the models; it does not match
any feature prefix.) Yahoo's index volume for `^BVSP` is unreliable: frequent
zeros, and it reports financial volume rather than share counts, with regime
changes over the sample.

`code/01_collect_data.py` gates this automatically — above a 1% bad-row rate it
disables the volume features and says so. If step 0 confirms the series is
bad, pin `use_volume=False` for both `br_*` markets in `code/config.py` rather
than leaving it to the auto-gate, so the decision is visible in the config
rather than in a log line. Feature count then drops from 26 to 24 for this arm.

If volume turns out to matter, the alternative source is `BOVA11.SA`, the
iShares Ibovespa ETF, which has genuine share volume — but it only starts in
November 2008 and would forfeit most of this arm's extra history.

## Notes on the Brazilian setting

**Annualization stays at 252.** This looks like a US convention that a
Brazilian replication ought to change, and it is not: 252 business days
(*dias úteis*) is the standard Brazilian convention, used throughout the BRL
rates market. B3 trades roughly 246–250 days a year once Carnival and Corpus
Christi are removed, much as the NYSE falls short of 252.

**Circuit breakers truncate realized volatility.** B3 halts trading at −10%
and −15%, and triggered six times in March 2020. Close-to-close returns still
exist on halt days, so RV remains computable, but it is *understated* —
precisely in the high-volatility regime where the study's central claim lives.
This is a genuine limitation of the target variable, not of the models, and it
has no US counterpart of comparable frequency.

**Daily, not intraday.** As in the original study, "realized volatility" here
is a rolling realized variance of daily returns, not an intraday estimator.
B3 intraday data is not freely available, so this arm inherits the limitation
rather than choosing it.
