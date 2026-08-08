"""
Step 1: download raw inputs and build the engineered feature panel.

Market-aware: `--market us` reproduces the original S&P 500 panel, while
`--market br_long` / `--market br_iv` build the Brazilian ones. See
code/config.py for what differs.

Three things changed relative to the original single-market version:

  * Downloads retry with backoff, and `--use-cache` reuses whatever is already
    on disk. The original made one unguarded call and, when Yahoo refused,
    failed with `IndexError: index 0 is out of bounds` raised from a print
    statement -- which is how it fails today behind a rate limit.

  * A validation gate reports volume quality, implied-vol forward-fill counts,
    calendar mismatch and rows lost to dropna, and refuses to write a panel
    that is quietly degenerate. This matters much more for Brazil than for the
    US: Yahoo's ^BVSP volume is unreliable, and two model features depend on it.

  * dropna is scoped to the columns the study actually uses. The original
    dropped any row with a NaN anywhere, including in rv_22d_fwd, a target the
    5-day study never touches -- discarding ~22 usable rows from the end of
    every sample.

Implied-volatility columns are named `iv_*`. The original called them `vix_*`,
which would be actively misleading in Brazilian output files.

Usage:
    python code/01_collect_data.py --market us
    python code/01_collect_data.py --market br_long --use-cache
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

# The implied-vol series is expected in percentage points (VIX-style: ~19.0
# means 19%). iv_lag1 / iv_lag5 are divided by this to reach decimals.
IV_SCALE = 100.0

MAX_VOLUME_BAD_FRACTION = 0.01  # above this, volume features are switched off
MAX_IV_FFILL_FRACTION = 0.05  # above this, refuse to build the panel


class DataQualityError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [str(c).lower() for c in df.columns]
    df.index.name = "date"
    return df


def download_yahoo(ticker: str, start: str, end: str, retries: int = 4) -> pd.DataFrame:
    """Download with exponential backoff. Raises rather than returning empty."""
    import yfinance as yf

    delay = 2.0
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                             progress=False)
            if df is not None and len(df):
                df = _normalise(df)
                print(f"  {ticker}: {len(df):,} rows, "
                      f"{df.index[0].date()} to {df.index[-1].date()}")
                return df
            last_error = RuntimeError("empty response")
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc

        if attempt < retries:
            print(f"  {ticker}: attempt {attempt}/{retries} failed "
                  f"({last_error}); retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2

    raise DataQualityError(
        f"could not download {ticker} after {retries} attempts: {last_error}. "
        f"If you are rate limited, re-run later or use --use-cache."
    )


def load_cached(path: Path, label: str) -> pd.DataFrame | None:
    if path.exists():
        df = pd.read_parquet(path)
        print(f"  {label}: {len(df):,} cached rows from {path.name}")
        return df
    return None


def load_iv_csv(path: Path) -> pd.DataFrame:
    """Load a manually-downloaded implied-vol series (IVol-BR).

    Accepts any CSV with a recognisable date column and a numeric value
    column; the value column is whichever numeric column has the most
    non-null entries.
    """
    if not path.exists():
        raise DataQualityError(
            f"implied-vol file not found: {path}\n"
            f"IVol-BR is distributed as a download rather than an API. Fetch "
            f"it, place it at that path, and see data/br_iv/PROVENANCE.md."
        )

    raw = pd.read_csv(path)
    date_col = next(
        (c for c in raw.columns
         if str(c).strip().lower() in {"date", "data", "dt", "dia", "referencia"}),
        raw.columns[0],
    )
    idx = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)

    numeric = raw.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")
    if numeric.empty or not len(numeric.columns):
        raise DataQualityError(f"no numeric column found in {path}")
    value_col = numeric.notna().sum().idxmax()

    out = pd.DataFrame({"close": numeric[value_col].values}, index=idx)
    out = out[out.index.notna() & out["close"].notna()].sort_index()
    out.index.name = "date"
    print(f"  IV csv: {len(out):,} rows from column {value_col!r}, "
          f"{out.index[0].date()} to {out.index[-1].date()}")
    return out


# ----------------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------------

def compute_realized_volatility(px: pd.DataFrame, cfg: config.MarketConfig) -> pd.DataFrame:
    """Realized-volatility measures from daily closes.

    Note this is a rolling realized variance of *daily* returns, not an
    intraday realized variance. B3 intraday data is not freely available, so
    the Brazilian arms inherit the same limitation as the original study.
    """
    ann = np.sqrt(cfg.annualization)
    df = pd.DataFrame(index=px.index)

    df["log_return"] = np.log(px["close"] / px["close"].shift(1))
    df["abs_return"] = df["log_return"].abs()
    df["squared_return"] = df["log_return"] ** 2

    # Parkinson high-low range estimator.
    df["range_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(px["high"] / px["low"])) ** 2
    )

    for window in cfg.rolling_windows:
        df[f"rv_{window}d"] = (
            df["squared_return"].rolling(window).mean().apply(np.sqrt) * ann
        )

    # Forward realized volatility -- the forecast targets. shift(-h) then
    # rolling(h) averages squared returns over t+1..t+h.
    for horizon in cfg.target_horizons:
        df[f"rv_{horizon}d_fwd"] = (
            df["squared_return"].shift(-horizon).rolling(horizon).mean()
            .apply(np.sqrt) * ann
        )

    return df


def build_features(px: pd.DataFrame, iv: pd.DataFrame | None,
                   cfg: config.MarketConfig, report: dict) -> pd.DataFrame:
    """Build the feature panel. Every feature is backward-looking."""
    rv = compute_realized_volatility(px, cfg)

    if cfg.use_iv:
        if iv is None:
            raise DataQualityError(
                f"market {cfg.name!r} sets use_iv=True but no implied-vol "
                f"series was loaded"
            )
        raw_iv = iv["close"].reindex(rv.index)
        n_missing = int(raw_iv.isna().sum())
        rv["iv_close"] = raw_iv.ffill()

        # Rows where the equity index traded but the IV index did not. On
        # those days the IV feature is stale (carried forward), never
        # forward-looking -- but a large count means Arm B leans on ffill.
        report["iv_ffill_days"] = n_missing
        report["iv_ffill_fraction"] = n_missing / max(len(rv), 1)

        rv["iv_lag1"] = rv["iv_close"].shift(1) / IV_SCALE
        rv["iv_lag5"] = rv["iv_close"].rolling(5).mean().shift(1) / IV_SCALE
        rv["iv_change"] = rv["iv_close"].pct_change().shift(1)

    for lag in cfg.feature_lags:
        rv[f"rv_5d_lag{lag}"] = rv["rv_5d"].shift(lag)
        rv[f"abs_ret_lag{lag}"] = rv["abs_return"].shift(lag)

    # HAR components (Corsi 2009): daily, weekly, monthly.
    rv["har_daily"] = rv["rv_5d"].shift(1)
    rv["har_weekly"] = rv["rv_5d"].rolling(5).mean().shift(1)
    rv["har_monthly"] = rv["rv_5d"].rolling(22).mean().shift(1)

    # Leverage effect.
    rv["neg_return_5d"] = rv["log_return"].clip(upper=0).rolling(5).sum().shift(1)
    rv["pos_return_5d"] = rv["log_return"].clip(lower=0).rolling(5).sum().shift(1)

    if report["use_volume"]:
        rv["log_volume"] = np.log(px["volume"] + 1)
        rv["volume_change"] = rv["log_volume"].pct_change().shift(1)
        rv["volume_ma_ratio"] = (
            rv["log_volume"] / rv["log_volume"].rolling(22).mean()
        ).shift(1)

    dow = rv.index.dayofweek
    for d in range(5):
        rv[f"dow_{d}"] = (dow == d).astype(int)

    rv["month"] = rv.index.month
    return rv


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def gate_volume(px: pd.DataFrame, cfg: config.MarketConfig, report: dict) -> bool:
    """Decide whether volume features are trustworthy for this market.

    Yahoo's ^BVSP volume is frequently zero or missing. The original code fed
    such a series straight through np.nan_to_num(..., posinf=0), which turns a
    degenerate volume_ma_ratio into a column of zeros -- a silently useless
    feature rather than a loud failure.
    """
    if "volume" not in px.columns:
        report["volume_verdict"] = "absent"
        return False

    v = px["volume"]
    n_bad = int((v == 0).sum() + v.isna().sum())
    frac = n_bad / max(len(v), 1)
    report["volume_bad_rows"] = n_bad
    report["volume_bad_fraction"] = frac

    if cfg.use_volume is not None:
        report["volume_verdict"] = f"forced by config ({cfg.use_volume})"
        return cfg.use_volume

    ok = frac <= MAX_VOLUME_BAD_FRACTION
    report["volume_verdict"] = (
        f"auto: {frac:.2%} bad rows -> {'kept' if ok else 'DISABLED'}"
    )
    return ok


def gate_iv_scale(panel: pd.DataFrame, report: dict) -> None:
    """The /100 rescaling assumes a VIX-style percentage-point quote."""
    if "iv_close" not in panel.columns:
        return
    med = float(panel["iv_close"].median())
    report["iv_median"] = med
    if not (1.0 < med < 200.0):
        raise DataQualityError(
            f"implied-vol series has median {med:.4f}, which does not look "
            f"like percentage points. iv_lag1/iv_lag5 divide by {IV_SCALE:.0f} "
            f"and would be wrong. Rescale the source or adjust IV_SCALE."
        )


def gate_iv_ffill(report: dict) -> None:
    frac = report.get("iv_ffill_fraction", 0.0)
    if frac > MAX_IV_FFILL_FRACTION:
        raise DataQualityError(
            f"{frac:.2%} of rows required forward-filling the implied-vol "
            f"series (limit {MAX_IV_FFILL_FRACTION:.0%}). The equity and "
            f"implied-vol calendars disagree badly; check the source before "
            f"trusting this panel."
        )


def print_report(cfg: config.MarketConfig, report: dict) -> None:
    print("\n" + "-" * 68)
    print(f"DATA QUALITY REPORT -- market={cfg.name}")
    print("-" * 68)
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key:<26s} {value:.6g}")
        else:
            print(f"  {key:<26s} {value}")
    print("-" * 68)


# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_market_arg(ap)
    ap.add_argument("--use-cache", action="store_true",
                    help="reuse raw parquet files already on disk")
    ap.add_argument("--legacy-quirks", action="store_true",
                    help="reproduce the original published panel exactly "
                         "(drops any row with a NaN in any column). Used only "
                         "by the regression test.")
    args = ap.parse_args()

    cfg = config.get(args.market)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"Collecting data: {cfg.equity_label} ({cfg.equity_ticker})"
          f"  market={cfg.name}")
    print(f"Window {cfg.start} .. {cfg.end}   annualization={cfg.annualization}")
    print("=" * 68)

    equity_path = cfg.data_dir / "equity_daily.parquet"
    iv_path = cfg.data_dir / "iv_daily.parquet"

    px = load_cached(equity_path, cfg.equity_ticker) if args.use_cache else None
    if px is None:
        px = download_yahoo(cfg.equity_ticker, cfg.start, cfg.end)
        px.to_parquet(equity_path)

    iv = None
    if cfg.use_iv:
        if args.use_cache:
            iv = load_cached(iv_path, str(cfg.iv_label))
        if iv is None:
            if cfg.iv_kind == "yfinance":
                iv = download_yahoo(str(cfg.iv_ref), cfg.start, cfg.end)
            elif cfg.iv_kind == "csv":
                iv = load_iv_csv(cfg.data_dir / str(cfg.iv_ref))
            else:
                raise DataQualityError(f"unknown iv_spec {cfg.iv_spec!r}")
            iv.to_parquet(iv_path)

    report: dict = {
        "equity_rows": len(px),
        "equity_start": str(px.index[0].date()),
        "equity_end": str(px.index[-1].date()),
    }
    report["use_volume"] = gate_volume(px, cfg, report)

    panel = build_features(px, iv, cfg, report)
    gate_iv_scale(panel, report)
    gate_iv_ffill(report)

    before = len(panel)
    if args.legacy_quirks:
        panel = panel.dropna()
        report["dropna_scope"] = "all columns (legacy)"
    else:
        needed = cfg.feature_columns(panel, use_volume=report["use_volume"])
        needed = needed + [cfg.target]
        panel = panel.dropna(subset=needed)
        report["dropna_scope"] = f"{len(needed)} used columns"
    report["rows_dropped_nan"] = before - len(panel)
    report["panel_rows"] = len(panel)
    report["panel_start"] = str(panel.index[0].date())
    report["panel_end"] = str(panel.index[-1].date())

    fcols = cfg.feature_columns(panel, use_volume=report["use_volume"])
    report["n_features"] = len(fcols)

    test_rows = int((panel.index >= cfg.test_start).sum())
    report["test_rows"] = test_rows
    if test_rows < 100:
        raise DataQualityError(
            f"only {test_rows} rows in the test window (>= {cfg.test_start}); "
            f"refusing to write a panel that cannot support evaluation."
        )

    panel.to_parquet(cfg.data_dir / "combined.parquet")

    stats_cols = [c for c in ["log_return", "rv_5d", "rv_22d", "iv_close"]
                  if c in panel.columns]
    panel[stats_cols].describe().to_csv(cfg.data_dir / "summary_stats.csv")

    print_report(cfg, report)
    print(f"\nFeatures ({len(fcols)}): {', '.join(fcols)}")
    print(f"\nWrote {cfg.data_dir / 'combined.parquet'}")
    print(f"Wrote {cfg.data_dir / 'summary_stats.csv'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DataQualityError as exc:
        print(f"\nDATA QUALITY FAILURE: {exc}", file=sys.stderr)
        sys.exit(2)
