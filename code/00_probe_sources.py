"""
Step 0: data-availability probe for the Brazilian arms.

Run this FIRST, on a machine with working network access. Every date in
code/config.py for the br_* markets is provisional until this has been run and
data/br_*/PROVENANCE.md has been filled in from its output.

It answers four questions that the rest of the plan depends on:

  1. Does Yahoo actually serve ^BVSP back to 2004, and how many rows?
  2. Is ^BVSP's `volume` column usable? Yahoo's index volume is unreliable --
     frequent zeros, and it reports financial volume rather than share counts
     with regime changes over time. Two model features depend on it.
  3. When does IVol-BR actually start, and in what format?
  4. Do the Ibovespa and IVol-BR trading calendars agree? They should, since
     both follow B3. (This is why IVol-BR was chosen over ^VXEWZ, which
     follows the US CBOE calendar and would mismatch on Carnival, Corpus
     Christi, Thanksgiving and July 4th.)

Usage:
    python code/00_probe_sources.py
    python code/00_probe_sources.py --ivolbr path/to/downloaded_ivolbr.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

PROBE_TICKERS = ["^BVSP", "^GSPC", "^VXEWZ", "EWZ", "BOVA11.SA", "BRL=X"]

IVOLBR_NOTE = """
IVol-BR is the Brazilian implied-volatility index built from IBOVESPA options,
published by NEFIN (Nucleo de Economia Financeira) at FEA-USP. It is the
genuine BRL-denominated VIX analogue.

It is a plain CSV at a stable URL, so 01_collect_data.py fetches it directly --
no manual step, and a committed snapshot at data/br_iv/ivolbr_raw.csv serves as
the offline fallback. Coverage and terms are recorded in
data/br_iv/PROVENANCE.md.

Pass --ivolbr to inspect a local copy instead of the committed snapshot.
"""


def probe_yahoo(start: str, end: str) -> None:
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed; skipping Yahoo probe")
        return

    print("=" * 72)
    print("YAHOO FINANCE COVERAGE")
    print("=" * 72)

    for ticker in PROBE_TICKERS:
        try:
            df = yf.download(
                ticker, start=start, end=end, auto_adjust=True, progress=False
            )
        except Exception as exc:  # noqa: BLE001 - probe reports, never raises
            print(f"\n{ticker:<12s} FAILED: {type(exc).__name__}: {exc}")
            continue

        if df is None or len(df) == 0:
            print(f"\n{ticker:<12s} EMPTY RESPONSE (rate limited, or no data)")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.columns = [c.lower() for c in df.columns]

        print(f"\n{ticker:<12s} {len(df):>6,} rows   "
              f"{df.index[0].date()} -> {df.index[-1].date()}")

        if "volume" in df.columns:
            v = df["volume"]
            n_zero = int((v == 0).sum())
            n_nan = int(v.isna().sum())
            print(f"{'':<12s} volume: {n_zero:,} zeros ({n_zero / len(v):.1%}), "
                  f"{n_nan:,} NaN ({n_nan / len(v):.1%})")
            if n_zero or n_nan:
                bad = v[(v == 0) | v.isna()]
                print(f"{'':<12s}   bad-volume span: "
                      f"{bad.index[0].date()} -> {bad.index[-1].date()}")
                by_year = bad.groupby(bad.index.year).size()
                worst = by_year.sort_values(ascending=False).head(5)
                print(f"{'':<12s}   worst years: "
                      + ", ".join(f"{y}={n}" for y, n in worst.items()))
            verdict = "USABLE" if (n_zero + n_nan) / len(v) < 0.01 else "SUSPECT"
            print(f"{'':<12s}   -> volume features: {verdict}")

        # Days per calendar year: sanity-check the 252 annualisation.
        per_year = df.groupby(df.index.year).size()
        full = per_year[(per_year.index > df.index[0].year)
                        & (per_year.index < df.index[-1].year)]
        if len(full):
            print(f"{'':<12s} trading days/yr: median={full.median():.0f} "
                  f"min={full.min()} max={full.max()}")


def probe_ivolbr(path: Path | None) -> pd.DataFrame | None:
    print()
    print("=" * 72)
    print("IVol-BR")
    print("=" * 72)

    if path is None:
        default = Path(__file__).resolve().parent.parent / "data" / "br_iv" / "ivolbr_raw.csv"
        if default.exists():
            path = default
            print(f"using the committed snapshot: {path}")
        else:
            print(IVOLBR_NOTE)
            print("No --ivolbr path and no committed snapshot; nothing to inspect.")
            return None

    if not path.exists():
        print(f"file not found: {path}")
        return None

    raw = pd.read_csv(path)
    print(f"file: {path}  ({path.stat().st_size:,} bytes)")
    print(f"shape: {raw.shape}")
    print(f"columns: {list(raw.columns)}")
    print("\nfirst 3 rows:")
    print(raw.head(3).to_string())
    print("\nlast 3 rows:")
    print(raw.tail(3).to_string())
    return raw


def probe_calendar_overlap(start: str, end: str, ivolbr: pd.DataFrame | None) -> None:
    if ivolbr is None:
        return
    try:
        import yfinance as yf
    except ImportError:
        return

    print()
    print("=" * 72)
    print("CALENDAR AGREEMENT: ^BVSP vs IVol-BR")
    print("=" * 72)

    bvsp = yf.download("^BVSP", start=start, end=end, auto_adjust=True,
                       progress=False)
    if bvsp is None or len(bvsp) == 0:
        print("could not fetch ^BVSP; skipping")
        return

    # Reuse the production parser rather than re-deriving the date column --
    # IVol-BR ships year/month/day columns, so naive detection picks "year".
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "collector", Path(__file__).resolve().parent / "01_collect_data.py")
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    iv_idx = pd.DatetimeIndex(collector._parse_iv_table(ivolbr, "IVol-BR").index)
    bv_idx = pd.DatetimeIndex(bvsp.index)

    common_start = max(iv_idx.min(), bv_idx.min())
    iv_w = iv_idx[iv_idx >= common_start]
    bv_w = bv_idx[bv_idx >= common_start]

    only_bvsp = bv_w.difference(iv_w)
    only_iv = iv_w.difference(bv_w)

    print(f"overlap window starts {common_start.date()}")
    print(f"  ^BVSP days:   {len(bv_w):,}")
    print(f"  IVol-BR days: {len(iv_w):,}")
    print(f"  ^BVSP only (IV would be forward-filled): {len(only_bvsp):,} "
          f"({len(only_bvsp) / max(len(bv_w), 1):.2%})")
    print(f"  IVol-BR only (dropped on merge):         {len(only_iv):,}")
    if len(only_bvsp):
        print(f"    e.g. {[d.date().isoformat() for d in only_bvsp[:8]]}")
    if len(only_iv):
        print(f"    e.g. {[d.date().isoformat() for d in only_iv[:8]]}")
    print("\nA large ^BVSP-only count means the merge leans on ffill and the "
          "implied-vol features go stale; investigate before trusting Arm B.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ivolbr", type=Path, default=None,
                    help="path to a manually downloaded IVol-BR CSV")
    ap.add_argument("--start", default="2004-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()

    probe_yahoo(args.start, args.end)
    iv = probe_ivolbr(args.ivolbr)
    probe_calendar_overlap(args.start, args.end, iv)

    print()
    print("=" * 72)
    print("NEXT STEPS")
    print("=" * 72)
    print("""
1. Record the ^BVSP row count and date range in data/br_long/PROVENANCE.md.
2. If ^BVSP volume came back SUSPECT, pin use_volume=False for both br_*
   markets in code/config.py (feature counts become 24 for Arm A, 28 for
   Arm B) rather than leaving it to the auto-gate.
3. Set br_iv's `start` in code/config.py from IVol-BR's actual first date,
   rounded up to the next January.
4. Resolve the IVol-BR redistribution question before committing any raw file.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
