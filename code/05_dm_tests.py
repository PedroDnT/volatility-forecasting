"""
Step 5: Diebold-Mariano (1995) tests on QLIKE loss differentials.

For each ordered model pair (A, B), computes
    d_t = L_QLIKE(actual_t, A_t) - L_QLIKE(actual_t, B_t)
and tests H_0: E[d] = 0 with a Newey-West HAC standard error at bandwidth
h - 1, the rule of thumb for an h-day forecast horizon.

A negative t-statistic means A's QLIKE is lower (A is better). Two-sided.

Usage:
    python code/05_dm_tests.py --market us
"""

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

QLIKE_EPS = 1e-8


def qlike_loss(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    ratio = (actual ** 2) / (predicted ** 2 + QLIKE_EPS)
    return ratio - np.log(ratio + QLIKE_EPS) - 1


def newey_west_se(d: np.ndarray, q: int) -> float:
    """HAC long-run standard error of the mean, Bartlett kernel, bandwidth q."""
    d = d - d.mean()
    n = len(d)
    s = (d @ d) / n
    for k in range(1, q + 1):
        s += 2 * (1 - k / (q + 1)) * ((d[k:] @ d[:-k]) / n)
    return float(np.sqrt(s / n)) if s > 0 else float("nan")


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, q: int) -> dict:
    d = loss_a - loss_b
    d = d[~np.isnan(d)]
    if len(d) < 10:
        return {"n": int(len(d)), "mean_diff": float("nan"),
                "t_stat": float("nan"), "p_value": float("nan"),
                "se": float("nan")}
    se = newey_west_se(d, q)
    t = d.mean() / se if se and not np.isnan(se) and se > 0 else float("nan")
    p = 2 * (1 - stats.norm.cdf(abs(t))) if not np.isnan(t) else float("nan")
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "t_stat": float(t),
            "p_value": float(p), "se": float(se)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_market_arg(ap)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = config.get(args.market)
    out_dir = args.out or cfg.results_dir

    fc = pd.read_parquet(out_dir / "forecasts_5d.parquet")
    actual = fc["actual"].values
    models = [c for c in fc.columns if c != "actual"]
    losses = {m: qlike_loss(actual, fc[m].values) for m in models}

    q = cfg.horizon - 1
    results = {f"{a} vs {b}": dm_test(losses[a], losses[b], q)
               for a, b in permutations(models, 2)}

    payload = {
        "metadata": {
            "market": cfg.name,
            "loss_function": "QLIKE",
            "forecast_horizon_days": cfg.horizon,
            "hac_bandwidth": q,
            "test": "Diebold-Mariano (1995) two-sided",
            "interpretation": "negative t_stat means A beats B; p < 0.05 significant",
        },
        "pairs": results,
    }
    with open(out_dir / "dm_tests.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {out_dir / 'dm_tests.json'}")

    ranked = sorted(models, key=lambda m: np.nanmean(losses[m]))
    best = ranked[0]
    print(f"\nBest by mean QLIKE: {best}. Pairwise vs the field:")
    print(f"  {'Pair':<34s} {'mean diff':>12s} {'t-stat':>8s} {'p-val':>8s}")
    for other in ranked[1:]:
        r = results[f"{best} vs {other}"]
        sig = "sig" if r["p_value"] < 0.05 else "ns"
        print(f"  {best + ' vs ' + other:<34s} {r['mean_diff']:>+12.5f} "
              f"{r['t_stat']:>+8.3f} {r['p_value']:>8.4f}  ({sig})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
