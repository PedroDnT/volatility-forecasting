"""
Step 5b: Model Confidence Set, Superior Predictive Ability, and Reality Check.

The original release shipped a results/mcs_spa_results.json stamped
`"computed_with": "vol-eval 0.1.0"`, but no such package exists on PyPI, no
script in code/ produced the file, it was absent from the README's contents
tree, and the audit never touched it. An entire subsection of the paper rested
on numbers with no reproduction path. This script replaces it using the `arch`
package, which is already a dependency and implements all three tests.

  MCS  (Hansen, Lunde & Nason 2011) -- the set of models that cannot be
       rejected as containing the best forecaster at a given confidence level.
  SPA  (Hansen 2005) -- is a nominated benchmark beaten by anything in the
       panel? Reports the lower / consistent / upper p-values.
  RC   (White 2000) -- the unstudentized ancestor of SPA.

All three use the stationary bootstrap of Politis & Romano (1994) on QLIKE
losses. Pairwise Diebold-Mariano over 42 ordered pairs inflates the family-wise
error rate; these are the joint tests that do not.

Usage:
    python code/05b_mcs_spa.py --market us
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

QLIKE_EPS = 1e-8
N_BOOTSTRAP = 2000


def qlike_loss(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    ratio = (actual ** 2) / (predicted ** 2 + QLIKE_EPS)
    return ratio - np.log(ratio + QLIKE_EPS) - 1


def _construct(cls, *args, seed: int, **kwargs):
    """arch has moved between `seed` and `random_state` across versions."""
    try:
        return cls(*args, seed=seed, **kwargs)
    except TypeError:
        try:
            return cls(*args, random_state=np.random.RandomState(seed), **kwargs)
        except TypeError:
            return cls(*args, **kwargs)


def run_mcs(losses: pd.DataFrame, alpha: float, block_size: int,
            seed: int) -> dict:
    from arch.bootstrap import MCS

    mcs = _construct(MCS, losses, size=alpha, reps=N_BOOTSTRAP,
                     block_size=block_size, method="max",
                     bootstrap="stationary", seed=seed)
    mcs.compute()

    included = sorted(str(m) for m in mcs.included)
    excluded = sorted(str(m) for m in mcs.excluded)
    pvalues = {str(k): float(v) for k, v in
               mcs.pvalues.iloc[:, 0].to_dict().items()}
    return {
        "alpha": alpha,
        "confidence": f"{(1 - alpha) * 100:.0f}%",
        "survivors": included,
        "eliminated": excluded,
        "mcs_p_values": pvalues,
        "n_bootstrap": N_BOOTSTRAP,
        "block_size": block_size,
        "statistic": "t_max",
    }


def run_spa(losses: pd.DataFrame, benchmark: str, block_size: int, seed: int,
            studentize: bool) -> dict:
    from arch.bootstrap import SPA

    others = [c for c in losses.columns if c != benchmark]
    spa = _construct(SPA, losses[benchmark], losses[others], reps=N_BOOTSTRAP,
                     block_size=block_size, bootstrap="stationary",
                     studentize=studentize, seed=seed)
    spa.compute()
    p = {str(k): float(v) for k, v in spa.pvalues.to_dict().items()}
    return {
        "benchmark": benchmark,
        "alternatives": others,
        "p_values": p,
        "studentized": studentize,
        "n_bootstrap": N_BOOTSTRAP,
        "block_size": block_size,
    }


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

    losses = pd.DataFrame(
        {m: qlike_loss(actual, fc[m].values) for m in models}, index=fc.index
    ).dropna()

    # Stationary-bootstrap mean block length. Overlapping h-day targets induce
    # serial dependence of roughly h; 2h is a conventional, conservative choice.
    block_size = cfg.horizon * 2

    print(f"market={cfg.name}  N={len(losses)}  models={len(models)}  "
          f"block_size={block_size}  reps={N_BOOTSTRAP}")

    payload: dict = {
        "market": cfg.name,
        "loss_function": "QLIKE",
        "forecast_horizon_days": cfg.horizon,
        "n_observations": int(len(losses)),
        "bootstrap": "stationary (Politis & Romano 1994)",
        "seed": cfg.seed,
    }

    for alpha, key in [(0.10, "mcs_90"), (0.05, "mcs_95")]:
        print(f"\nMCS at {(1 - alpha) * 100:.0f}% ...")
        payload[key] = run_mcs(losses, alpha, block_size, cfg.seed)
        print(f"  survivors ({len(payload[key]['survivors'])}): "
              f"{', '.join(payload[key]['survivors'])}")
        if payload[key]["eliminated"]:
            print(f"  eliminated: {', '.join(payload[key]['eliminated'])}")

    mean_loss = losses.mean().sort_values()
    benchmarks = ["HAR-RV", "GARCH", mean_loss.index[0]]
    seen: set[str] = set()
    for bench in benchmarks:
        if bench in seen or bench not in losses.columns:
            continue
        seen.add(bench)
        print(f"\nSPA, benchmark = {bench} ...")
        payload[f"spa_{bench.replace('-', '_').lower()}"] = run_spa(
            losses, bench, block_size, cfg.seed, studentize=True)
        payload[f"reality_check_{bench.replace('-', '_').lower()}"] = run_spa(
            losses, bench, block_size, cfg.seed, studentize=False)
        p = payload[f"spa_{bench.replace('-', '_').lower()}"]["p_values"]
        print(f"  SPA p-values: " + ", ".join(f"{k}={v:.4f}" for k, v in p.items()))

    with open(out_dir / "mcs_spa_results.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {out_dir / 'mcs_spa_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
