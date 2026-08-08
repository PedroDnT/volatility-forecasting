"""
Step 4: subperiod metrics, regime splits, and LightGBM split importances.

Market-aware. Reads results/<market>/forecasts_5d.parquet and writes
metrics_subperiod.json, feature_importance.json and forecasts_5d.csv into the
same directory.

The feature list comes from config.MarketConfig.feature_columns() rather than
a local copy of the prefix filter. In the original both this script and the
model runner carried their own copy, which is a standing invitation for the
importances to be computed over a different feature set than the forecasts.

Usage:
    python code/04_subperiod_and_importance.py --market us
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


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    from scipy.stats import pearsonr

    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    keys = ["MSE", "MAE", "RMSE", "QLIKE", "R2", "MZ_R2", "N"]
    if len(a) == 0:
        return {k: float("nan") for k in keys}

    mse = np.mean((a - p) ** 2)
    ratio = (a ** 2) / (p ** 2 + QLIKE_EPS)
    qlike = np.mean(ratio - np.log(ratio + QLIKE_EPS) - 1)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    corr, _ = pearsonr(a, p)
    return {
        "MSE": float(mse), "MAE": float(np.mean(np.abs(a - p))),
        "RMSE": float(np.sqrt(mse)), "QLIKE": float(qlike),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "MZ_R2": float(corr ** 2), "N": int(len(a)),
    }


def subperiod_metrics(fc: pd.DataFrame, cfg: config.MarketConfig) -> dict:
    actual = fc["actual"].values
    models = [c for c in fc.columns if c != "actual"]
    out: dict = {"by_year": {}, "by_regime": {}}

    is_split = np.asarray(fc.index.year == cfg.split_year)
    is_after = np.asarray(fc.index.year > cfg.split_year)

    out["by_year"][f"{cfg.split_year}_high_vol"] = {
        "N": int(is_split.sum()),
        "models": {m: metrics(actual[is_split], fc[m].values[is_split])
                   for m in models},
    }
    out["by_year"][f"after_{cfg.split_year}_lower_vol"] = {
        "N": int(is_after.sum()),
        "models": {m: metrics(actual[is_after], fc[m].values[is_after])
                   for m in models},
    }

    threshold = float(np.nanpercentile(actual, cfg.regime_quantile))
    is_high = actual >= threshold
    out["by_regime"]["meta"] = {
        "quantile": cfg.regime_quantile, "threshold": threshold,
    }
    out["by_regime"]["high_vol"] = {
        "N": int(is_high.sum()),
        "models": {m: metrics(actual[is_high], fc[m].values[is_high])
                   for m in models},
    }
    out["by_regime"]["lower_vol"] = {
        "N": int((~is_high).sum()),
        "models": {m: metrics(actual[~is_high], fc[m].values[~is_high])
                   for m in models},
    }
    return out


def compute_feature_importance(cfg: config.MarketConfig, panel_path: Path) -> dict:
    import lightgbm as lgb

    df = pd.read_parquet(panel_path)
    fcols = cfg.feature_columns(df)

    train_val = df[:cfg.val_end].dropna(subset=[cfg.target])
    X = np.nan_to_num(train_val[fcols].values, nan=0, posinf=0, neginf=0)
    y = np.nan_to_num(train_val[cfg.target].values, nan=0, posinf=0, neginf=0)

    # Chronological 80/20 split with an embargo, so the validation block used
    # for early stopping does not overlap the training block's forward targets.
    n_train = int(len(train_val) * 0.8)
    X_tr, y_tr = X[:n_train - cfg.embargo], y[:n_train - cfg.embargo]
    X_va, y_va = X[n_train:], y[n_train:]

    ds_tr = lgb.Dataset(X_tr, y_tr)
    ds_va = lgb.Dataset(X_va, y_va, reference=ds_tr)
    model = lgb.train(
        {
            "objective": "regression", "metric": "mse", "num_leaves": 31,
            "learning_rate": 0.05, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1,
            "n_jobs": -1, "seed": cfg.seed, "bagging_seed": cfg.seed,
            "feature_fraction_seed": cfg.seed, "data_random_seed": cfg.seed,
        },
        ds_tr, num_boost_round=500, valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    importance = model.feature_importance(importance_type="split")
    ranked = sorted(zip(fcols, importance.tolist()), key=lambda kv: -kv[1])
    total = sum(importance) or 1
    return {
        "market": cfg.name,
        "fit_window": f"{train_val.index[0].date()} to {cfg.val_end}",
        "fit_rows": int(len(train_val)),
        "best_iteration": int(model.best_iteration or model.current_iteration()),
        "importance_type": "split",
        "n_features": len(fcols),
        "ranked": [
            {"rank": i + 1, "feature": f, "importance": int(v),
             "share": float(v) / float(total)}
            for i, (f, v) in enumerate(ranked)
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_market_arg(ap)
    ap.add_argument("--input", type=Path, default=None,
                    help="override the input panel path")
    ap.add_argument("--out", type=Path, default=None,
                    help="override the results directory")
    args = ap.parse_args()

    cfg = config.get(args.market)
    panel_path = args.input or (cfg.data_dir / "combined.parquet")
    out_dir = args.out or cfg.results_dir

    fc = pd.read_parquet(out_dir / "forecasts_5d.parquet")
    print(f"Loaded forecasts: {len(fc):,} rows, {fc.index[0].date()} to "
          f"{fc.index[-1].date()}  market={cfg.name}")

    sub = subperiod_metrics(fc, cfg)
    with open(out_dir / "metrics_subperiod.json", "w") as fh:
        json.dump(sub, fh, indent=2)
    print("Wrote metrics_subperiod.json")

    print("Fitting LightGBM for feature importance...")
    fi = compute_feature_importance(cfg, panel_path)
    with open(out_dir / "feature_importance.json", "w") as fh:
        json.dump(fi, fh, indent=2)
    print(f"Wrote feature_importance.json (best_iteration={fi['best_iteration']})")

    fc.to_csv(out_dir / "forecasts_5d.csv")
    print(f"Wrote forecasts_5d.csv")

    print(f"\nTop-10 features ({fi['n_features']} total):")
    for r in fi["ranked"][:10]:
        print(f"  {r['rank']:2d}. {r['feature']:<22s} "
              f"importance={r['importance']:5d}  share={r['share']:.3f}")

    for label, block in sub["by_year"].items():
        print(f"\n{label} (N={block['N']}) QLIKE ranking:")
        for name, m in sorted(block["models"].items(), key=lambda kv: kv[1]["QLIKE"]):
            print(f"  {name:<12s} QLIKE={m['QLIKE']:.4f}  RMSE={m['RMSE']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
