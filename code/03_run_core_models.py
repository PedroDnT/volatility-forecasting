"""
Step 3: run all seven model configurations and dump forecasts + metrics.

Two methodological defects in the original are corrected here.

FIX 1 -- the validation set was nested inside the training set.
    The original set `train = df[:batch_start]` while `val = df["2019":"2021"]`,
    so every validation row was also a training row. Early stopping therefore
    never fired: all fifteen LightGBM fits logged "Did not meet early stopping",
    and both tree models trained the full 500 rounds on every batch. Here the
    validation block is the `val_window` rows immediately preceding the batch,
    and training is everything before that, with an `embargo` gap between them.

FIX 2 -- GARCH reported in-sample filtered volatility, not a forecast.
    The original fit on returns through the *end* of each batch and then read
    `res.conditional_volatility` back over that same batch, so the parameters
    had seen the data they were scored on. Here parameters are estimated
    strictly before each batch, then held fixed while the variance filter is
    run forward (arch's `.fix()`), which is the conservative walk-forward
    design the paper describes but did not adopt.

    Fixing the look-ahead also forces a horizon repair. `conditional_volatility`
    is a *one-day* conditional volatility, and it was being scored against
    rv_5d_fwd, a *five-day* forward realized volatility. Once parameters are
    frozen the natural object is a genuine multi-step forecast, so this now
    predicts what it is graded on:

        sigma_5d(t) = sqrt(mean(sigma^2_{t+1..t+5})) * sqrt(252)

    Expect the GARCH family's numbers to move substantially against the
    published table. This is the single largest source of divergence, and it
    cuts at the paper's headline finding, since the GARCH family is what wins
    the high-volatility subperiod.

`--legacy-quirks` restores both original behaviours. It exists only so the
regression test can prove this refactor is behaviour-preserving before the
fixes are switched on; no legacy output belongs in the analysis.

Usage:
    python code/03_run_core_models.py --market us
    python code/03_run_core_models.py --market br_long
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

QLIKE_EPS = 1e-8


# ----------------------------------------------------------------------------
# GARCH family
# ----------------------------------------------------------------------------

GARCH_SPECS = [
    ("GARCH", "Garch", {"p": 1, "q": 1}),
    ("EGARCH", "EGARCH", {"p": 1, "q": 1}),
    ("GJR-GARCH", "GARCH", {"p": 1, "o": 1, "q": 1}),
]


def _garch_batches(index: pd.DatetimeIndex, size: int):
    for i in range(0, len(index), size):
        yield index[i:i + size]


def fit_garch(returns: pd.Series, cfg: config.MarketConfig, legacy: bool,
              meta: dict) -> dict[str, pd.Series]:
    from arch import arch_model

    ann = np.sqrt(cfg.annualization)
    scaled = returns.dropna() * 100
    test_index = scaled[cfg.test_start:].index

    out = {name: [] for name, _, _ in GARCH_SPECS}
    dates: list = []
    fit_log: list[dict] = []

    for batch in _garch_batches(test_index, cfg.garch_refit):
        if legacy:
            train = scaled[:batch[-1]]
        else:
            train = scaled[scaled.index < batch[0]]

        fit_log.append({
            "batch_start": str(batch[0].date()),
            "batch_end": str(batch[-1].date()),
            "train_rows": int(len(train)),
            "train_end": str(train.index[-1].date()),
        })

        for name, vol, kwargs in GARCH_SPECS:
            try:
                am = arch_model(train, vol=vol, **kwargs, dist="Normal",
                                mean="Zero")
                res = am.fit(disp="off", show_warning=False,
                             options={"maxiter": 150})

                if legacy:
                    # Original behaviour: in-sample filtered one-day vol.
                    values = (res.conditional_volatility.loc[batch] * ann / 100
                              ).values[:len(batch)]
                else:
                    values = _walkforward_forecast(
                        arch_model, scaled, batch, res.params, vol, kwargs,
                        cfg, ann,
                    )
                out[name].extend(values)
            except Exception as exc:  # noqa: BLE001 - a failed fit is a NaN
                print(f"    {name} failed on {batch[0].date()}: "
                      f"{type(exc).__name__}: {exc}")
                out[name].extend([np.nan] * len(batch))

        dates.extend(batch)
        print(f"  GARCH: {len(dates)}/{len(test_index)}")

    meta["garch"] = {
        "mode": "legacy_in_sample_filter" if legacy else "walkforward_forecast",
        "horizon": 1 if legacy else cfg.horizon,
        "refit_days": cfg.garch_refit,
        "batches": fit_log,
    }
    return {k: pd.Series(v, index=dates[:len(v)], name=k) for k, v in out.items()}


def _walkforward_forecast(arch_model, scaled: pd.Series, batch, params,
                          vol: str, kwargs: dict, cfg: config.MarketConfig,
                          ann: float) -> np.ndarray:
    """Multi-step variance forecast from every origin in `batch`.

    Parameters were estimated strictly before the batch and are frozen here via
    `.fix()`. The variance *filter* still advances with each new observation,
    which is correct: the return at time t is information available at time t.
    """
    full = scaled[scaled.index <= batch[-1]]
    fixed = arch_model(full, vol=vol, **kwargs, dist="Normal",
                       mean="Zero").fix(params, first_obs=full.index[0])

    try:
        fc = fixed.forecast(horizon=cfg.horizon, start=batch[0], reindex=False)
        var = fc.variance
        if var.isna().all(axis=None):
            raise ValueError("analytic forecast unavailable")
    except Exception:
        # EGARCH has no closed-form multi-step forecast; simulate. Seeded so
        # the run stays reproducible.
        fc = fixed.forecast(
            horizon=cfg.horizon, start=batch[0], reindex=False,
            method="simulation", simulations=1000,
            rng=np.random.default_rng(cfg.seed).standard_normal,
        )
        var = fc.variance

    var = var.reindex(batch)
    # Average variance over t+1..t+h, then annualize. /100 undoes the x100
    # rescaling applied to returns before fitting.
    return (np.sqrt(var.mean(axis=1)) * ann / 100).values


# ----------------------------------------------------------------------------
# Walk-forward splits shared by HAR and the tree models
# ----------------------------------------------------------------------------

def walkforward_split(df: pd.DataFrame, batch_start_pos: int,
                      cfg: config.MarketConfig, legacy: bool):
    """Return (train_slice, val_slice) for a batch beginning at a position.

    Both are strictly before the batch. `embargo` rows are held out between
    train and val, and between val and the batch, because rv_5d_fwd at row t is
    built from returns at t+1..t+5 and would otherwise reach across the seam.
    """
    if legacy:
        train = df.iloc[:batch_start_pos + 1]
        val = df[cfg.train_end:cfg.val_end]
        return train, val

    val_end_pos = batch_start_pos - cfg.embargo
    val_start_pos = val_end_pos - cfg.val_window
    train_end_pos = val_start_pos - cfg.embargo

    if train_end_pos <= cfg.val_window:
        raise SystemExit(
            f"not enough history before {df.index[batch_start_pos].date()} for "
            f"a {cfg.val_window}-row validation block; shorten val_window or "
            f"start the sample earlier."
        )

    return df.iloc[:train_end_pos], df.iloc[val_start_pos:val_end_pos]


def fit_har(df: pd.DataFrame, cfg: config.MarketConfig, legacy: bool,
            meta: dict) -> pd.Series:
    from sklearn.linear_model import LinearRegression

    feats = ["har_daily", "har_weekly", "har_monthly"]
    test = df[cfg.test_start:]
    preds: list = []
    dates: list = []

    for i in range(0, len(test), cfg.model_refit):
        batch = test.iloc[i:i + cfg.model_refit]
        pos = df.index.get_loc(batch.index[0])
        train, _ = walkforward_split(df, pos, cfg, legacy)

        m = LinearRegression().fit(train[feats].values, train[cfg.target].values)
        preds.extend(m.predict(batch[feats].values))
        dates.extend(batch.index)

    meta["har"] = {"refit_days": cfg.model_refit, "features": feats}
    return pd.Series(preds, index=dates[:len(preds)], name="HAR-RV")


def fit_trees(df: pd.DataFrame, cfg: config.MarketConfig, legacy: bool,
              meta: dict, fcols: list[str]):
    import lightgbm as lgb
    import xgboost as xgb

    out = {"LightGBM": [], "XGBoost": []}
    dates: list = []
    fit_log: list[dict] = []
    test = df[cfg.test_start:]

    def clean(frame):
        return np.nan_to_num(frame.values, nan=0, posinf=0, neginf=0)

    for i in range(0, len(test), cfg.model_refit):
        batch = test.iloc[i:i + cfg.model_refit]
        pos = df.index.get_loc(batch.index[0])
        train, val = walkforward_split(df, pos, cfg, legacy)

        X_tr, y_tr = clean(train[fcols]), clean(train[cfg.target])
        X_va, y_va = clean(val[fcols]), clean(val[cfg.target])
        X_te = clean(batch[fcols])

        lgb_params = {
            "objective": "regression", "metric": "mse", "num_leaves": 31,
            "learning_rate": 0.05, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1,
            "n_jobs": -1,
        }
        xgb_params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "early_stopping_rounds": 30, "eval_metric": "rmse",
            "verbosity": 0, "n_jobs": -1,
        }
        if not legacy:
            # The original set no seeds and relied on library defaults. Those
            # happen to be deterministic today, but nothing defends that if a
            # default ever changes. Legacy mode must keep using the defaults,
            # or the regression test would compare different models.
            lgb_params.update({
                "seed": cfg.seed, "bagging_seed": cfg.seed,
                "feature_fraction_seed": cfg.seed, "data_random_seed": cfg.seed,
            })
            xgb_params["random_state"] = cfg.seed

        ds_tr = lgb.Dataset(X_tr, y_tr)
        ds_va = lgb.Dataset(X_va, y_va, reference=ds_tr)
        lgb_m = lgb.train(
            lgb_params, ds_tr, 500, valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        out["LightGBM"].extend(lgb_m.predict(X_te))

        xgb_m = xgb.XGBRegressor(**xgb_params)
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        out["XGBoost"].extend(xgb_m.predict(X_te))

        fit_log.append({
            "batch_start": str(batch.index[0].date()),
            "train_rows": int(len(train)),
            "train_end": str(train.index[-1].date()),
            "val_rows": int(len(val)),
            "val_start": str(val.index[0].date()),
            "val_end": str(val.index[-1].date()),
            "lgb_best_iteration": int(lgb_m.best_iteration or 500),
            "xgb_best_iteration": int(getattr(xgb_m, "best_iteration", 500) or 500),
        })
        dates.extend(batch.index)
        print(f"  Trees: {len(dates)}/{len(test)}")

    meta["trees"] = {
        "mode": "legacy_nested_validation" if legacy else "walkforward",
        "refit_days": cfg.model_refit,
        "val_window": cfg.val_window,
        "embargo": cfg.embargo,
        "n_features": len(fcols),
        "features": fcols,
        "batches": fit_log,
    }
    return {k: pd.Series(v, index=dates[:len(v)], name=k) for k, v in out.items()}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    from scipy.stats import pearsonr

    common = actual.index.intersection(predicted.index)
    a = actual.loc[common].values
    p = predicted.loc[common].values
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]

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


def assert_no_leakage(meta: dict, cfg: config.MarketConfig) -> None:
    """Verification 3 and 4 from the plan, enforced at runtime."""
    for batch in meta.get("garch", {}).get("batches", []):
        if batch["train_end"] >= batch["batch_start"]:
            raise AssertionError(
                f"GARCH fit window ends {batch['train_end']} but the batch "
                f"starts {batch['batch_start']} -- look-ahead in parameters."
            )
    for batch in meta.get("trees", {}).get("batches", []):
        if not (batch["train_end"] < batch["val_start"] <= batch["val_end"]
                < batch["batch_start"]):
            raise AssertionError(
                f"train/val/test ordering violated at {batch['batch_start']}: "
                f"train_end={batch['train_end']} val={batch['val_start']}.."
                f"{batch['val_end']}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_market_arg(ap)
    ap.add_argument("--legacy-quirks", action="store_true",
                    help="restore the original nested validation split and "
                         "in-sample GARCH filter (regression test only)")
    ap.add_argument("--input", type=Path, default=None,
                    help="override the input panel path")
    ap.add_argument("--out", type=Path, default=None,
                    help="override the results directory")
    args = ap.parse_args()

    cfg = config.get(args.market)
    legacy = args.legacy_quirks
    in_path = args.input or (cfg.data_dir / "combined.parquet")
    out_dir = args.out or cfg.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"Volatility horse race -- market={cfg.name} "
          f"({cfg.equity_label}){'  [LEGACY QUIRKS]' if legacy else ''}")
    print("=" * 68)

    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df):,} rows, {df.index[0].date()} to "
          f"{df.index[-1].date()} from {in_path}")

    fcols = cfg.feature_columns(df)
    print(f"Features ({len(fcols)}): {', '.join(fcols)}")

    meta: dict = {
        "market": cfg.name,
        "legacy_quirks": legacy,
        "target": cfg.target,
        "annualization": cfg.annualization,
        "seed": cfg.seed,
        "input": str(in_path),
    }
    forecasts: dict[str, pd.Series] = {}

    print("\n[1/3] GARCH family...")
    forecasts.update(fit_garch(df["log_return"], cfg, legacy, meta))

    print("\n[2/3] HAR-RV...")
    forecasts["HAR-RV"] = fit_har(df, cfg, legacy, meta)

    print("\n[3/3] LightGBM + XGBoost...")
    forecasts.update(fit_trees(df, cfg, legacy, meta, fcols))

    print("\nBuilding ensemble...")
    common = None
    for member in cfg.ensemble_members:
        idx = forecasts[member].index
        common = idx if common is None else common.intersection(idx)
    ens = sum(forecasts[m].loc[common] for m in cfg.ensemble_members) / len(
        cfg.ensemble_members)
    ens.name = "Ensemble"
    forecasts["Ensemble"] = ens

    if not legacy:
        assert_no_leakage(meta, cfg)
        print("  leakage assertions passed")

    actual = df[cfg.test_start:][cfg.target]
    all_metrics = {name: metrics(actual, preds) for name, preds in forecasts.items()}

    frame = pd.DataFrame(forecasts)
    frame["actual"] = actual
    frame.to_parquet(out_dir / "forecasts_5d.parquet")

    pd.DataFrame(all_metrics).T.to_csv(out_dir / "metrics_5d.csv")
    with open(out_dir / "metrics_5d.json", "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    with open(out_dir / "fit_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n" + "=" * 68)
    print(f"RANKING (QLIKE, lower = better) -- market={cfg.name}")
    print("=" * 68)
    ranked = sorted(all_metrics.items(), key=lambda kv: kv[1]["QLIKE"])
    for i, (name, m) in enumerate(ranked, 1):
        print(f"  {i}. {name:<12s} QLIKE={m['QLIKE']:.6f}  "
              f"RMSE={m['RMSE']:.6f}  MZ_R2={m['MZ_R2']:.4f}  N={m['N']}")

    if not legacy:
        iters = [b["lgb_best_iteration"] for b in meta["trees"]["batches"]]
        early = sum(1 for it in iters if it < 500)
        print(f"\nEarly stopping fired in {early}/{len(iters)} LightGBM fits "
              f"(median best_iteration={int(np.median(iters))})")

    print(f"\nWrote artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
