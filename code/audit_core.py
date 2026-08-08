"""
Market-agnostic audit primitives.

The original 06_audit.py checked the *paper text* against the *committed
artifacts*, with the paper's values hard-coded in a Python dict. That proves
internal consistency, not reproduction: it never re-runs a model, so it cannot
detect that regenerating the artifacts produces different numbers. Which is
exactly what happens -- re-running the committed pipeline on a current
toolchain shifts XGBoost's QLIKE by 0.0067 and fails five of its checks while
the audit on the shipped parquet still passes.

So the checks here split in two:

  * invariants that must hold for any market, checked against recomputation
    from the forecasts panel (structure, ensemble identity, metrics files, DM
    tests, and the absence of look-ahead recorded in fit_metadata.json);
  * expected headline values loaded from a JSON file rather than hard-coded,
    so the same code audits the US and both Brazilian arms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

QLIKE_EPS = 1e-8


class Report:
    """Accumulates pass/fail lines so a caller can exit non-zero."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def ok(self, msg: str) -> None:
        self.checks += 1
        print(f"  ok:   {msg}")

    def fail(self, msg: str) -> None:
        self.checks += 1
        self.failures.append(msg)
        print(f"  FAIL: {msg}")

    def check(self, condition: bool, msg: str) -> bool:
        (self.ok if condition else self.fail)(msg)
        return condition

    def close(self, msg: str, actual: float, expected: float,
              tol: float = 1e-4) -> bool:
        diff = abs(actual - expected)
        return self.check(
            diff < tol,
            f"{msg}: expected={expected:.6f} computed={actual:.6f} diff={diff:.2e}",
        )

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")


def qlike_loss(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    ratio = (actual ** 2) / (predicted ** 2 + QLIKE_EPS)
    return ratio - np.log(ratio + QLIKE_EPS) - 1


def newey_west_se(d: np.ndarray, q: int) -> float:
    d = d - d.mean()
    n = len(d)
    s = (d @ d) / n
    for k in range(1, q + 1):
        s += 2 * (1 - k / (q + 1)) * ((d[k:] @ d[:-k]) / n)
    return float(np.sqrt(s / n)) if s > 0 else float("nan")


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict | None:
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    # pearsonr needs at least two points, so guard on 2 rather than 0.
    if len(a) < 2:
        return None
    mse = np.mean((a - p) ** 2)
    ratio = (a ** 2) / (p ** 2 + QLIKE_EPS)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    corr, _ = stats.pearsonr(a, p)
    return {
        "MSE": float(mse),
        "MAE": float(np.mean(np.abs(a - p))),
        "RMSE": float(np.sqrt(mse)),
        "QLIKE": float(np.mean(ratio - np.log(ratio + QLIKE_EPS) - 1)),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "MZ_R2": float(corr ** 2),
        "N": int(len(a)),
    }


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------

def check_structure(fc: pd.DataFrame, rep: Report) -> None:
    rep.section("STRUCTURE: no NaN, inf, negative or zero forecasts")
    for col in fc.columns:
        v = fc[col].values
        if np.any(np.isnan(v)):
            rep.fail(f"{col} has {int(np.isnan(v).sum())} NaN")
        elif np.any(np.isinf(v)):
            rep.fail(f"{col} has inf")
        elif np.any(v < 0):
            rep.fail(f"{col} has {int((v < 0).sum())} negative values")
        elif np.any(v == 0):
            rep.fail(f"{col} has {int((v == 0).sum())} zero values")
        else:
            rep.ok(f"{col}: min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")


def check_ensemble(fc: pd.DataFrame, members: list[str], rep: Report) -> None:
    rep.section(f"ENSEMBLE IDENTITY: mean of {', '.join(members)}")
    if "Ensemble" not in fc.columns:
        rep.fail("no Ensemble column")
        return
    expected = sum(fc[m] for m in members) / len(members)
    diff = float((fc["Ensemble"] - expected).abs().max())
    rep.check(diff < 1e-9, f"ensemble identity holds, max-abs-diff={diff:.2e}")


def check_metrics_file(fc: pd.DataFrame, path: Path, rep: Report) -> None:
    rep.section("METRICS FILE vs recomputation from the forecasts panel")
    if not path.exists():
        rep.fail(f"missing {path.name}")
        return
    stored = json.load(open(path))
    actual = fc["actual"].values
    for model in [c for c in fc.columns if c != "actual"]:
        if model not in stored:
            rep.fail(f"{model} absent from {path.name}")
            continue
        computed = metrics(actual, fc[model].values)
        for key in ["QLIKE", "RMSE", "MAE", "MZ_R2"]:
            if key in stored[model]:
                rep.close(f"{model}.{key}", computed[key], stored[model][key])


def check_subperiods(fc: pd.DataFrame, path: Path, split_year: int,
                     quantile: float, rep: Report) -> None:
    rep.section("SUBPERIOD FILE vs recomputation")
    if not path.exists():
        rep.fail(f"missing {path.name}")
        return
    stored = json.load(open(path))
    actual = fc["actual"].values
    models = [c for c in fc.columns if c != "actual"]

    masks = {
        f"{split_year}_high_vol": np.asarray(fc.index.year == split_year),
        f"after_{split_year}_lower_vol": np.asarray(fc.index.year > split_year),
    }
    by_year = stored.get("by_year", {})
    for label, mask in masks.items():
        block = by_year.get(label)
        if block is None:
            # The legacy root-level snapshot names the post-split block
            # "2023_2025_lower_vol" rather than "after_2022_lower_vol".
            block = next(
                (v for k, v in by_year.items()
                 if k != f"{split_year}_high_vol" and label.startswith("after_")),
                None,
            )
        if block is None:
            rep.fail(f"by_year.{label} absent")
            continue
        rep.check(block["N"] == int(mask.sum()),
                  f"by_year.{label}.N = {int(mask.sum())}")
        for m in models:
            computed = metrics(actual[mask], fc[m].values[mask])
            rep.close(f"{label}.{m}.QLIKE", computed["QLIKE"],
                      block["models"][m]["QLIKE"])

    threshold = float(np.nanpercentile(actual, quantile))
    stored_threshold = stored.get("by_regime", {}).get("meta", {}).get("threshold")
    if stored_threshold is not None:
        rep.close("regime threshold", threshold, stored_threshold)


def check_dm(fc: pd.DataFrame, path: Path, horizon: int, rep: Report) -> None:
    rep.section("DM TESTS vs re-derivation")
    if not path.exists():
        rep.fail(f"missing {path.name}")
        return
    stored = json.load(open(path))["pairs"]
    actual = fc["actual"].values
    models = [c for c in fc.columns if c != "actual"]
    losses = {m: qlike_loss(actual, fc[m].values) for m in models}
    q = horizon - 1

    for key, expected in stored.items():
        a, b = key.split(" vs ")
        d = losses[a] - losses[b]
        d = d[~np.isnan(d)]
        se = newey_west_se(d, q)
        t = d.mean() / se if se and not np.isnan(se) and se > 0 else float("nan")
        p = 2 * (1 - stats.norm.cdf(abs(t))) if not np.isnan(t) else float("nan")
        if abs(t - expected["t_stat"]) > 0.01 or abs(p - expected["p_value"]) > 0.001:
            rep.fail(f"DM {key}: recomputed t={t:.3f} p={p:.4f}, "
                     f"stored t={expected['t_stat']:.3f} p={expected['p_value']:.4f}")
        else:
            rep.ok(f"DM {key}: t={t:+.3f} p={p:.4f}")


def check_no_leakage(path: Path, rep: Report) -> None:
    """Re-verify from fit_metadata.json that no window saw its own test data."""
    rep.section("LOOK-AHEAD: fit windows strictly precede their batches")
    if not path.exists():
        rep.fail(f"missing {path.name}")
        return
    meta = json.load(open(path))

    if meta.get("legacy_quirks"):
        rep.fail("fit_metadata records legacy_quirks=true; this run carries "
                 "the original nested validation split and in-sample GARCH "
                 "filter and must not be used for analysis")
        return

    garch = meta.get("garch", {})
    bad = [b for b in garch.get("batches", [])
           if b["train_end"] >= b["batch_start"]]
    rep.check(not bad,
              f"GARCH: all {len(garch.get('batches', []))} fit windows end "
              f"before their batch starts")
    rep.check(garch.get("mode") == "walkforward_forecast" and garch.get("horizon", 0) > 1,
              f"GARCH mode={garch.get('mode')} horizon={garch.get('horizon')} "
              f"(a genuine multi-step forecast, not a 1-day in-sample filter "
              f"scored against a 5-day target)")

    trees = meta.get("trees", {})
    bad = [b for b in trees.get("batches", [])
           if not (b["train_end"] < b["val_start"] <= b["val_end"] < b["batch_start"])]
    rep.check(not bad,
              f"trees: train < val < test ordering holds for all "
              f"{len(trees.get('batches', []))} batches")

    iters = [b["lgb_best_iteration"] for b in trees.get("batches", [])]
    if iters:
        fired = sum(1 for i in iters if i < 500)
        rep.check(fired > 0,
                  f"early stopping fired in {fired}/{len(iters)} LightGBM fits "
                  f"(median best_iteration={int(np.median(iters))})")


def check_expected(fc: pd.DataFrame, path: Path, rep: Report) -> None:
    """Compare headline numbers against a committed expectations file."""
    rep.section(f"EXPECTED VALUES ({path.name})")
    if not path.exists():
        rep.ok(f"{path.name} absent; skipping (generate with --write-expected)")
        return

    expected = json.load(open(path))
    actual = fc["actual"].values

    for model, block in expected.get("full_sample", {}).items():
        if model not in fc.columns:
            rep.fail(f"{model} not in forecasts panel")
            continue
        computed = metrics(actual, fc[model].values)
        for key, value in block.items():
            rep.close(f"full.{model}.{key}", computed[key], value, tol=5e-4)

    for label, block in expected.get("by_year", {}).items():
        if label.startswith("after_"):
            mask = np.asarray(fc.index.year > int(label.split("_")[1]))
        else:
            mask = np.asarray(fc.index.year == int(label.split("_")[0]))
        for model, value in block.items():
            computed = metrics(actual[mask], fc[model].values[mask])
            rep.close(f"{label}.{model}.QLIKE", computed["QLIKE"], value, tol=5e-4)


def build_expected(fc: pd.DataFrame, split_year: int) -> dict:
    """Generate an expectations file from the current artifacts."""
    actual = fc["actual"].values
    models = [c for c in fc.columns if c != "actual"]

    out: dict = {"full_sample": {}, "by_year": {}}
    for m in models:
        c = metrics(actual, fc[m].values)
        out["full_sample"][m] = {k: round(c[k], 6)
                                 for k in ["QLIKE", "RMSE", "MAE", "MZ_R2"]}

    for label, mask in [
        (f"{split_year}_high_vol", np.asarray(fc.index.year == split_year)),
        (f"after_{split_year}_lower_vol", np.asarray(fc.index.year > split_year)),
    ]:
        out["by_year"][label] = {
            m: round(metrics(actual[mask], fc[m].values[mask])["QLIKE"], 6)
            for m in models
        }
    return out
