"""
Regression tests for the multi-market pipeline.

These exist mainly to guard the two corrections in 03_run_core_models.py. The
original defects were both invisible at the call site -- a nested validation
split and an in-sample GARCH filter each look perfectly reasonable in the code
that produced them, and neither raised anything. The tests that matter here are
therefore the ones that assert on window *boundaries*, not on outputs.

Run:  python -m pytest tests/ -v
      python -m pytest tests/ -v -m "not slow"     # skip the 4-minute one
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code"))

import audit_core  # noqa: E402
import config  # noqa: E402


def _load(stem: str):
    """Import a module whose filename starts with a digit."""
    spec = importlib.util.spec_from_file_location(
        stem.replace(".py", ""), REPO / "code" / stem)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("03_run_core_models.py")
collector = _load("01_collect_data.py")

LEGACY_PANEL = REPO / "data" / "combined.parquet"


@pytest.fixture(scope="module")
def panel():
    if not LEGACY_PANEL.exists():
        pytest.skip("legacy US panel not present")
    return pd.read_parquet(LEGACY_PANEL)


# ---------------------------------------------------------------------------
# Market configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("market,with_vol,without_vol", [
    ("us", 30, 28),
    ("us_2018", 30, 28),
    ("br_long", 26, 24),
    ("br_iv", 30, 28),
])
def test_feature_counts(panel, market, with_vol, without_vol):
    """Arm A drops exactly the four implied-vol features."""
    cfg = config.get(market)
    assert len(cfg.feature_columns(panel, use_volume=True)) == with_vol
    assert len(cfg.feature_columns(panel, use_volume=False)) == without_vol


def test_iv_features_absent_from_arm_a(panel):
    cols = config.get("br_long").feature_columns(panel)
    assert not [c for c in cols if c.startswith(("iv_", "vix_"))]


def test_both_iv_prefixes_recognised():
    """New panels use iv_*; the committed legacy panel uses vix_*."""
    cfg = config.get("us")
    legacy = pd.DataFrame(columns=["vix_close", "vix_lag1", "har_daily"])
    modern = pd.DataFrame(columns=["iv_close", "iv_lag1", "har_daily"])
    assert len(cfg.feature_columns(legacy, use_volume=False)) == 3
    assert len(cfg.feature_columns(modern, use_volume=False)) == 3


def test_forecast_targets_never_become_features(panel):
    for market in config.MARKETS:
        cols = config.get(market).feature_columns(panel)
        assert not [c for c in cols if "_fwd" in c]


def test_brazilian_arms_share_evaluation_windows():
    """The A-vs-B comparison is only clean if the windows are identical."""
    a, b = config.get("br_long"), config.get("br_iv")
    assert (a.test_start, a.test_end) == (b.test_start, b.test_end)
    assert a.val_end == b.val_end
    assert a.train_end == b.train_end
    assert a.start != b.start  # ... and the training start is the only change


def test_us_2018_matches_the_brazilian_window():
    """Otherwise Brazil-vs-US contrasts different regimes, not markets."""
    us18, br = config.get("us_2018"), config.get("br_iv")
    assert (us18.test_start, us18.test_end) == (br.test_start, br.test_end)
    assert us18.split_year == br.split_year


def test_iv_arm_stops_where_ivolbr_stops():
    """IVol-BR's last observation is 2022-04-29. Evaluating past it would
    silently score against a forward-filled constant."""
    br = config.get("br_iv")
    assert br.test_end == "2022-04-29"
    assert br.start == "2011-08-01"


def test_split_year_falls_inside_every_test_window():
    """subperiod_metrics splits on == split_year; a split year outside the
    window would leave one block empty."""
    for name, cfg in config.MARKETS.items():
        lo = int(cfg.test_start[:4])
        hi = int(cfg.test_end[:4]) if cfg.test_end else 9999
        assert lo <= cfg.split_year <= hi, name


def test_annualization_is_252_everywhere():
    """252 is the Brazilian 'dias uteis' convention, not a US leftover."""
    for market in config.MARKETS:
        assert config.get(market).annualization == 252


# ---------------------------------------------------------------------------
# FIX 1: the validation block must not overlap training
# ---------------------------------------------------------------------------

def test_walkforward_split_is_disjoint_and_embargoed(panel):
    cfg = config.get("us")
    batch_pos = panel.index.get_loc(panel[cfg.test_start:].index[0])
    train, val = runner.walkforward_split(panel, batch_pos, cfg, legacy=False)

    assert len(train) > 0 and len(val) == cfg.val_window
    assert train.index.intersection(val.index).empty, "train and val overlap"

    train_end = panel.index.get_loc(train.index[-1])
    val_start = panel.index.get_loc(val.index[0])
    val_end = panel.index.get_loc(val.index[-1])

    assert val_start - train_end >= cfg.embargo, "embargo violated train->val"
    assert batch_pos - val_end >= cfg.embargo, "embargo violated val->test"
    assert val_end < batch_pos, "validation reaches into the test batch"


def test_walkforward_split_holds_for_every_batch(panel):
    cfg = config.get("us")
    test = panel[cfg.test_start:]
    for i in range(0, len(test), cfg.model_refit):
        pos = panel.index.get_loc(test.index[i])
        train, val = runner.walkforward_split(panel, pos, cfg, legacy=False)
        assert train.index[-1] < val.index[0] < val.index[-1] < test.index[i]


def test_legacy_split_reproduces_the_original_defect(panel):
    """The flag is only meaningful if it really restores the nested split."""
    cfg = config.get("us")
    pos = panel.index.get_loc(panel[cfg.test_start:].index[0])
    train, val = runner.walkforward_split(panel, pos, cfg, legacy=True)
    assert not train.index.intersection(val.index).empty, (
        "legacy mode should reproduce the nested validation set"
    )


# ---------------------------------------------------------------------------
# Loss functions and HAC standard errors
# ---------------------------------------------------------------------------

def test_qlike_is_minimised_by_a_perfect_forecast():
    actual = np.array([0.10, 0.20, 0.35, 0.15])
    assert audit_core.qlike_loss(actual, actual).sum() == pytest.approx(0, abs=1e-6)
    worse = audit_core.qlike_loss(actual, actual * 1.5)
    assert (worse > 0).all()


def test_qlike_penalises_under_forecasting_more_than_over():
    """The asymmetry is the reason QLIKE is preferred for volatility."""
    actual = np.array([0.20])
    under = audit_core.qlike_loss(actual, np.array([0.10]))[0]
    over = audit_core.qlike_loss(actual, np.array([0.40]))[0]
    assert under > over


def test_newey_west_reduces_to_the_plain_standard_error_at_zero_lag():
    rng = np.random.default_rng(0)
    d = rng.standard_normal(500)
    expected = np.sqrt(((d - d.mean()) @ (d - d.mean())) / len(d) / len(d))
    assert audit_core.newey_west_se(d, 0) == pytest.approx(expected, rel=1e-12)


def test_metrics_ignore_nan_pairs():
    actual = np.array([0.1, np.nan, 0.3, 0.4])
    pred = np.array([0.1, 0.2, 0.35, 0.45])
    assert audit_core.metrics(actual, pred)["N"] == 3


def test_metrics_returns_none_below_two_observations():
    """pearsonr raises on a single point; metrics must not propagate that."""
    actual = np.array([0.1, np.nan, np.nan])
    pred = np.array([0.1, 0.2, 0.3])
    assert audit_core.metrics(actual, pred) is None


# ---------------------------------------------------------------------------
# Data-quality gates
# ---------------------------------------------------------------------------

def test_volume_gate_disables_features_on_a_degenerate_series():
    """^BVSP volume is frequently zero; that must switch the features off
    rather than flow through nan_to_num into a column of zeros."""
    cfg = config.get("br_long")  # use_volume=None -> auto
    px = pd.DataFrame({"volume": [0] * 60 + [1_000] * 40})
    report: dict = {}
    assert collector.gate_volume(px, cfg, report) is False
    assert report["volume_bad_fraction"] == pytest.approx(0.6)


def test_volume_gate_keeps_a_clean_series():
    cfg = config.get("br_long")
    px = pd.DataFrame({"volume": list(range(1, 101))})
    assert collector.gate_volume(px, cfg, {}) is True


def test_volume_gate_respects_an_explicit_config_override():
    cfg = config.get("us")  # use_volume=True
    px = pd.DataFrame({"volume": [0] * 100})
    assert collector.gate_volume(px, cfg, {}) is True


def test_iv_scale_gate_rejects_a_decimal_quoted_series():
    """iv_lag1/iv_lag5 divide by 100, so a decimal series would be silently
    wrong by two orders of magnitude."""
    decimals = pd.DataFrame({"iv_close": np.full(100, 0.19)})
    with pytest.raises(collector.DataQualityError, match="percentage points"):
        collector.gate_iv_scale(decimals, {})

    points = pd.DataFrame({"iv_close": np.full(100, 19.0)})
    collector.gate_iv_scale(points, {})  # must not raise


def test_iv_ffill_gate_rejects_a_badly_mismatched_calendar():
    cfg = config.get("br_iv")
    with pytest.raises(collector.DataQualityError, match="forward-filling"):
        collector.gate_iv_ffill({"iv_ffill_fraction": 0.25}, cfg)
    collector.gate_iv_ffill({"iv_ffill_fraction": 0.01}, cfg)


def test_ivolbr_gets_a_looser_ffill_ceiling_than_vix():
    """IVol-BR has ~4% blanks and skips B3 sessions; ^VIX does neither."""
    assert config.get("br_iv").max_iv_ffill > config.get("us").max_iv_ffill


def test_equity_csv_does_not_mangle_iso_dates(tmp_path):
    """dayfirst parsing turns 2020-01-02 into 1 February. It must not."""
    path = tmp_path / "px.csv"
    path.write_text("Date,Open,High,Low,Close,Volume\n"
                    "2020-01-02,1,2,0.5,1.5,10\n"
                    "2020-01-03,1,2,0.5,1.4,12\n")
    df = collector.load_equity_csv(path)
    assert [str(d.date()) for d in df.index] == ["2020-01-02", "2020-01-03"]


def test_equity_csv_accepts_portuguese_headers(tmp_path):
    path = tmp_path / "px.csv"
    path.write_text("Data,Abertura,Maxima,Minima,Fechamento\n"
                    "02/01/2020,1,2,0.5,1.5\n"
                    "03/01/2020,1,2,0.5,1.4\n")
    df = collector.load_equity_csv(path)
    assert {"open", "high", "low", "close"} <= set(df.columns)
    assert [str(d.date()) for d in df.index] == ["2020-01-02", "2020-01-03"]


def test_equity_csv_rejects_a_table_without_prices(tmp_path):
    path = tmp_path / "px.csv"
    path.write_text("Date,Something\n2020-01-02,1\n")
    with pytest.raises(collector.DataQualityError, match="missing required"):
        collector.load_equity_csv(path)


def test_parses_the_nefin_ivolbr_layout():
    """IVol-BR ships year/month/day integer columns, not a date column, and
    carries blank values that must be dropped rather than coerced to zero."""
    raw = pd.DataFrame({
        "year": [2011, 2011, 2011],
        "month": [8, 8, 8],
        "day": [1, 4, 5],
        "ivolbr": [21.28711, None, 23.56743],
    })
    out = collector._parse_iv_table(raw, "IVol-BR")
    assert list(out.columns) == ["close"]
    assert len(out) == 2, "the blank row should be dropped, not filled"
    assert str(out.index[0].date()) == "2011-08-01"
    assert out["close"].iloc[0] == pytest.approx(21.28711)


@pytest.mark.skipif(not (REPO / "data" / "br_iv" / "ivolbr_raw.csv").exists(),
                    reason="IVol-BR snapshot not present")
def test_committed_ivolbr_snapshot_matches_the_configured_window():
    out = collector._parse_iv_table(
        pd.read_csv(REPO / "data" / "br_iv" / "ivolbr_raw.csv"), "IVol-BR")
    cfg = config.get("br_iv")
    assert str(out.index[0].date()) == cfg.start
    assert str(out.index[-1].date()) == cfg.test_end
    # Percentage points, so the /100 rescaling in build_features is right.
    assert 1.0 < out["close"].median() < 200.0


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def test_audit_rejects_legacy_artifacts(tmp_path):
    """Legacy runs carry the defects and must never be audited as analysis."""
    meta = tmp_path / "fit_metadata.json"
    meta.write_text(json.dumps({"legacy_quirks": True, "garch": {}, "trees": {}}))
    rep = audit_core.Report()
    audit_core.check_no_leakage(meta, rep)
    assert rep.failures, "an audit of a legacy run should fail"


def test_audit_catches_garch_lookahead(tmp_path):
    meta = tmp_path / "fit_metadata.json"
    meta.write_text(json.dumps({
        "legacy_quirks": False,
        "garch": {"mode": "walkforward_forecast", "horizon": 5, "batches": [
            {"batch_start": "2022-01-03", "train_end": "2022-02-01"},
        ]},
        "trees": {"batches": []},
    }))
    rep = audit_core.Report()
    audit_core.check_no_leakage(meta, rep)
    assert any("fit windows end" in f for f in rep.failures)


def test_audit_catches_a_broken_ensemble_identity():
    fc = pd.DataFrame({
        "LightGBM": [0.1, 0.2], "HAR-RV": [0.2, 0.3], "GARCH": [0.3, 0.4],
        "Ensemble": [0.9, 0.9],
    })
    rep = audit_core.Report()
    audit_core.check_ensemble(fc, ["LightGBM", "HAR-RV", "GARCH"], rep)
    assert rep.failures


def test_expected_roundtrip_is_self_consistent(tmp_path):
    idx = pd.to_datetime(
        [f"2022-01-{d:02d}" for d in range(1, 16)]
        + [f"2023-01-{d:02d}" for d in range(1, 16)]
    )
    rng = np.random.default_rng(0)
    actual = pd.Series(rng.uniform(0.1, 0.3, len(idx)), index=idx)
    fc = pd.DataFrame({"GARCH": actual * 1.05, "actual": actual}, index=idx)

    payload = audit_core.build_expected(fc, split_year=2022)
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(payload))

    rep = audit_core.Report()
    audit_core.check_expected(fc, path, rep)
    assert not rep.failures


# ---------------------------------------------------------------------------
# Slow end-to-end regression
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_legacy_mode_reproduces_the_committed_forecasts(tmp_path):
    """Proves the refactor is behaviour-preserving.

    LightGBM is bit-identical. HAR-RV lands at machine epsilon rather than
    exactly zero -- sklearn's least-squares solve is not bitwise reproducible
    across calls. The GARCH family goes through a numerical optimizer, so ~1e-7
    is the achievable bar there. XGBoost is excluded: it drifts between library
    versions independently of anything in this repo, which is what
    requirements.txt exists to pin.
    """
    import subprocess

    committed = REPO / "results" / "forecasts_5d.parquet"
    if not (committed.exists() and LEGACY_PANEL.exists()):
        pytest.skip("legacy artifacts not present")

    proc = subprocess.run([
        sys.executable, str(REPO / "code" / "03_run_core_models.py"),
        "--market", "us", "--legacy-quirks",
        "--input", str(LEGACY_PANEL), "--out", str(tmp_path),
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]

    old = pd.read_parquet(committed)
    new = pd.read_parquet(tmp_path / "forecasts_5d.parquet")
    assert old.index.equals(new.index)

    assert (old["LightGBM"] - new["LightGBM"]).abs().max() == 0.0, "LightGBM not bit-identical"
    assert (old["HAR-RV"] - new["HAR-RV"]).abs().max() < 1e-12, "HAR-RV drifted"
    for col in ["GARCH", "EGARCH", "GJR-GARCH", "Ensemble"]:
        assert (old[col] - new[col]).abs().max() < 1e-7, f"{col} drifted"
