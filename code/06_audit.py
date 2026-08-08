"""
Step 6: end-to-end audit.

Verifies that a market's committed artifacts are internally consistent and free
of look-ahead, and -- with --regenerate -- that re-running the pipeline
actually reproduces them.

That last part is the point. The original audit compared the paper's prose
against the committed forecasts panel, with the expected values written into a
Python dict. It never re-ran a model, so it could not detect regeneration
drift, and it did not: on a current toolchain the shipped artifacts still pass
while a fresh run of the same pipeline fails five checks.

Usage:
    python code/06_audit.py --market us
    python code/06_audit.py --market us --regenerate
    python code/06_audit.py --market us --write-expected

    # audit the legacy root-level snapshot against the published paper claims
    python code/06_audit.py --market us --results-dir results \\
        --expected results/expected_paper.json

Exits 0 if every check passes, 1 otherwise.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import audit_core as ac  # noqa: E402
import config  # noqa: E402

CODE_DIR = Path(__file__).parent

# Tolerances for --regenerate. GARCH goes through a numerical optimizer, so
# bit-equality is not achievable across runs; the tree models are deterministic
# once seeded, but only within a fixed library version -- see requirements.txt.
REGEN_TOL = 1e-6


def regenerate(cfg: config.MarketConfig, committed: Path, panel: Path,
               rep: ac.Report) -> None:
    rep.section("REGENERATE: rerun the pipeline and diff against committed")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cmd = [
            sys.executable, str(CODE_DIR / "03_run_core_models.py"),
            "--market", cfg.name, "--out", str(tmp_dir),
        ]
        if panel is not None:
            cmd += ["--input", str(panel)]

        print(f"  running: {' '.join(cmd[1:])}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            rep.fail(f"regeneration failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
            return

        fresh = pd.read_parquet(tmp_dir / "forecasts_5d.parquet")
        old = pd.read_parquet(committed / "forecasts_5d.parquet")

        if not old.index.equals(fresh.index):
            rep.fail(f"index changed: {len(old)} committed vs {len(fresh)} fresh rows")
            return

        for col in old.columns:
            diff = float((old[col] - fresh[col]).abs().max())
            rep.check(diff < REGEN_TOL,
                      f"{col} reproduces (max-abs-diff={diff:.3e}, tol={REGEN_TOL:.0e})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_market_arg(ap)
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="override the results directory to audit")
    ap.add_argument("--input", type=Path, default=None,
                    help="override the input panel path (for --regenerate)")
    ap.add_argument("--expected", type=Path, default=None,
                    help="expectations file (default: <results-dir>/expected.json)")
    ap.add_argument("--regenerate", action="store_true",
                    help="rerun the model pipeline and diff against committed "
                         "artifacts")
    ap.add_argument("--write-expected", action="store_true",
                    help="regenerate the expectations file from current "
                         "artifacts instead of auditing against it")
    args = ap.parse_args()

    cfg = config.get(args.market)
    results_dir = args.results_dir or cfg.results_dir
    expected_path = args.expected or (results_dir / "expected.json")

    fc_path = results_dir / "forecasts_5d.parquet"
    if not fc_path.exists():
        print(f"no forecasts panel at {fc_path}", file=sys.stderr)
        return 1
    fc = pd.read_parquet(fc_path)

    if args.write_expected:
        payload = ac.build_expected(fc, cfg.split_year)
        payload["market"] = cfg.name
        with open(expected_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote {expected_path}")
        return 0

    rep = ac.Report()
    print("=" * 68)
    print(f"AUDIT  market={cfg.name}  results={results_dir}")
    print(f"  forecasts: {len(fc):,} rows  {fc.index[0].date()} -> "
          f"{fc.index[-1].date()}")
    print(f"  columns: {fc.columns.tolist()}")
    print("=" * 68)

    ac.check_structure(fc, rep)
    ac.check_ensemble(fc, list(cfg.ensemble_members), rep)
    ac.check_metrics_file(fc, results_dir / "metrics_5d.json", rep)
    ac.check_subperiods(fc, results_dir / "metrics_subperiod.json",
                        cfg.split_year, cfg.regime_quantile, rep)
    ac.check_dm(fc, results_dir / "dm_tests.json", cfg.horizon, rep)

    meta_path = results_dir / "fit_metadata.json"
    if meta_path.exists():
        ac.check_no_leakage(meta_path, rep)
    else:
        rep.section("LOOK-AHEAD")
        rep.ok("no fit_metadata.json (legacy artifacts predate it); skipping")

    ac.check_expected(fc, expected_path, rep)

    if args.regenerate:
        regenerate(cfg, results_dir, args.input, rep)

    rep.section("FINAL")
    if rep.failures:
        print(f"\n{len(rep.failures)} of {rep.checks} CHECK(S) FAILED")
        for f in rep.failures:
            print(f"  - {f}")
        return 1
    print(f"\nALL {rep.checks} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
