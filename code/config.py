"""
Market configuration registry for the multi-market volatility horse race.

The whole point of this module is that the United States and Brazil run through
*identical* model code. If the two markets were forked into separate scripts,
any difference in the results would be confounded with differences in the
implementations, and the comparison would be worthless. So everything that
differs between markets lives here as data, and nothing differs in the models.

Three markets are registered:

  us       ^GSPC + ^VIX, 2004-2025, 30 features. The original study, re-run
           through the corrected pipeline so that it is a like-for-like
           baseline for Brazil rather than a comparison against published
           numbers produced by different (buggy) code.

  br_long  ^BVSP, 2004-2025, no implied-vol features. Arm A. Buys the full
           Brazilian regime sequence -- 2008, the 2014-16 recession, Joesley
           Day in 2017, COVID, the 2022 election -- at the cost of the four
           implied-vol features.

  br_iv    ^BVSP + IVol-BR, ~2012-2025, 30 features. Arm B. Buys feature
           parity with the US study at the cost of training history, because
           no Brazilian implied-volatility index exists before ~2011.

Arm A and Arm B deliberately share the same validation window and the same
test window, so their QLIKE numbers are computed over identical evaluation
days. The only things that vary between them are the training start and the
presence of the implied-vol features, which is what makes the A-vs-B contrast
a clean measurement of how much of the result depends on having implied vol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Prefixes of columns that are eligible to become model features. A column is
# a feature if it starts with one of these and is not a forecast target.
# Both 03_run_core_models.py and 04_subperiod_and_importance.py must derive
# their feature list from feature_columns() below -- in the original code each
# script carried its own copy of this filter, which is a latent source of
# silent divergence between the forecasts and the feature importances.
_BASE_FEATURE_PREFIXES = (
    "rv_5d_lag",
    "abs_ret_lag",
    "har_",
    "neg_return",
    "pos_return",
    "range_vol",
    "dow_",
    "log_return",
)

# Implied-volatility feature prefixes. New panels are built with `iv_*`
# columns because calling an IVol-BR series "vix" would be actively
# misleading in the Brazilian output files. `vix_` is still recognised so
# that the legacy US panel committed at the repository root -- which uses the
# old names -- keeps working for the regression test.
_IV_FEATURE_PREFIXES = ("iv_", "vix_")

_VOLUME_FEATURE_PREFIXES = ("volume",)


@dataclass(frozen=True)
class MarketConfig:
    """Everything that differs between markets. Models read this; they never
    branch on the market name themselves."""

    name: str
    equity_ticker: str
    equity_label: str

    # Implied volatility. `iv_spec` is either None, "yfinance:<TICKER>", or
    # "csv:<filename under data/<market>/>".
    iv_spec: str | None
    iv_label: str | None

    start: str
    end: str
    train_end: str
    val_end: str
    test_start: str

    use_iv: bool
    # None means "decide from the data" -- see the volume gate in
    # 01_collect_data.py. True/False force the choice.
    use_volume: bool | None

    notes: str = ""

    # --- Shared across every market. Held here rather than as module-level
    # constants so a market *could* override them, and so the values are
    # visible in one place when reading results.
    #
    # 252 is deliberate for Brazil as well as the US. It looks like a US
    # convention that a Brazilian replication ought to change, and it is not:
    # 252 business days ("dias uteis") is the standard Brazilian
    # annualisation convention, used throughout the BRL rates market. B3
    # trades roughly 246-250 days a year once Carnival and Corpus Christi are
    # taken out, exactly as the NYSE falls short of 252.
    annualization: int = 252

    horizon: int = 5  # forecast horizon in trading days; target is rv_5d_fwd
    garch_refit: int = 44  # GARCH refit cadence (bimonthly)
    model_refit: int = 66  # HAR / tree refit cadence (quarterly)

    # Walk-forward validation block used for early stopping, in trading days.
    # See fit_trees() -- the original code validated on a window nested inside
    # its own training set, so early stopping never fired.
    val_window: int = 252

    # Gap held out between training and validation, and between validation and
    # test, in trading days. The target rv_5d_fwd at time t is built from
    # returns at t+1..t+5, so without an embargo the last `horizon` training
    # rows peek into the validation block.
    embargo: int = 5

    rolling_windows: tuple[int, ...] = (5, 22, 66)
    feature_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 22)
    target_horizons: tuple[int, ...] = (1, 5, 22)

    # Subperiod definitions for the robustness tables. Identical across
    # markets so the splits stay comparable.
    split_year: int = 2022
    regime_quantile: float = 75.0

    models: tuple[str, ...] = (
        "GARCH",
        "EGARCH",
        "GJR-GARCH",
        "HAR-RV",
        "LightGBM",
        "XGBoost",
        "Ensemble",
    )
    ensemble_members: tuple[str, ...] = ("LightGBM", "HAR-RV", "GARCH")

    # Seeds. The original code set none and relied on library defaults. That
    # happened to be deterministic, but it is undefended against a default
    # changing underneath us.
    seed: int = 42

    @property
    def target(self) -> str:
        return f"rv_{self.horizon}d_fwd"

    @property
    def data_dir(self) -> Path:
        return REPO_ROOT / "data" / self.name

    @property
    def results_dir(self) -> Path:
        return REPO_ROOT / "results" / self.name

    @property
    def iv_kind(self) -> str | None:
        """'yfinance', 'csv', or None."""
        if not self.iv_spec:
            return None
        return self.iv_spec.split(":", 1)[0]

    @property
    def iv_ref(self) -> str | None:
        """Ticker or filename, depending on iv_kind."""
        if not self.iv_spec:
            return None
        return self.iv_spec.split(":", 1)[1]

    def feature_columns(self, df, use_volume: bool | None = None) -> list[str]:
        """The model feature list for this market, derived from the panel.

        Single source of truth -- every script that needs features calls this.
        """
        prefixes = list(_BASE_FEATURE_PREFIXES)
        if self.use_iv:
            prefixes.extend(_IV_FEATURE_PREFIXES)

        vol = self.use_volume if use_volume is None else use_volume
        if vol:
            prefixes.extend(_VOLUME_FEATURE_PREFIXES)

        return [
            c
            for c in df.columns
            if c.startswith(tuple(prefixes)) and "_fwd" not in c
        ]


MARKETS: dict[str, MarketConfig] = {
    "us": MarketConfig(
        name="us",
        equity_ticker="^GSPC",
        equity_label="S&P 500",
        iv_spec="yfinance:^VIX",
        iv_label="CBOE VIX",
        start="2004-01-01",
        end="2025-12-31",
        train_end="2018-12-31",
        val_end="2021-12-31",
        test_start="2022-01-01",
        use_iv=True,
        use_volume=True,
        notes=(
            "The original study, re-run through the corrected pipeline. Not "
            "expected to match the published numbers: the published run "
            "validated on a window nested inside its training set, and scored "
            "an in-sample 1-day GARCH filter against a 5-day forward target."
        ),
    ),
    # -------------------------------------------------------------------
    # Brazilian arms. Dates below are PROVISIONAL until 00_probe_sources.py
    # has been run on a machine with network access -- see the step-0 notes
    # in data/br_long/PROVENANCE.md and data/br_iv/PROVENANCE.md.
    # -------------------------------------------------------------------
    "br_long": MarketConfig(
        name="br_long",
        equity_ticker="^BVSP",
        equity_label="Ibovespa",
        iv_spec=None,
        iv_label=None,
        start="2004-01-01",
        end="2025-12-31",
        train_end="2018-12-31",
        val_end="2021-12-31",
        test_start="2022-01-01",
        use_iv=False,
        use_volume=None,  # gated on data quality; ^BVSP volume is unreliable
        notes=(
            "Arm A. Full Brazilian history, no implied-vol features. Keeps "
            "2008, the 2014-16 recession and Joesley Day in the training set."
        ),
    ),
    "br_iv": MarketConfig(
        name="br_iv",
        equity_ticker="^BVSP",
        equity_label="Ibovespa",
        iv_spec="csv:ivolbr_raw.csv",
        iv_label="IVol-BR (FGV EESP)",
        start="2012-01-01",  # PROVISIONAL: IVol-BR is expected to start ~Aug 2011
        end="2025-12-31",
        train_end="2018-12-31",
        val_end="2021-12-31",
        test_start="2022-01-01",
        use_iv=True,
        use_volume=None,
        notes=(
            "Arm B. Feature parity with the US study. Shares br_long's "
            "validation and test windows exactly, so the A-vs-B QLIKE "
            "comparison is computed over identical evaluation days."
        ),
    ),
}


def get(name: str) -> MarketConfig:
    if name not in MARKETS:
        raise SystemExit(
            f"unknown market {name!r}; choose one of {', '.join(MARKETS)}"
        )
    return MARKETS[name]


def add_market_arg(parser):
    """Shared --market flag so every script spells it the same way."""
    parser.add_argument(
        "--market",
        default="us",
        choices=sorted(MARKETS),
        help="which market configuration to run (default: us)",
    )
    return parser
