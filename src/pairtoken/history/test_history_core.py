from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import history_core as core
import run_history as runner


def test_contract_is_exact_retrospective_thirteen_fold_family() -> None:
    contract = core.load_contract()
    assert contract["claim_boundary"]["sample_role"] == "retrospective_development_robustness"
    assert contract["claim_boundary"]["not_confirmatory"] is True
    assert contract["data"]["confirmation_rows_permitted"] is False
    assert tuple(row["name"] for row in contract["folds"]) == core.EXPECTED_FOLDS
    assert len(contract["folds"]) == 13
    assert contract["model"]["declared_fit_count"] == 91
    assert contract["amendments"][0]["amendment_id"] == "A01_prefit_endpoint_history_floor"
    assert contract["amendments"][0]["fit_or_result_observed_before_amendment"] is False
    assert contract["amendments"][0]["result_driven"] is False
    assert contract["walk_forward"] == {
        "training_days": 365,
        "validation_days": 90,
        "target_purge_hours": 1,
        "refit_frequency": "quarterly",
        "scalers_fit_on_training_only": True,
        "positions_forced_flat_at_refit": False,
    }


def test_contract_rejects_confirmation_permission() -> None:
    contract = core.load_contract()
    contract["data"]["confirmation_rows_permitted"] = True
    with pytest.raises(ValueError, match="permits confirmation"):
        core.validate_contract(contract)


def test_first_fold_offset_preserves_exact_formation_and_feature_history() -> None:
    contract = core.load_contract()
    first = core.utc(contract["folds"][0]["test_start"])
    validation_end = first - pd.Timedelta(hours=1)
    validation_start = validation_end - pd.Timedelta(days=90)
    training_end = validation_start - pd.Timedelta(hours=1)
    training_start = training_end - pd.Timedelta(days=365)
    assert first == pd.Timestamp("2022-04-08T02:00:00Z")
    assert training_start == pd.Timestamp("2021-01-08T00:00:00Z")
    component_start = pd.Timestamp("2021-01-01T00:00:00Z")
    assert (training_start - component_start) / pd.Timedelta(hours=1) == 168
    assert validation_end - validation_start == pd.Timedelta(days=90)
    assert training_end - training_start == pd.Timedelta(days=365)


def test_clock_is_contiguous_and_final_hour_is_liquidation_not_action() -> None:
    contract = core.load_contract()
    clock = core.expected_decision_clock(contract)
    terminal = core.utc(contract["data"]["boundary_policy"]["terminal_zero_target"])
    assert clock[0] == pd.Timestamp("2022-04-08T02:00:00Z")
    assert clock[-1] == pd.Timestamp("2025-06-30T22:00:00Z")
    assert clock[-1] + pd.Timedelta(hours=1) == terminal
    assert terminal == pd.Timestamp("2025-06-30T23:00:00Z")
    assert terminal not in clock
    assert terminal < core.HARD_CUTOFF_EXCLUSIVE
    assert clock.equals(pd.date_range(clock[0], clock[-1], freq="1h"))


def test_preflight_partition_audit_enforces_endpoint_history_floor() -> None:
    contract = core.load_contract()
    clock = pd.date_range("2021-01-01T00:00:00Z", "2025-06-30T23:00:00Z", freq="1h")
    rows = runner._validate_fold_partitions(SimpleNamespace(clock=clock), contract)
    assert rows[0]["minimum_source_index"] == runner.frozen_core.ENDPOINT_HISTORY_HOURS
    assert rows[0]["training_start"] == "2021-01-08T00:00:00+00:00"

    broken = core.load_contract()
    broken["folds"][0]["test_start"] = "2022-04-01T02:00:00Z"
    # Bypass contract validation deliberately to exercise the materialization
    # floor itself rather than the separately frozen boundary check.
    with pytest.raises(ValueError, match="endpoint-history floor"):
        runner._validate_fold_partitions(SimpleNamespace(clock=clock), broken)


def test_frozen_parent_sources_are_hash_bound() -> None:
    contract = core.load_contract()
    observed = runner.validate_frozen_sources(contract)
    assert observed == contract["frozen_parent"]["source_sha256"]


def test_performance_metrics_compound_and_include_initial_peak() -> None:
    result = core.performance_metrics([0.10, -0.20, 0.05], turnover=2.0, gross_sum=0.03)
    expected_total = 1.10 * 0.80 * 1.05 - 1.0
    assert result["total_return"] == pytest.approx(expected_total)
    assert result["maximum_drawdown"] == pytest.approx(-0.20)
    assert result["linearized_break_even_cost_bps"] == pytest.approx(150.0)


def test_moving_block_bootstrap_is_deterministic_and_requires_continuity() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=30, freq="1D", tz="UTC"),
            "difference": np.linspace(-0.001, 0.002, 30),
        }
    )
    first = core.moving_block_bootstrap(frame, replications=200, block_days=5, seed=17)
    second = core.moving_block_bootstrap(frame, replications=200, block_days=5, seed=17)
    assert first == second
    assert first["observations"] == 30
    assert first["interpretation"] == "descriptive_after_model_selection_not_confirmatory"
    with pytest.raises(ValueError, match="continuous daily path"):
        core.moving_block_bootstrap(frame.drop(index=10), replications=20, block_days=5)


def _daily_fixture() -> pd.DataFrame:
    dates = pd.date_range("2022-04-01", periods=12, freq="1D", tz="UTC")
    rows: list[dict[str, object]] = []
    for series_id, shift in (("N02", 0.0002), ("C02", 0.0)):
        for position, date in enumerate(dates):
            rows.append(
                {
                    "series_id": series_id,
                    "cost_bps": 0.5,
                    "fold": "2022Q2" if position < 6 else "2022Q3",
                    "date": date,
                    "net_return": shift + (0.001 if position % 2 == 0 else -0.0005),
                    "gross_return": shift + (0.0011 if position % 2 == 0 else -0.0004),
                    "turnover": 0.4,
                }
            )
    return pd.DataFrame(rows)


def test_leave_one_fold_out_and_stability_cover_each_series_fold() -> None:
    daily = _daily_fixture()
    leave = core.leave_one_fold_out_metrics(daily)
    assert len(leave) == 4
    quarterly_rows: list[dict[str, object]] = []
    for (series, fold), group in daily.groupby(["series_id", "fold"]):
        quarterly_rows.append(
            {
                "series_id": series,
                "cost_bps": 0.5,
                "fold": fold,
                **core.performance_metrics(
                    group["net_return"],
                    turnover=float(group["turnover"].sum()),
                    gross_sum=float(group["gross_return"].sum()),
                ),
            }
        )
    stability = core.stability_summary(pd.DataFrame(quarterly_rows), leave)
    assert set(stability["series_id"]) == {"N02", "C02"}
    assert (stability["quarters"] == 2).all()
    assert stability["minimum_leave_one_quarter_out_sharpe"].notna().all()


def test_coin_contribution_concentration_uses_absolute_shares() -> None:
    coin = pd.DataFrame(
        {
            "series_id": ["N02", "N02", "N02"],
            "cost_bps": [0.5, 0.5, 0.5],
            "coin": ["A", "B", "C"],
            "net_return_contribution": [0.03, -0.01, 0.0],
        }
    )
    result = core.contribution_concentration(coin).iloc[0]
    assert result["largest_absolute_contribution_coin"] == "A"
    assert result["largest_absolute_contribution_fraction"] == pytest.approx(0.75)
    assert result["absolute_contribution_hhi"] == pytest.approx(0.75**2 + 0.25**2)


def test_terminal_attribution_stays_on_same_day() -> None:
    clock = pd.DatetimeIndex(
        [pd.Timestamp("2025-06-30T22:00:00Z"), pd.Timestamp("2025-06-30T23:00:00Z")],
        name="timestamp",
    )
    bars = pd.DataFrame(
        {
            "price_return": [0.01, 0.0],
            "funding_return": [0.0, -0.001],
            "transaction_cost": [0.0, 0.0002],
            "net_return": [0.01, -0.0012],
            "gross_return": [0.01, -0.001],
            "turnover": [0.4, 0.4],
        },
        index=clock,
    )
    coin_rows = []
    for timestamp, price, funding, cost in zip(clock, (0.01, 0.0), (0.0, -0.001), (0.0, 0.0002), strict=True):
        coin_rows.append(
            {
                "timestamp": timestamp,
                "coin": "BTCUSDT",
                "price_return": price,
                "funding_return": funding,
                "transaction_cost": cost,
            }
        )
    ledger = SimpleNamespace(bars=bars, coin_ledger=pd.DataFrame(coin_rows).set_index(["timestamp", "coin"]))
    daily, daily_coin = runner._continuous_daily_attribution(ledger)
    assert list(daily.index) == [pd.Timestamp("2025-06-30T00:00:00Z")]
    assert pd.DatetimeIndex(daily_coin["date"]).unique().tolist() == [pd.Timestamp("2025-06-30T00:00:00Z")]
    expected = (1.0 + 0.01) * (1.0 - 0.0012) - 1.0
    assert daily.iloc[0]["net_return"] == pytest.approx(expected)
