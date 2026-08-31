from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import portfolio_evaluation as evaluation


def dependencies() -> evaluation.AuditedDependencies:
    return evaluation.install_audited_dependencies(evaluation.DEFAULT_AUDITED_ROOT)


def synthetic_economic() -> evaluation.EconomicInputs:
    clock = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    symbols = ("BTCUSDT", "ETHUSDT")
    pair_ids = ("ETHUSDT__BTCUSDT",)
    legs = np.asarray([[1, 0]], dtype=np.int64)
    weights = np.empty((len(clock), 1, 2), dtype=float)
    weights[:, 0, 0] = 0.60
    weights[:, 0, 1] = -0.40
    returns = np.zeros((len(clock), 2), dtype=float)
    returns[:, 0] = np.linspace(-0.001, 0.001, len(clock))
    returns[:, 1] = np.linspace(0.002, -0.001, len(clock))
    funding_boundary = np.zeros_like(returns)
    funding_next = np.zeros_like(returns)
    # The action at row five carries into a nonzero funding boundary at row six.
    funding_next[5] = [0.0001, 0.0002]
    funding_boundary[6] = funding_next[5]
    decision = np.ones_like(returns, dtype=bool)
    target = np.ones_like(returns, dtype=bool)
    terminal = np.zeros_like(returns, dtype=bool)
    terminal[20, 1] = True
    # ETH is unavailable after its proven terminal mark; its boundary funding
    # is zero because the position closed at the in-hour terminal price.
    decision[21:, 1] = False
    target[21:, 1] = False
    returns[21:, 1] = np.nan
    funding_boundary[21:, 1] = 0.0
    funding_next[20:, 1] = 0.0
    growth = 1.0 + returns
    pair_decision = decision[:, 0:1] & decision[:, 1:2]
    pair_target = target[:, 0:1] & target[:, 1:2]
    cashflow = np.zeros((len(clock), 1), dtype=float)
    price = weights[:, 0, 0] * returns[:, 1] + weights[:, 0, 1] * returns[:, 0]
    next_funding = -(
        weights[:, 0, 0] * growth[:, 1] * funding_next[:, 1]
        + weights[:, 0, 1] * growth[:, 0] * funding_next[:, 0]
    )
    cashflow[:, 0] = np.where(pair_target[:, 0], price + next_funding, 0.0)
    opens = np.full_like(returns, 100.0)
    contract = evaluation.load_contract()
    acquisition_hashes = {
        field: contract["data"][field]
        for field in (
            "acquisition_manifest_sha256",
            "archive_availability_sha256",
            "artifact_hashes_sha256",
        )
    }
    return evaluation.EconomicInputs(
        clock=clock,
        symbols=symbols,
        pair_ids=pair_ids,
        pair_leg_indices=legs,
        pair_weights=weights,
        coin_open_prices=opens,
        coin_open_to_open_returns=returns,
        funding_at_boundary=funding_boundary,
        funding_at_next_boundary=funding_next,
        next_boundary_growth=growth,
        coin_decision_available=decision,
        coin_target_available=target,
        terminal_exit_mask=terminal,
        pair_decision_available=pair_decision,
        pair_target_available=pair_target,
        pair_target_cashflows=cashflow,
        metadata={
            "schema": "paired_hourly_economic_inputs_v2",
            "hard_cutoff_exclusive": evaluation.DEVELOPMENT_END_EXCLUSIVE.isoformat(),
            "pair_cashflow_formula": "sum_l w[t,l]*r[t,l] - sum_l w[t,l]*(1+r[t,l])*funding[t+1,l]",
            "symbols": list(symbols),
            "pair_ids": list(pair_ids),
            "transient_open_gap_report": {
                "policy": "no_fill_no_shortening_mask_outcome_only",
                "no_fill": True,
                "no_shortening": True,
                "feature_and_decision_availability_unchanged": True,
                "coin_gap_cells": 0,
                "pair_label_cells_masked": 0,
                "oos_coin_gap_cells": 0,
                "oos_pair_label_cells": 0,
                "events": [],
            },
            "provenance": {
                "contract_sha256": evaluation.sha256_file(evaluation.CONTRACT_PATH),
                **acquisition_hashes,
                "audited_source_hashes": evaluation.AUDITED_SOURCE_HASHES,
                "builder_sha256": evaluation.sha256_file(Path(evaluation.__file__)),
            },
        },
    )


def write_manifest(path: Path) -> Path:
    economic = synthetic_economic()
    contract = evaluation.load_contract()
    acquisition_hashes = {
        field: contract["data"][field]
        for field in (
            "acquisition_manifest_sha256",
            "archive_availability_sha256",
            "artifact_hashes_sha256",
        )
    }
    one_period = evaluation.assert_one_period_ledger_identity(economic, dependencies())
    multi_period = evaluation.assert_multi_period_ledger_identity(
        economic, dependencies()
    )
    manifest = path.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "schema": "paired_hourly_economic_inputs_manifest_v2",
                "status": "complete_immutable_shared_training_and_ledger_input",
                "artifact_path": str(path.resolve()),
                "artifact_sha256": evaluation.sha256_file(path),
                "contract_sha256": evaluation.sha256_file(evaluation.CONTRACT_PATH),
                "clock_start": economic.clock[0].isoformat(),
                "clock_end": economic.clock[-1].isoformat(),
                "hourly_observations": len(economic.clock),
                "symbols": list(economic.symbols),
                "pair_ids_sha256": hashlib.sha256(
                    "\n".join(economic.pair_ids).encode()
                ).hexdigest(),
                "pairs": len(economic.pair_ids),
                "pair_decision_available_cells": int(
                    economic.pair_decision_available.sum()
                ),
                "pair_target_available_cells": int(
                    economic.pair_target_available.sum()
                ),
                "terminal_exit_cells": int(economic.terminal_exit_mask.sum()),
                "funding_verified_archives_loaded": 1,
                "funding_missing_archives": 0,
                "funding_included_in_target": True,
                "pair_target_formula": economic.metadata["pair_cashflow_formula"],
                "one_period_ledger_identity": one_period,
                "multi_period_ledger_identity": multi_period,
                "transient_open_gap_report": economic.metadata[
                    "transient_open_gap_report"
                ],
                "audited_source_hashes": evaluation.AUDITED_SOURCE_HASHES,
                **acquisition_hashes,
                "latest_raw_month": "2025-06",
                "july_or_later_path_count": 0,
                "confirmation_rows_parsed": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    path.with_suffix(".sha256").write_text(
        f"{evaluation.sha256_file(path)}  {path.name}\n", encoding="ascii"
    )
    return manifest


def use_zero_gap_test_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = evaluation.load_contract()
    contract["data"]["transient_open_gap_policy"] = {
        "rule": "no_fill_no_shortening_mask_outcome_only",
        "expected_coin_gap_cells": 0,
        "expected_pair_label_cells_masked": 0,
        "expected_events": [],
        "maximum_oos_coin_gap_cells": 0,
        "maximum_oos_pair_label_cells": 0,
    }
    monkeypatch.setattr(evaluation, "load_contract", lambda: contract)


def test_nonzero_funding_sign_growth_and_terminal_identity() -> None:
    economic = synthetic_economic()
    report = evaluation.assert_one_period_ledger_identity(economic, dependencies())
    assert report["ordinary_with_next_boundary_funding"]["passed"] is True
    assert report["ordinary_without_funding_event"]["passed"] is True
    assert report["terminal_delisting_mark"]["passed"] is True
    # Positive funding is paid by the long ETH leg and partially received by
    # the short BTC hedge, on notionals grown by (1+r).
    row = 5
    expected_funding = -(
        0.60 * (1.0 + economic.coin_open_to_open_returns[row, 1]) * 0.0002
        - 0.40 * (1.0 + economic.coin_open_to_open_returns[row, 0]) * 0.0001
    )
    price = (
        0.60 * economic.coin_open_to_open_returns[row, 1]
        - 0.40 * economic.coin_open_to_open_returns[row, 0]
    )
    assert economic.pair_target_cashflows[row, 0] == pytest.approx(
        price + expected_funding, abs=1e-15
    )


def test_multi_period_identity_includes_terminal_zero_row() -> None:
    report = evaluation.assert_multi_period_ledger_identity(
        synthetic_economic(), dependencies(), decision_count=12
    )
    assert report["passed"] is True
    assert report["terminal_zero_target_included"] is True
    assert report["contains_nonzero_next_boundary_funding"] is True
    assert report["absolute_error"] < 1e-13


def synthetic_market(
    *, transient_gap: bool, start: str = "2024-01-01"
) -> evaluation.MarketInputs:
    clock = pd.date_range(start, periods=900, freq="1h", tz="UTC")
    symbols = ["BTCUSDT", "ETHUSDT"]
    step = np.arange(len(clock), dtype=float)
    opens = pd.DataFrame(
        {
            "BTCUSDT": 100.0 * np.exp(0.00010 * step + 0.0020 * np.sin(step / 17.0)),
            "ETHUSDT": 80.0 * np.exp(0.00013 * step + 0.0025 * np.cos(step / 13.0)),
        },
        index=clock,
    )
    closes = opens * 1.0001
    if transient_gap:
        opens.loc[clock[400], "ETHUSDT"] = np.nan
        closes.loc[clock[400], "ETHUSDT"] = np.nan
    else:
        opens.loc[clock[700] :, "ETHUSDT"] = np.nan
        closes.loc[clock[700] :, "ETHUSDT"] = np.nan
    previous = pd.DataFrame(True, index=clock, columns=symbols)
    funding = pd.DataFrame(
        columns=symbols, index=pd.DatetimeIndex([], tz="UTC"), dtype=float
    )
    return evaluation.MarketInputs(
        open_prices=opens,
        last_closes=closes,
        previous_minute_available=previous,
        funding_rates=funding,
        manifest_rows=pd.DataFrame(),
        loaded_months=("2024-01", "2024-02"),
        funding_archive_count=4,
        missing_funding_archive_count=0,
    )


def tiny_contract(
    *, expected_events: list[dict] | None = None, expected_coin_gap_cells: int = 0
) -> dict:
    contract = evaluation.load_contract()
    contract["target"]["hedge_estimation"] = {
        "lookback_hours": 48,
        "minimum_complete_hours": 24,
        "one_full_hour_lag": True,
        "beta_clip": [-3.0, 3.0],
        "pair_gross_normalization": 1.0,
    }
    contract["data"]["transient_open_gap_policy"] = {
        "rule": "no_fill_no_shortening_mask_outcome_only",
        "expected_coin_gap_cells": expected_coin_gap_cells,
        "expected_pair_label_cells_masked": expected_coin_gap_cells,
        "expected_events": [] if expected_events is None else expected_events,
        "maximum_oos_coin_gap_cells": 0,
        "maximum_oos_pair_label_cells": 0,
    }
    return contract


def test_transient_gap_masks_only_outcome_without_fill_or_shortening() -> None:
    market = synthetic_market(transient_gap=True)
    decision = market.open_prices.index[399]
    missing = market.open_prices.index[400]
    expected_events = [
        {
            "decision_timestamp": decision.isoformat(),
            "missing_next_open_timestamp": missing.isoformat(),
            "symbols": ["ETHUSDT"],
        }
    ]
    economic = evaluation.build_economic_inputs(
        market=market,
        symbols=("BTCUSDT", "ETHUSDT"),
        pair_ids=("ETHUSDT__BTCUSDT",),
        pair_leg_indices=np.asarray([[1, 0]]),
        contract=tiny_contract(
            expected_events=expected_events, expected_coin_gap_cells=1
        ),
        provenance={},
    )
    report = economic.metadata["transient_open_gap_report"]
    assert report["coin_gap_cells"] == 1
    assert report["pair_label_cells_masked"] == 1
    assert report["oos_coin_gap_cells"] == 0
    assert report["events"][0]["coin_gap_cells"] == 1
    assert report["events"][0]["pair_label_cells_masked"] == 1
    # At t the causal decision/feature mask remains available. Only ETH's
    # unmeasurable t->t+1 outcome and the pair label are unavailable.
    assert economic.coin_decision_available[399].tolist() == [True, True]
    assert economic.coin_target_available[399].tolist() == [True, False]
    assert economic.pair_decision_available[399, 0]
    assert not economic.pair_target_available[399, 0]
    assert np.isnan(economic.coin_open_to_open_returns[399, 1])
    assert not economic.terminal_exit_mask[399, 1]
    # Adjacent fully observed labels remain intact.
    assert economic.pair_target_available[398, 0]
    assert economic.pair_target_available[401, 0]
    deps = dependencies()
    for row in (397, 401):
        test_clock = economic.clock.take([row, row + 1])
        pair_targets = pd.DataFrame(
            [1.0, 0.0], index=test_clock, columns=economic.pair_ids
        )
        hedge = np.zeros((2, 2, 1), dtype=float)
        hedge[:, 1, 0] = economic.pair_weights[row, 0, 0]
        hedge[:, 0, 0] = economic.pair_weights[row, 0, 1]
        ledger = deps.backtest_multicoin_pair_ledger(
            pair_targets,
            hedge,
            pd.DataFrame(
                economic.coin_open_to_open_returns[[row, row + 1]],
                index=test_clock,
                columns=economic.symbols,
            ),
            pd.DataFrame(
                [
                    economic.funding_at_boundary[row],
                    economic.funding_at_next_boundary[row],
                ],
                index=test_clock,
                columns=economic.symbols,
            ),
            transaction_cost_bps=0.0,
            coin_symbols=economic.symbols,
            availability=pd.DataFrame(
                economic.coin_decision_available[[row, row + 1]],
                index=test_clock,
                columns=economic.symbols,
            ),
            constraints=deps.PortfolioConstraints(
                maximum_total_gross=2.0,
                maximum_absolute_coin_weight=2.0,
                maximum_absolute_dollar_net=2.0,
            ),
            liquidate_at_end=True,
        )
        observed = float((1.0 + ledger.bars["gross_return"]).prod() - 1.0)
        assert observed == pytest.approx(
            economic.pair_target_cashflows[row, 0], abs=1e-14
        )


def test_transient_gap_in_declared_oos_clock_fails_closed() -> None:
    market = synthetic_market(transient_gap=True, start="2024-03-20")
    decision = market.open_prices.index[399]
    missing = market.open_prices.index[400]
    contract = tiny_contract(
        expected_coin_gap_cells=1,
        expected_events=[
            {
                "decision_timestamp": decision.isoformat(),
                "missing_next_open_timestamp": missing.isoformat(),
                "symbols": ["ETHUSDT"],
            }
        ],
    )
    with pytest.raises(ValueError, match="test-fold clock"):
        evaluation.build_economic_inputs(
            market=market,
            symbols=("BTCUSDT", "ETHUSDT"),
            pair_ids=("ETHUSDT__BTCUSDT",),
            pair_leg_indices=np.asarray([[1, 0]]),
            contract=contract,
            provenance={},
        )


def test_true_final_source_open_is_the_only_terminal_mark() -> None:
    economic = evaluation.build_economic_inputs(
        market=synthetic_market(transient_gap=False),
        symbols=("BTCUSDT", "ETHUSDT"),
        pair_ids=("ETHUSDT__BTCUSDT",),
        pair_leg_indices=np.asarray([[1, 0]]),
        contract=tiny_contract(),
        provenance={},
    )
    locations = np.argwhere(economic.terminal_exit_mask)
    assert locations.tolist() == [[699, 1]]
    assert economic.coin_target_available[699, 1]
    assert not economic.coin_decision_available[700, 1]


def test_artifact_is_create_only_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_zero_gap_test_contract(monkeypatch)
    path = tmp_path / "hourly_economic_inputs.npz"
    evaluation.save_economic_inputs(path, synthetic_economic())
    write_manifest(path)
    loaded = evaluation.load_economic_inputs(path)
    assert loaded.metadata["schema"] == "paired_hourly_economic_inputs_v2"
    with pytest.raises(FileExistsError):
        evaluation.save_economic_inputs(path, synthetic_economic())
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="differs from its manifest"):
        evaluation.load_economic_inputs(path)


def test_manifest_contract_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_zero_gap_test_contract(monkeypatch)
    path = tmp_path / "hourly_economic_inputs.npz"
    evaluation.save_economic_inputs(path, synthetic_economic())
    manifest = write_manifest(path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["contract_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="paired contract"):
        evaluation.load_economic_inputs(path)


def test_exact_five_by_six_by_four_family_is_frozen() -> None:
    contract = evaluation.load_contract()
    contexts = {
        (series_id, fold["name"], float(cost))
        for series_id in evaluation.SERIES
        for fold in contract["folds"]
        for cost in contract["portfolio"]["cost_bps"]
    }
    assert tuple(evaluation.SERIES) == ("B00", "N01", "C01", "N02", "C02")
    assert len(contract["folds"]) == 6
    assert len(contexts) == 120


def test_forecast_diagnostics_require_every_content_identity_field() -> None:
    digest = "a" * 64
    run_manifest = {
        "contract_sha256": digest,
        "component_cache_sha256": digest,
        "pair_axis_sha256": digest,
        "source_hashes": {
            "paired_neural_core.py": digest,
            "paired_neural_models.py": digest,
            "run_paired_neural.py": digest,
        },
        "source_bundle_sha256": digest,
        "runtime_environment_sha256": digest,
        "economic_artifact_sha256": digest,
        "economic_manifest_sha256": digest,
    }
    identity = evaluation._expected_forecast_identity(
        run_manifest=run_manifest,
        fold="2023Q2",
        representation="native_1m",
        model="lstm_seed_mean",
        seed="all3",
        role="fixed_three_seed_arithmetic_mean",
    )
    extras = {
        "seeds": list(evaluation.FIXED_SEEDS),
        "aggregation": "arithmetic_mean_never_best_seed",
    }
    diagnostics = {**identity, **extras}
    evaluation._validate_forecast_diagnostics(
        diagnostics,
        expected_identity=identity,
        expected_extras=extras,
        source="synthetic.npz",
    )
    assert set(identity) == evaluation.FORECAST_DIAGNOSTIC_IDENTITY_KEYS
    for field in evaluation.FORECAST_DIAGNOSTIC_IDENTITY_KEYS:
        swapped = dict(diagnostics)
        swapped[field] = "swapped"
        with pytest.raises(ValueError, match=field):
            evaluation._validate_forecast_diagnostics(
                swapped,
                expected_identity=identity,
                expected_extras=extras,
                source="synthetic.npz",
            )
    with pytest.raises(ValueError, match="schema"):
        evaluation._validate_forecast_diagnostics(
            {**diagnostics, "unexpected": True},
            expected_identity=identity,
            expected_extras=extras,
            source="synthetic.npz",
        )


def bootstrap_frame() -> pd.DataFrame:
    rows = []
    for year in (2023, 2024):
        dates = pd.date_range(f"{year}-04-01", f"{year}-12-31", freq="1D", tz="UTC")
        for date in dates:
            quarter = (date.month - 1) // 3 + 1
            rows.append(
                {"fold": f"{year}Q{quarter}", "date": date, "difference": 0.0001}
            )
    return pd.DataFrame(rows)


def test_paired_bootstrap_requires_exact_declared_calendar() -> None:
    frame = bootstrap_frame()
    result = evaluation.moving_block_bootstrap(
        frame, replications=100, block_days=7, seed=7
    )
    assert result["observations"] == 550
    assert result["observed_mean_daily_difference"] == pytest.approx(0.0001)
    with pytest.raises(ValueError, match="exact Q2-Q4 calendar"):
        evaluation.moving_block_bootstrap(
            frame.iloc[:-1], replications=10, block_days=7, seed=7
        )


def test_terminal_boundary_is_compounded_into_preceding_day() -> None:
    deps = dependencies()
    clock = pd.DatetimeIndex(
        [pd.Timestamp("2024-06-30T23:00:00Z"), pd.Timestamp("2024-07-01T00:00:00Z")]
    )
    pair_targets = pd.DataFrame([0.5, 0.0], index=clock, columns=["ETHUSDT__BTCUSDT"])
    hedge = np.zeros((2, 2, 1), dtype=float)
    hedge[:, 1, 0] = 0.6
    hedge[:, 0, 0] = -0.4
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.0, 0.0]], index=clock, columns=["BTCUSDT", "ETHUSDT"]
    )
    funding = pd.DataFrame(
        [[0.0, 0.0], [0.0001, 0.0002]], index=clock, columns=returns.columns
    )
    available = pd.DataFrame(True, index=clock, columns=returns.columns)
    ledger = deps.backtest_multicoin_pair_ledger(
        pair_targets,
        hedge,
        returns,
        funding,
        transaction_cost_bps=0.5,
        coin_symbols=tuple(returns.columns),
        availability=available,
        constraints=deps.PortfolioConstraints(
            maximum_total_gross=2.0,
            maximum_absolute_coin_weight=2.0,
            maximum_absolute_dollar_net=2.0,
        ),
        liquidate_at_end=False,
    )
    daily, _coin, keys = evaluation.economic_daily_attribution(
        ledger, terminal_timestamp=clock[-1]
    )
    assert list(daily.index) == [pd.Timestamp("2024-06-30T00:00:00Z")]
    assert keys[0] == keys[1]
    assert daily.iloc[0]["turnover"] > 0.0
    assert daily.iloc[0]["funding_return_contribution"] != 0.0


def test_cli_routes_prepare_and_evaluate() -> None:
    prepare = evaluation.arguments(
        [
            "prepare-economic",
            "--cache-root",
            "/tmp/cache",
            "--acquisition-manifest",
            "/tmp/manifest",
            "--output",
            "/tmp/economic.npz",
        ]
    )
    assert prepare.function is evaluation.prepare_economic
    evaluate = evaluation.arguments(
        [
            "evaluate",
            "--components",
            "/tmp/components.npz",
            "--economic-inputs",
            "/tmp/economic.npz",
            "--forecast-output",
            "/tmp/forecast",
            "--output",
            "/tmp/portfolio",
        ]
    )
    assert evaluate.function is evaluation.evaluate
    assert evaluate.economic_manifest is None


def test_default_audited_root_has_expected_path_shape() -> None:
    root = evaluation.DEFAULT_AUDITED_ROOT
    assert (root / "src/pairs_research/crypto_multicoin_execution.py").is_file()
    deps = dependencies()
    assert deps.source_hashes == evaluation.AUDITED_SOURCE_HASHES
