from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paired_neural_core as core  # noqa: E402
import paired_neural_models as models  # noqa: E402
import run_paired_neural as runner  # noqa: E402


WORKSPACE = HERE.parents[2]
SHORT_MANIFEST = (
    WORKSPACE / "artifacts/acquisition/top24_1m_dev_202309_202506_v1"
)
REMOTE_CACHE = WORKSPACE / "data/binance_vision_top24_1m_dev"
REMOTE_MANIFEST = (
    WORKSPACE / "artifacts/acquisition/top24_1m_dev_202101_202506_v1"
)
if REMOTE_CACHE.is_dir():
    INTEGRATION_CACHE = REMOTE_CACHE
    INTEGRATION_MANIFEST = REMOTE_MANIFEST
else:
    INTEGRATION_CACHE = WORKSPACE / "data/binance_vision_top24_202101_202606"
    INTEGRATION_MANIFEST = SHORT_MANIFEST


def synthetic_closes(*, hours: int = 900, seed: int = 7) -> pd.DataFrame:
    index = pd.date_range(
        "2023-01-01T00:00:00Z", periods=hours * 60, freq="1min", tz="UTC"
    )
    rng = np.random.default_rng(seed)
    btc = rng.normal(0.0, 8e-5, len(index))
    eth = 0.7 * btc + rng.normal(0.0, 9e-5, len(index))
    bnb = -0.2 * btc + rng.normal(0.0, 1.1e-4, len(index))
    returns = np.column_stack((btc, eth, bnb))
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(
        close, index=index, columns=("BTCUSDT", "ETHUSDT", "BNBUSDT")
    )


def synthetic_economic_inputs(*, hours: int = 900) -> core.HourlyEconomicInputs:
    clock = pd.date_range(
        "2023-01-01T00:00:00Z", periods=hours, freq="1h", tz="UTC"
    )
    symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
    legs = np.asarray(((1, 0), (2, 0), (2, 1)), dtype=np.int64)
    pair_ids = tuple(f"{symbols[a]}__{symbols[b]}" for a, b in legs)
    phase = np.arange(hours, dtype=np.float64)
    signal_weight = 0.35 + 0.1 * np.sin(phase / 17.0)
    weights = np.empty((hours, len(pair_ids), 2), dtype=np.float64)
    for pair in range(len(pair_ids)):
        shifted = np.roll(signal_weight, pair * 3)
        weights[:, pair, 0] = shifted
        weights[:, pair, 1] = -(1.0 - shifted)
    rng = np.random.default_rng(712)
    returns = rng.normal(0.0, 0.002, size=(hours, len(symbols)))
    growth = 1.0 + returns
    funding_next = np.zeros_like(returns)
    funding_next[7::8, :] = np.asarray((1e-4, -7e-5, 5e-5))
    funding_boundary = np.zeros_like(returns)
    funding_boundary[1:] = funding_next[:-1]
    decision = np.ones_like(returns, dtype=bool)
    target_available_coin = np.ones_like(returns, dtype=bool)
    pair_decision = np.ones((hours, len(pair_ids)), dtype=bool)
    pair_target = np.ones_like(pair_decision)
    targets = np.zeros_like(pair_target, dtype=np.float64)
    for pair, pair_legs in enumerate(legs):
        targets[:, pair] = np.sum(
            weights[:, pair] * (growth[:, pair_legs] - 1.0)
            - weights[:, pair] * growth[:, pair_legs] * funding_next[:, pair_legs],
            axis=1,
        )
    transient_report = {
        "policy": "no_fill_no_shortening_mask_outcome_only",
        "no_fill": True,
        "no_shortening": True,
        "feature_and_decision_availability_unchanged": True,
        "coin_gap_cells": 0,
        "pair_label_cells_masked": 0,
        "oos_coin_gap_cells": 0,
        "oos_pair_label_cells": 0,
        "events": [],
    }
    metadata = {
        "schema": "paired_hourly_economic_inputs_v2",
        "clock_start": clock[0].isoformat(),
        "clock_end": clock[-1].isoformat(),
        "hard_cutoff_exclusive": core.DEVELOPMENT_END_EXCLUSIVE.isoformat(),
        "price_timing": "arithmetic open t to next open t+1; source terminal uses final in-hour close",
        "funding_timing": "action at t excludes funding[t] and includes funding[t+1] on grown fixed quantity; terminal exit includes no next-boundary funding",
        "pair_cashflow_formula": "sum_l w[t,l]*r[t,l] - sum_l w[t,l]*(1+r[t,l])*funding[t+1,l]",
        "funding_sign": "positive rate is paid by a long and received by a short",
        "hedge_formula": "signal=1/(1+abs(beta)); hedge=-beta/(1+abs(beta))",
        "hedge_information": "test fixture",
        "symbols": list(symbols),
        "pair_ids": list(pair_ids),
        "transient_open_gap_report": transient_report,
        "provenance": {"audited_source_hashes": {"fixture.py": "a" * 64}},
    }
    return core.HourlyEconomicInputs(
        clock=clock,
        symbols=symbols,
        pair_ids=pair_ids,
        pair_leg_indices=legs,
        pair_weights=weights,
        coin_open_prices=np.full_like(returns, 100.0),
        coin_open_to_open_returns=returns,
        funding_at_boundary=funding_boundary,
        next_boundary_growth=growth,
        funding_at_next_boundary=funding_next,
        coin_decision_available=decision,
        coin_target_available=target_available_coin,
        pair_decision_available=pair_decision,
        pair_target_cashflows=targets,
        pair_target_available=pair_target,
        terminal_exit_mask=np.zeros_like(returns, dtype=bool),
        metadata=metadata,
    )


def write_economic_fixture(
    tmp_path: Path, economic: core.HourlyEconomicInputs
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "economic_v2.npz"
    np.savez_compressed(
        artifact,
        metadata_json=np.asarray(json.dumps(economic.metadata, sort_keys=True, separators=(",", ":"))),
        clock_ns=economic.clock.asi8,
        symbols=np.asarray(economic.symbols),
        pair_ids=np.asarray(economic.pair_ids),
        pair_leg_indices=economic.pair_leg_indices,
        pair_weights=economic.pair_weights,
        coin_open_prices=economic.coin_open_prices,
        coin_open_to_open_returns=economic.coin_open_to_open_returns,
        funding_at_boundary=economic.funding_at_boundary,
        funding_at_next_boundary=economic.funding_at_next_boundary,
        next_boundary_growth=economic.next_boundary_growth,
        coin_decision_available=economic.coin_decision_available,
        coin_target_available=economic.coin_target_available,
        terminal_exit_mask=economic.terminal_exit_mask,
        pair_decision_available=economic.pair_decision_available,
        pair_target_available=economic.pair_target_available,
        pair_target_cashflows=economic.pair_target_cashflows,
    )
    one_period = {
        name: {"passed": True}
        for name in (
            "ordinary_with_next_boundary_funding",
            "ordinary_without_funding_event",
            "terminal_delisting_mark",
        )
    }
    manifest = {
        "schema": "paired_hourly_economic_inputs_manifest_v2",
        "status": "complete_immutable_shared_training_and_ledger_input",
        "artifact_path": str(artifact),
        "artifact_sha256": core.sha256_file(artifact),
        "contract_sha256": "b" * 64,
        "clock_start": economic.clock[0].isoformat(),
        "clock_end": economic.clock[-1].isoformat(),
        "hourly_observations": len(economic.clock),
        "symbols": list(economic.symbols),
        "pair_ids_sha256": hashlib.sha256("\n".join(economic.pair_ids).encode()).hexdigest(),
        "pairs": len(economic.pair_ids),
        "pair_decision_available_cells": int(economic.pair_decision_available.sum()),
        "pair_target_available_cells": int(economic.pair_target_available.sum()),
        "terminal_exit_cells": int(economic.terminal_exit_mask.sum()),
        "funding_verified_archives_loaded": 1,
        "funding_missing_archives": 0,
        "funding_included_in_target": True,
        "pair_target_formula": "sum_l w[t,l]*r[t,l] - sum_l w[t,l]*(1+r[t,l])*funding[t+1,l]",
        "one_period_ledger_identity": one_period,
        "multi_period_ledger_identity": {
            "passed": True,
            "terminal_zero_target_included": True,
        },
        "transient_open_gap_report": economic.metadata["transient_open_gap_report"],
        "audited_source_hashes": {"fixture.py": "a" * 64},
        "acquisition_manifest_sha256": "c" * 64,
        "archive_availability_sha256": "d" * 64,
        "artifact_hashes_sha256": "e" * 64,
        "latest_raw_month": "2025-06",
        "july_or_later_path_count": 0,
        "confirmation_rows_parsed": False,
    }
    manifest_path = artifact.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return artifact, manifest_path


def test_contract_predeclares_bounded_family_and_never_best_seed() -> None:
    contract = json.loads((HERE / "contract.json").read_text(encoding="utf-8"))
    assert contract["candidate_budget"]["maximum_unique_strategy_return_series"] == 5
    assert contract["candidate_budget"]["maximum_full_model_fits"] == 78
    assert contract["models"]["common_training"]["fixed_seeds"] == [1701, 1702, 1703]
    assert "never select a seed" in contract["models"]["common_training"][
        "seed_aggregation"
    ]
    assert contract["models"]["common_training"]["cost_aware_training_objective"] is False
    assert contract["portfolio"]["same_engine_and_inputs_for_every_representation_and_model"]
    gap_policy = contract["data"]["transient_open_gap_policy"]
    assert gap_policy["rule"] == "no_fill_no_shortening_mask_outcome_only"
    assert gap_policy["expected_coin_gap_cells"] == 16
    assert gap_policy["expected_pair_label_cells_masked"] == 312
    assert gap_policy["maximum_oos_coin_gap_cells"] == 0
    assert gap_policy["maximum_oos_pair_label_cells"] == 0


def test_cutoff_is_validated_before_manifest_or_cache_access(tmp_path: Path) -> None:
    with pytest.raises(core.ConfirmationFirewallError):
        core.load_development_closes(
            tmp_path / "cache-does-not-exist",
            manifest_directory=tmp_path / "manifest-does-not-exist",
            symbols=("BTCUSDT", "ETHUSDT"),
            start="2025-06-01T00:00:00Z",
            end_exclusive="2025-07-01T00:00:00.000001Z",
        )
    assert not (tmp_path / "cache-does-not-exist").exists()


def test_july_boundary_maps_to_june_without_reserved_month() -> None:
    assert core.development_months(
        "2025-01-01T00:00:00Z", "2025-07-01T00:00:00Z"
    ) == ("2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06")
    with pytest.raises(core.ConfirmationFirewallError):
        core.require_development_month("2025-07")


def test_control_preserves_four_endpoints_and_destroys_order() -> None:
    rng = np.random.default_rng(3)
    native = rng.normal(size=(4, 5, 60)).astype(np.float32)
    control = core.causal_15m_control(native)
    assert control.shape == native.shape
    assert not np.array_equal(control, native)
    np.testing.assert_allclose(
        native.reshape(4, 5, 4, 15).sum(axis=-1),
        control.reshape(4, 5, 4, 15).sum(axis=-1),
        rtol=2e-6,
        atol=2e-9,
    )


def test_checked_in_manifest_is_verified_and_full_run_rejects_short_stage() -> None:
    manifest = core.load_acquisition_manifest(
        SHORT_MANIFEST, require_full_contract=False
    )
    assert manifest.first_month == "2023-09"
    assert manifest.last_month == "2025-06"
    assert len(manifest.universe) == 24
    assert len(manifest.kline_records) == 24 * 22
    with pytest.raises(ValueError, match="full January-2021 manifest"):
        core.load_acquisition_manifest(SHORT_MANIFEST, require_full_contract=True)


def test_manifest_declared_404_is_all_missing_without_opening_cache(tmp_path: Path) -> None:
    # MATIC October 2024 is a verified remote 404 in this receipt.  A nonexistent
    # cache root is sufficient: the reader must materialize NaN and never inspect
    # a raw path for a manifest-declared 404.
    closes = core.load_development_closes(
        tmp_path / "nonexistent-cache",
        manifest_directory=SHORT_MANIFEST,
        symbols=("MATICUSDT",),
        start="2024-10-01T00:00:00Z",
        end_exclusive="2024-11-01T00:00:00Z",
        require_full_contract_manifest=False,
    )
    assert closes.shape == (31 * 24 * 60, 1)
    assert closes["MATICUSDT"].isna().all()
    assert not (tmp_path / "nonexistent-cache").exists()


def test_manifest_declared_available_but_absent_cache_is_hard_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        core.load_development_closes(
            tmp_path / "empty-cache",
            manifest_directory=SHORT_MANIFEST,
            symbols=("BTCUSDT",),
            start="2023-09-01T00:00:00Z",
            end_exclusive="2023-10-01T00:00:00Z",
            require_full_contract_manifest=False,
        )


def test_available_archive_is_bound_to_manifest_and_official_checksum() -> None:
    closes = core.load_development_closes(
        INTEGRATION_CACHE,
        manifest_directory=INTEGRATION_MANIFEST,
        symbols=("BTCUSDT", "ETHUSDT"),
        start="2023-09-01T00:00:00Z",
        end_exclusive="2023-10-01T00:00:00Z",
        require_full_contract_manifest=False,
    )
    assert closes.shape == (30 * 24 * 60, 2)
    assert not closes.isna().any().any()


def test_target_is_next_hour_and_features_do_not_use_it() -> None:
    base = synthetic_closes()
    first = core.build_components_from_closes(base, allow_synthetic_target=True)
    usable = np.flatnonzero(first.feature_available.all(axis=1))[10]
    decision = first.clock[usable]
    # Add one return shock inside [decision, decision+1h).  Reconstructing all
    # subsequent closes changes exactly that minute return and hence target t.
    shocked = base.copy()
    shock_location = shocked.index.get_loc(decision + pd.Timedelta(30, unit="min"))
    log_values = np.log(shocked["ETHUSDT"].to_numpy())
    log_values[shock_location:] += 0.01
    shocked["ETHUSDT"] = np.exp(log_values)
    second = core.build_components_from_closes(shocked, allow_synthetic_target=True)
    first_x = first.features([usable], representation="native_1m")
    second_x = second.features([usable], representation="native_1m")
    np.testing.assert_array_equal(first_x, second_x)
    assert not np.array_equal(first.target_cashflows[usable], second.target_cashflows[usable])
    np.testing.assert_array_equal(first.pair_weights[usable], second.pair_weights[usable])


def test_paired_features_share_every_coordinate_after_minute_path() -> None:
    components = core.build_components_from_closes(
        synthetic_closes(), allow_synthetic_target=True
    )
    rows = np.flatnonzero(components.feature_available.all(axis=1))[:12]
    audit = core.paired_feature_audit(components, rows)
    assert audit["maximum_common_tail_difference"] == 0.0
    assert audit["maximum_15m_endpoint_difference"] < 1e-8
    assert audit["maximum_within_block_path_difference"] > 0.0


def test_control_canonicalizes_nonfinite_paths_for_unavailable_pairs() -> None:
    components = core.build_components_from_closes(
        synthetic_closes(), allow_synthetic_target=True
    )
    row = int(np.flatnonzero(components.feature_available.all(axis=1))[0])
    weights = components.pair_weights.copy()
    feature_available = components.feature_available.copy()
    weights[row - 1, 0] = np.nan
    feature_available[row, 0] = False
    with_missing_pair = replace(
        components,
        pair_weights=weights,
        feature_available=feature_available,
    )

    native = with_missing_pair.features([row], representation="native_1m")
    control = with_missing_pair.features([row], representation="causal_15m_control")
    assert np.isfinite(native).all()
    assert np.isfinite(control).all()
    np.testing.assert_array_equal(native[0, 0], 0.0)
    np.testing.assert_array_equal(control[0, 0], 0.0)
    audit = core.paired_feature_audit(with_missing_pair, [row])
    assert audit["maximum_common_tail_difference"] == 0.0


def test_materialization_and_all_three_model_interfaces_smoke() -> None:
    components = core.build_components_from_closes(
        synthetic_closes(), allow_synthetic_target=True
    )
    usable = np.flatnonzero(
        components.feature_available.all(axis=1)
        & components.target_available.all(axis=1)
    )[:180]
    fold = models.cycle.PilotFold(
        "synthetic", usable[:80], usable[80:130], usable[130:]
    )
    panel, local_fold, _ = models.materialize_fold(
        components, fold, representation="native_1m"
    )
    ridge = models.fit_endpoint_ridge(panel, local_fold)
    test = local_fold.test_indices
    assert ridge.predict(panel.raw_features[test], panel.feature_available[test]).shape == (50, 3)
    lstm = models.fit_lstm(
        panel,
        local_fold,
        config=models.LSTMConfig(
            maximum_epochs=1,
            early_stopping_patience=1,
            batch_size=32,
            seed=1701,
            device="cpu",
        ),
    )
    assert lstm.predict(panel.raw_features[test], panel.feature_available[test]).shape == (50, 3)
    transformer = models.fit_itransformer(
        panel,
        local_fold,
        config=models.transformer_config(
            seed=1701,
            device="cpu",
            maximum_epochs=1,
            early_stopping_patience=1,
            batch_size=32,
        ),
    )
    prediction = transformer.predict(
        panel.raw_features[test],
        panel.zero_auxiliary[test],
        panel.feature_available[test],
    )
    assert prediction.shape == (50, 3)
    assert np.isfinite(prediction[panel.feature_available[test]]).all()


def test_endpoint_ridge_is_exactly_invariant_to_representation() -> None:
    components = core.build_components_from_closes(
        synthetic_closes(), allow_synthetic_target=True
    )
    usable = np.flatnonzero(components.feature_available.all(axis=1))[:180]
    fold = models.cycle.PilotFold("synthetic", usable[:80], usable[80:130], usable[130:])
    predictions: list[np.ndarray] = []
    for representation in ("native_1m", "causal_15m_control"):
        panel, local_fold, _ = models.materialize_fold(
            components, fold, representation=representation
        )
        fitted = models.fit_endpoint_ridge(panel, local_fold)
        test = local_fold.test_indices
        predictions.append(
            fitted.predict(panel.raw_features[test], panel.feature_available[test])
        )
    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_seed_aggregation_requires_all_fixed_seeds_and_never_selects_best() -> None:
    predictions = {
        1701: np.asarray([[1.0, np.nan]], dtype=np.float32),
        1702: np.asarray([[2.0, np.nan]], dtype=np.float32),
        1703: np.asarray([[6.0, np.nan]], dtype=np.float32),
    }
    result = models.mean_seed_predictions(predictions)
    assert result[0, 0] == pytest.approx(3.0)
    assert np.isnan(result[0, 1])
    with pytest.raises(ValueError, match="exact fixed seeds"):
        models.mean_seed_predictions({1701: predictions[1701], 1702: predictions[1702]})


def test_factorized_cache_roundtrip_preserves_target_semantics(tmp_path: Path) -> None:
    components = core.build_components_from_closes(
        synthetic_closes(hours=500), allow_synthetic_target=True
    )
    path = tmp_path / "components.npz"
    core.save_components(path, components, provenance={"test": "yes"})
    loaded, metadata = core.load_components(path)
    assert metadata["decision_semantics"] == "clock t is start of target interval [t,t+1h)"
    np.testing.assert_array_equal(loaded.pair_path_endpoints, components.pair_path_endpoints)
    np.testing.assert_array_equal(loaded.target_cashflows, components.target_cashflows)
    np.testing.assert_array_equal(loaded.feature_available, components.feature_available)
    with pytest.raises(FileExistsError):
        core.save_components(path, components, provenance={"test": "yes"})


def test_production_component_builder_requires_audited_economic_input() -> None:
    with pytest.raises(ValueError, match="requires audited hourly economic inputs"):
        core.build_components_from_closes(synthetic_closes(hours=300))


def test_economic_target_adapter_is_exact_and_manifest_verified(tmp_path: Path) -> None:
    economic = synthetic_economic_inputs(hours=300)
    artifact, manifest = write_economic_fixture(tmp_path, economic)
    loaded, receipt = core.load_hourly_economic_inputs(artifact, manifest)
    assert receipt["schema"] == "paired_hourly_economic_inputs_manifest_v2"
    np.testing.assert_array_equal(loaded.pair_target_cashflows, economic.pair_target_cashflows)
    components = core.build_components_from_closes(
        synthetic_closes(hours=300), economic_inputs=loaded
    )
    np.testing.assert_array_equal(components.target_cashflows, economic.pair_target_cashflows)
    np.testing.assert_array_equal(components.target_available, economic.pair_target_available)
    assert np.any(np.abs(components.target_cashflows) > 0.0)


def test_feature_availability_never_uses_current_outcome_availability() -> None:
    closes = synthetic_closes(hours=400)
    economic = synthetic_economic_inputs(hours=400)
    first = core.build_components_from_closes(closes, economic_inputs=economic)
    row = int(np.flatnonzero(first.feature_available.all(axis=1))[10])
    changed_mask = economic.pair_target_available.copy()
    changed_targets = economic.pair_target_cashflows.copy()
    changed_mask[row] = False
    changed_targets[row] = 0.0
    changed = replace(
        economic,
        pair_target_available=changed_mask,
        pair_target_cashflows=changed_targets,
    )
    second = core.build_components_from_closes(closes, economic_inputs=changed)
    np.testing.assert_array_equal(first.feature_available, second.feature_available)
    np.testing.assert_array_equal(
        first.features([row], representation="native_1m"),
        second.features([row], representation="native_1m"),
    )
    assert first.target_available[row].all()
    assert not second.target_available[row].any()


def test_transient_open_gap_masks_only_outcome_and_preserves_component_features() -> None:
    closes = synthetic_closes(hours=400)
    base = synthetic_economic_inputs(hours=400)
    base_components = core.build_components_from_closes(closes, economic_inputs=base)
    row, coin = 220, 0
    opens = base.coin_open_prices.copy()
    returns = base.coin_open_to_open_returns.copy()
    growth = base.next_boundary_growth.copy()
    coin_decision = base.coin_decision_available.copy()
    coin_target = base.coin_target_available.copy()
    pair_decision = base.pair_decision_available.copy()
    pair_target = base.pair_target_available.copy()
    targets = base.pair_target_cashflows.copy()
    opens[row + 1, coin] = np.nan
    returns[row, coin] = np.nan
    growth[row, coin] = np.nan
    coin_target[row, coin] = False
    affected_pairs = np.any(base.pair_leg_indices == coin, axis=1)
    pair_target[row, affected_pairs] = False
    targets[row, affected_pairs] = 0.0
    coin_decision[row + 1, coin] = False
    coin_target[row + 1, coin] = False
    returns[row + 1, coin] = np.nan
    growth[row + 1, coin] = np.nan
    pair_decision[row + 1, affected_pairs] = False
    pair_target[row + 1, affected_pairs] = False
    targets[row + 1, affected_pairs] = 0.0
    report = {
        "policy": "no_fill_no_shortening_mask_outcome_only",
        "no_fill": True,
        "no_shortening": True,
        "feature_and_decision_availability_unchanged": True,
        "coin_gap_cells": 1,
        "pair_label_cells_masked": int(affected_pairs.sum()),
        "oos_coin_gap_cells": 0,
        "oos_pair_label_cells": 0,
        "events": [
            {
                "decision_timestamp": base.clock[row].isoformat(),
                "missing_next_open_timestamp": base.clock[row + 1].isoformat(),
                "symbols": [base.symbols[coin]],
                "coin_gap_cells": 1,
                "pair_label_cells_masked": int(affected_pairs.sum()),
            }
        ],
    }
    metadata = dict(base.metadata)
    metadata["transient_open_gap_report"] = report
    gapped = replace(
        base,
        coin_open_prices=opens,
        coin_open_to_open_returns=returns,
        next_boundary_growth=growth,
        coin_decision_available=coin_decision,
        coin_target_available=coin_target,
        pair_decision_available=pair_decision,
        pair_target_available=pair_target,
        pair_target_cashflows=targets,
        metadata=metadata,
    )
    gapped_components = core.build_components_from_closes(
        closes, economic_inputs=gapped
    )
    np.testing.assert_array_equal(
        base_components.feature_available[row], gapped_components.feature_available[row]
    )
    np.testing.assert_array_equal(
        base_components.features([row], representation="native_1m"),
        gapped_components.features([row], representation="native_1m"),
    )
    assert gapped.coin_decision_available[row, coin]
    assert not gapped.coin_target_available[row, coin]
    assert np.isnan(gapped.coin_open_to_open_returns[row, coin])
    assert not gapped_components.target_available[row, affected_pairs].any()


def test_previous_hour_path_uses_w_t_minus_1_not_current_w_t() -> None:
    components = core.build_components_from_closes(
        synthetic_closes(hours=400),
        economic_inputs=synthetic_economic_inputs(hours=400),
    )
    row = int(np.flatnonzero(components.feature_available.all(axis=1))[5])
    episode = row - 1
    pair = 0
    signal, hedge = components.pair_leg_indices[pair]
    signal_path = components.coin_minute_returns[episode, signal].astype(np.float64)
    hedge_path = components.coin_minute_returns[episode, hedge].astype(np.float64)
    expected = (
        components.pair_weights[episode, pair, 0]
        * signal_path
        + components.pair_weights[episode, pair, 1]
        * hedge_path
    ).astype(np.float32)
    wrong_current_weight = (
        components.pair_weights[row, pair, 0]
        * signal_path
        + components.pair_weights[row, pair, 1]
        * hedge_path
    ).astype(np.float32)
    observed = components.features([row], representation="native_1m")[0, pair, :60]
    np.testing.assert_array_equal(observed, expected)
    assert not np.allclose(observed, wrong_current_weight, atol=0.0, rtol=0.0)


def test_economic_loader_rejects_v1_and_extra_npz_member(tmp_path: Path) -> None:
    economic = synthetic_economic_inputs(hours=220)
    artifact, manifest_path = write_economic_fixture(tmp_path, economic)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "paired_hourly_economic_inputs_manifest_v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema differs"):
        core.load_hourly_economic_inputs(artifact, manifest_path)

    artifact, manifest_path = write_economic_fixture(tmp_path / "second", economic)
    with np.load(artifact, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["unexpected"] = np.asarray(1)
    replacement = tmp_path / "extra_member.npz"
    np.savez_compressed(replacement, **arrays)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_path"] = str(replacement)
    manifest["artifact_sha256"] = core.sha256_file(replacement)
    replacement_manifest = tmp_path / "extra_member.manifest.json"
    replacement_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="NPZ members differ"):
        core.load_hourly_economic_inputs(replacement, replacement_manifest)


def _forecast_test_identity() -> dict[str, object]:
    sources = {
        "contract.json": "1" * 64,
        "paired_neural_core.py": "2" * 64,
        "paired_neural_models.py": "3" * 64,
        "run_paired_neural.py": "4" * 64,
        "parent_cycle_pilot.py": "5" * 64,
        "parent_raw_4h_contract.json": "6" * 64,
    }
    return runner.forecast_identity(
        contract_sha256="a" * 64,
        component_sha256="b" * 64,
        pair_axis_sha256_value="c" * 64,
        source_hashes=sources,
        source_bundle_sha256_value="d" * 64,
        runtime_environment_sha256_value="e" * 64,
        economic_artifact_sha256="f" * 64,
        economic_manifest_sha256="0" * 64,
        fold="2023Q2",
        representation="native_1m",
        model="itransformer_seed_mean",
        seed="all3",
        role="fixed_three_seed_arithmetic_mean",
    )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("contract_sha256", "9" * 64),
        ("component_sha256", "9" * 64),
        ("pair_axis_sha256", "9" * 64),
        ("fold", "2023Q3"),
        ("representation", "causal_15m_control"),
        ("model", "lstm_seed_mean"),
        ("seed", 1701),
        ("core_source_sha256", "9" * 64),
        ("models_source_sha256", "9" * 64),
        ("runner_source_sha256", "9" * 64),
        ("runtime_environment_sha256", "9" * 64),
        ("economic_artifact_sha256", "9" * 64),
        ("economic_manifest_sha256", "9" * 64),
    ],
)
def test_resume_rejects_every_swapped_identity(
    tmp_path: Path, field: str, replacement: object
) -> None:
    clock = pd.date_range("2023-04-01", periods=3, freq="1h", tz="UTC")
    mask = np.ones((3, 6), dtype=bool)
    predictions = np.arange(18, dtype=np.float32).reshape(3, 6) / 100.0
    portfolio = {"long_pairs": 2, "short_pairs": 2, "absolute_pair_target": 0.05}
    pair_targets = models.cycle.rank_pair_targets(predictions, mask, **portfolio)
    identity = _forecast_test_identity()
    path = tmp_path / "forecast.npz"
    runner._save_forecast(
        path,
        clock=clock,
        predictions=predictions,
        pair_targets=pair_targets,
        diagnostics=identity,
    )
    runner.validate_forecast_artifact(
        path,
        expected_clock=clock,
        expected_feature_mask=mask,
        expected_identity=identity,
        allowed_diagnostic_extras=set(),
        portfolio=portfolio,
        require_pair_targets=True,
        expected_predictions=predictions,
    )
    stale = dict(identity)
    stale[field] = replacement
    with pytest.raises(ValueError, match="identity differs"):
        runner.validate_forecast_artifact(
            path,
            expected_clock=clock,
            expected_feature_mask=mask,
            expected_identity=stale,
            allowed_diagnostic_extras=set(),
            portfolio=portfolio,
            require_pair_targets=True,
        )


def test_resume_rebuilds_ensemble_prediction_and_pair_targets(tmp_path: Path) -> None:
    clock = pd.date_range("2023-04-01", periods=2, freq="1h", tz="UTC")
    mask = np.ones((2, 6), dtype=bool)
    predictions = np.arange(12, dtype=np.float32).reshape(2, 6) / 100.0
    portfolio = {"long_pairs": 2, "short_pairs": 2, "absolute_pair_target": 0.05}
    identity = _forecast_test_identity()
    path = tmp_path / "good.npz"
    targets = models.cycle.rank_pair_targets(predictions, mask, **portfolio)
    runner._save_forecast(
        path,
        clock=clock,
        predictions=predictions,
        pair_targets=targets,
        diagnostics=identity,
    )
    changed_prediction = predictions.copy()
    changed_prediction[0, 0] += 1.0
    with pytest.raises(ValueError, match="prediction content differs"):
        runner.validate_forecast_artifact(
            path,
            expected_clock=clock,
            expected_feature_mask=mask,
            expected_identity=identity,
            allowed_diagnostic_extras=set(),
            portfolio=portfolio,
            require_pair_targets=True,
            expected_predictions=changed_prediction,
        )
    bad_targets = targets.copy()
    bad_targets[0, 0] += 0.01
    bad_path = tmp_path / "bad_targets.npz"
    runner._save_forecast(
        bad_path,
        clock=clock,
        predictions=predictions,
        pair_targets=bad_targets,
        diagnostics=identity,
    )
    with pytest.raises(ValueError, match="pair targets differ"):
        runner.validate_forecast_artifact(
            bad_path,
            expected_clock=clock,
            expected_feature_mask=mask,
            expected_identity=identity,
            allowed_diagnostic_extras=set(),
            portfolio=portfolio,
            require_pair_targets=True,
        )


def test_resume_rejects_missing_forecast_member_and_v1_component(tmp_path: Path) -> None:
    clock = pd.date_range("2023-04-01", periods=2, freq="1h", tz="UTC")
    mask = np.ones((2, 6), dtype=bool)
    predictions = np.zeros((2, 6), dtype=np.float32)
    incomplete = tmp_path / "incomplete.npz"
    np.savez_compressed(incomplete, clock_ns=clock.asi8, predictions=predictions)
    with pytest.raises(ValueError, match="member family differs"):
        runner.validate_forecast_artifact(
            incomplete,
            expected_clock=clock,
            expected_feature_mask=mask,
            expected_identity=_forecast_test_identity(),
            allowed_diagnostic_extras=set(),
            portfolio={"long_pairs": 2, "short_pairs": 2, "absolute_pair_target": 0.05},
            require_pair_targets=False,
        )

    components = core.build_components_from_closes(
        synthetic_closes(hours=300), allow_synthetic_target=True
    )
    v2 = tmp_path / "v2.npz"
    core.save_components(v2, components, provenance={"test": "yes"})
    with np.load(v2, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["schema"] = "paired_1m_factorized_components_v1"
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    v1 = tmp_path / "v1.npz"
    np.savez_compressed(v1, **arrays)
    with pytest.raises(ValueError, match="schema differs"):
        core.load_components(v1)


def test_discarded_benchmark_receipt_is_resource_only_and_exact(tmp_path: Path) -> None:
    sources = {"source.py": "1" * 64}
    environment = {"device": "cuda", "torch": "test"}
    receipt = {
        "schema": "paired_neural_discarded_benchmark_v2",
        "status": "full_fold_engineering_fit_completed_and_discarded",
        "contract_sha256": "a" * 64,
        "component_sha256": "b" * 64,
        "pair_axis_sha256": "c" * 64,
        "source_hashes": sources,
        "source_bundle_sha256": "d" * 64,
        "runtime_environment": environment,
        "runtime_environment_sha256": "e" * 64,
        "economic_artifact_sha256": "f" * 64,
        "economic_manifest_sha256": "0" * 64,
        "fold": "2023Q2",
        "representation": "native_1m",
        "model": "itransformer",
        "seed": 1701,
        "train_rows": 8760,
        "validation_rows": 2160,
        "test_rows_materialized_but_never_scored": 2184,
        "elapsed_seconds": 12.5,
        "peak_cuda_allocated_bytes": 123,
        "peak_cuda_reserved_bytes": 456,
        "peak_process_rss_bytes": 4_300_000_000,
        "fit_discarded": True,
        "contains_predictions_targets_ic_or_performance": False,
        "confirmation_parsed": False,
    }
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    validated = runner.validate_benchmark_receipt(
        path,
        contract_hash="a" * 64,
        component_hash="b" * 64,
        pair_hash="c" * 64,
        sources=sources,
        bundle_hash="d" * 64,
        environment=environment,
        environment_hash="e" * 64,
        economic_artifact_hash="f" * 64,
        economic_manifest_hash="0" * 64,
    )
    assert validated["peak_process_rss_bytes"] == 4_300_000_000
    assert not any(
        name in validated
        for name in ("predictions", "pair_targets", "IC", "loss", "performance_metrics")
    )
    receipt["predictions"] = [[1.0]]
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt schema differs"):
        runner.validate_benchmark_receipt(
            path,
            contract_hash="a" * 64,
            component_hash="b" * 64,
            pair_hash="c" * 64,
            sources=sources,
            bundle_hash="d" * 64,
            environment=environment,
            environment_hash="e" * 64,
            economic_artifact_hash="f" * 64,
            economic_manifest_hash="0" * 64,
        )


def test_completed_run_manifest_is_validated_and_returns_without_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = core.build_components_from_closes(
        synthetic_closes(hours=300), allow_synthetic_target=True
    )
    contract_hash = "a" * 64
    component_hash = "b" * 64
    sources = {"test.py": "c" * 64}
    bundle_hash = "d" * 64
    pair_hash = runner.pair_axis_sha256(components.pair_ids)
    environment = {"test": True}
    metadata = {
        "pair_axis_sha256": pair_hash,
        "provenance": {
            "paired_contract_sha256": contract_hash,
            "source_hashes": sources,
            "source_bundle_sha256": bundle_hash,
            "economic_artifact_sha256": "e" * 64,
            "economic_manifest_sha256": "f" * 64,
        },
    }
    output = tmp_path / "completed"
    output.mkdir()
    (output / "run_manifest.json").write_text("{}", encoding="utf-8")
    component_path = tmp_path / "components.npz"
    component_path.write_bytes(b"component")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "load_contract", lambda: {"folds": [], "portfolio": {}})
    monkeypatch.setattr(runner, "load_components", lambda _path: (components, metadata))
    monkeypatch.setattr(runner, "current_source_hashes", lambda: sources)
    monkeypatch.setattr(runner, "source_bundle_sha256", lambda _sources: bundle_hash)
    monkeypatch.setattr(runner, "configure_determinism", lambda seed=1701: None)
    monkeypatch.setattr(runner, "runtime_environment", lambda device: environment)
    monkeypatch.setattr(runner, "runtime_environment_sha256", lambda _env: "0" * 64)
    monkeypatch.setattr(runner, "validate_benchmark_receipt", lambda *args, **kwargs: {})
    hashes = {
        str(runner.CONTRACT_PATH): contract_hash,
        str(component_path): component_hash,
        str(benchmark_path): "1" * 64,
    }
    monkeypatch.setattr(runner, "sha256_file", lambda path: hashes[str(Path(path))])
    called: list[bool] = []

    def completed_validator(*args: object, **kwargs: object) -> dict[str, object]:
        called.append(True)
        return {"status": "already_complete_and_valid"}

    monkeypatch.setattr(runner, "validate_completed_forecast_run", completed_validator)
    monkeypatch.setattr(
        runner,
        "fit_endpoint_ridge",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected refit")),
    )
    result = runner.train_full(
        SimpleNamespace(
            components=component_path,
            output=output,
            device="cpu",
            benchmark_receipt=benchmark_path,
        )
    )
    assert result == 0
    assert called == [True]
