from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import acquisition_plan
import acquire_confirmation
import confirmation_governance as governance
import evaluate_confirmation
import prepare_confirmation
import train_confirmation


def _freeze(tmp_path: Path) -> Path:
    path = tmp_path / "source_freeze.json"
    governance.create_source_freeze(path)
    return path


def test_contract_is_exact_contiguous_four_refit_confirmation() -> None:
    contract = governance.load_contract()
    assert contract["confirmation_interval"]["start_inclusive_utc"] == (
        "2025-08-01T00:00:00Z"
    )
    assert contract["confirmation_interval"]["end_exclusive_utc"] == (
        "2026-06-30T23:00:00Z"
    )
    assert contract["confirmation_interval"]["expected_decision_hours"] == 8015
    assert sum(row["expected_test_hours"] for row in contract["refits"]) == 8015
    assert len(governance.expected_specs(contract)) == 32
    assert sum(spec.role == "individual_seed" for spec in governance.expected_specs(contract)) == 24
    assert sum(spec.role == "seed_mean" for spec in governance.expected_specs(contract)) == 8


def test_static_top24_axis_produces_exactly_276_oriented_pairs() -> None:
    contract = governance.load_contract()
    pairs = governance.pair_ids(contract["data"]["symbols_in_order"])
    assert len(pairs) == len(set(pairs)) == 276
    assert pairs[0] == "ETHUSDT__BTCUSDT"
    assert pairs[-1] == "AXSUSDT__VETUSDT"


def test_acquisition_plan_is_plan_only_and_never_reaches_july_2026() -> None:
    plan, rows = acquisition_plan.validate_plan()
    assert len(rows) == 1344
    assert {row["month"] for row in rows} == set(
        acquisition_plan.months("2024-03", "2026-06")
    )
    assert all(row["month"] < "2026-07" for row in rows)
    assert plan["archive_rules"]["july_2026_or_later_path_count"] == 0
    assert plan["status"] == "plan_only_no_archive_was_downloaded_opened_or_parsed"


def test_acquisition_requires_exact_development_overlap_receipt(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    receipt = {
        "schema": "pairtoken_1m_confirmation_development_overlap_v1",
        "status": "passed_exact_parent_parity_before_confirmation_firewall",
        "contract_sha256": governance.sha256_file(governance.CONTRACT_PATH),
        "source_freeze_sha256": governance.sha256_file(freeze),
        "development_start": "2024-03-01T00:00:00+00:00",
        "development_end_inclusive": "2025-06-30T23:00:00+00:00",
        "confirmation_rows_parsed": False,
        "exact_masks": {"clock": True, "pair_legs": True},
        "maximum_absolute_differences": {"weights": 0.0, "ledger": 0.0},
        "feature_rows_compared": 16,
        "ledger_hours_compared": 169,
        "audited_source_hashes": {"source.py": "a" * 64},
    }
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert acquire_confirmation.verify_overlap_receipt(path, freeze)["confirmation_rows_parsed"] is False
    receipt["maximum_absolute_differences"]["ledger"] = 1e-15
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(governance.ConfirmationGateError, match="did not pass exactly"):
        acquire_confirmation.verify_overlap_receipt(path, freeze)


def test_source_freeze_is_create_only_and_detects_receipt_tampering(tmp_path: Path) -> None:
    path = _freeze(tmp_path)
    verified = governance.verify_source_freeze(path)
    assert verified["confirmation_data_parsed"] is False
    assert verified["confirmation_metrics_computed"] is False
    with pytest.raises(FileExistsError):
        governance.create_source_freeze(path)
    tampered = json.loads(path.read_text())
    tampered["source_bundle_sha256"] = "0" * 64
    copy = tmp_path / "tampered.json"
    copy.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(governance.ConfirmationGateError, match="source-bundle"):
        governance.verify_source_freeze(copy)


def _diagnostics(spec: governance.ForecastSpec) -> dict[str, object]:
    return {
        "schema": governance.FORECAST_SCHEMA,
        "contract_sha256": governance.sha256_file(governance.CONTRACT_PATH),
        "source_freeze_sha256": "a" * 64,
        "source_bundle_sha256": "b" * 64,
        "input_artifact_sha256": "c" * 64,
        "runtime_sha256": "d" * 64,
        "fold": spec.fold,
        "representation": spec.representation,
        "model_id": spec.model_id,
        "seed": spec.seed,
        "role": spec.role,
        "target_horizon_hours": 1,
        "confirmation_metrics_computed": False,
        "test_outcomes_loaded": False,
        "epochs": 1,
        "training_loss": 0.2,
        "validation_loss": 0.3,
    }


def test_forecast_artifact_has_governed_clock_axis_and_no_outcome_metric(
    tmp_path: Path,
) -> None:
    contract = governance.load_contract()
    spec = governance.ForecastSpec(
        "R1_2025_08_01", "native_1m", "1701", "individual_seed"
    )
    fold = governance.fold_contract(contract, spec.fold)
    clock = governance.hourly_clock_ns(fold["test_start"], fold["test_end_exclusive"])
    pairs = governance.pair_ids(contract["data"]["symbols_in_order"])
    mask = np.ones((len(clock), 276), dtype=bool)
    values = np.zeros(mask.shape, dtype=np.float32)
    path = tmp_path / spec.filename
    governance.write_forecast_artifact(
        path,
        clock_ns=clock,
        pair_axis=pairs,
        feature_available=mask,
        predictions=values,
        diagnostics=_diagnostics(spec),
        pair_targets=None,
    )
    loaded = governance.read_forecast_artifact(path, spec, contract)
    assert loaded["predictions"].shape == (1464, 276)
    assert loaded["pair_targets"] is None

    bad = dict(_diagnostics(spec), mean_ic=0.4)
    bad_path = tmp_path / "bad.npz"
    governance.write_forecast_artifact(
        bad_path,
        clock_ns=clock,
        pair_axis=pairs,
        feature_available=mask,
        predictions=values,
        diagnostics=bad,
        pair_targets=None,
    )
    with pytest.raises(governance.ConfirmationGateError, match="outcome diagnostic"):
        governance.read_forecast_artifact(bad_path, spec, contract)


def test_blinded_loader_rejects_test_outcome_member_before_shape_processing(
    tmp_path: Path,
) -> None:
    freeze = _freeze(tmp_path)
    path = tmp_path / "leaking_input.npz"
    np.savez(path, metadata_json=np.asarray("{}"), test_returns=np.ones(1))
    with pytest.raises(governance.ConfirmationGateError, match="test-outcome members"):
        train_confirmation.load_blinded_fold_input(
            path,
            fold_name="R1_2025_08_01",
            representation="native_1m",
            source_freeze_path=freeze,
        )


def _tiny_bundle(representation: str, raw: np.ndarray) -> train_confirmation.BlindedFoldInput:
    rows, pairs = raw.shape[:2]
    mask = np.ones((rows, pairs), dtype=bool)
    target = np.arange(rows * pairs, dtype=np.float32).reshape(rows, pairs)
    clock = np.arange(rows, dtype=np.int64)
    return train_confirmation.BlindedFoldInput(
        "fold", representation, tuple(f"p{i}" for i in range(pairs)),
        clock, clock, clock,
        raw.copy(), raw.copy(), raw.copy(),
        mask.copy(), mask.copy(), mask.copy(),
        target.copy(), target.copy(), mask.copy(), mask.copy(),
        "a" * 64,
    )


def test_paired_input_audit_accepts_only_matched_15m_endpoint_control() -> None:
    generator = np.random.default_rng(7)
    native_raw = generator.normal(size=(2, 3, 232)).astype(np.float32)
    control_raw = native_raw.copy()
    path = native_raw[..., :60].reshape(2, 3, 4, 15)
    means = path.mean(axis=-1, keepdims=True)
    control_raw[..., :60] = np.broadcast_to(means, path.shape).reshape(2, 3, 60)
    native = _tiny_bundle("native_1m", native_raw)
    control = _tiny_bundle("causal_15m_control", control_raw)
    train_confirmation.paired_input_audit(native, control)
    broken = control_raw.copy()
    broken[0, 0, 60] += 1.0
    with pytest.raises(governance.ConfirmationGateError, match="common tails"):
        train_confirmation.paired_input_audit(
            native, _tiny_bundle("causal_15m_control", broken)
        )


def test_factorized_blinded_input_materializes_native_and_control_without_test_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path)
    contract = governance.load_contract()
    pairs = governance.pair_ids(contract["data"]["symbols_in_order"])
    source_clock = np.arange(175, dtype=np.int64) * 3_600_000_000_000
    train_rows = np.asarray([168, 169], dtype=np.int64)
    validation_rows = np.asarray([171, 172], dtype=np.int64)
    test_rows = np.asarray([174], dtype=np.int64)
    monkeypatch.setattr(
        train_confirmation,
        "_expected_partition_clocks",
        lambda _fold: (
            source_clock[train_rows], source_clock[validation_rows], source_clock[test_rows]
        ),
    )
    legs = np.asarray([
        (second, first) for first in range(24) for second in range(first + 1, 24)
    ], dtype=np.int64)
    generator = np.random.default_rng(11)
    minute = generator.normal(0.0, 0.001, size=(175, 24, 60)).astype(np.float32)
    weights = np.zeros((175, 276, 2), dtype=np.float32)
    weights[..., 0] = 0.5
    weights[..., 1] = -0.5
    endpoints = generator.normal(0.0, 0.01, size=(175, 276)).astype(np.float32)
    feature = np.ones((175, 276), dtype=bool)
    target_train = np.zeros((2, 276), dtype=np.float32)
    target_validation = np.zeros((2, 276), dtype=np.float32)
    available = np.ones((2, 276), dtype=bool)

    def write(representation: str) -> Path:
        arrays = {
            "pair_ids": np.asarray(pairs), "pair_leg_indices": legs,
            "clock_source_ns": source_clock, "train_rows": train_rows,
            "validation_rows": validation_rows, "test_rows": test_rows,
            "coin_minute_returns": minute, "pair_weights": weights.astype(np.float64),
            "pair_path_endpoints": endpoints, "feature_available": feature,
            "target_cashflows_train": target_train.astype(np.float64),
            "target_cashflows_validation": target_validation.astype(np.float64),
            "target_available_train": available,
            "target_available_validation": available,
        }
        metadata = {
            "schema": train_confirmation.INPUT_SCHEMA,
            "status": "complete_factorized_train_validation_labels_and_test_features_only",
            "contract_sha256": governance.sha256_file(governance.CONTRACT_PATH),
            "source_freeze_sha256": governance.sha256_file(freeze),
            "source_bundle_sha256": governance.verify_source_freeze(freeze)["source_bundle_sha256"],
            "fold": "R1_2025_08_01",
            "representation": representation,
            "pair_count": 276,
            "raw_dimension": 232,
            "auxiliary_dimension": 84,
            "auxiliary_values": "all_zeros_created_by_trainer",
            "pair_axis_sha256": governance.axis_sha256(pairs),
            "test_target_labels_included": False,
            "causal_test_history_features_included": True,
            "confirmation_metrics_computed": False,
            "factorized_storage": True,
            "factorized_content_sha256": train_confirmation._array_family_sha256(arrays),
        }
        path = tmp_path / f"{representation}.npz"
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            **arrays,
        )
        return path

    native = train_confirmation.load_blinded_fold_input(
        write("native_1m"), fold_name="R1_2025_08_01",
        representation="native_1m", source_freeze_path=freeze,
    )
    control = train_confirmation.load_blinded_fold_input(
        write("causal_15m_control"), fold_name="R1_2025_08_01",
        representation="causal_15m_control", source_freeze_path=freeze,
    )
    assert native.raw_train.shape == (2, 276, 232)
    assert native.raw_test.shape == (1, 276, 232)
    assert np.array_equal(native.raw_test[..., 60:], control.raw_test[..., 60:])
    native_blocks = native.raw_test[..., :60].reshape(1, 276, 4, 15).sum(axis=-1)
    control_blocks = control.raw_test[..., :60].reshape(1, 276, 4, 15).sum(axis=-1)
    assert np.allclose(native_blocks, control_blocks, atol=2e-9, rtol=2e-6)
    train_confirmation.paired_input_audit(native, control)


def test_preparation_serializes_exact_eight_and_loader_rejects_content_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path)
    contract = governance.load_contract()
    symbols = tuple(contract["data"]["symbols_in_order"])
    pairs = governance.pair_ids(symbols)
    source_clock = np.arange(176, dtype=np.int64) * 3_600_000_000_000
    train_global = np.asarray([169, 170], dtype=np.int64)
    validation_global = np.asarray([172, 173], dtype=np.int64)
    test_global = np.asarray([175], dtype=np.int64)
    clocks = (
        source_clock[train_global], source_clock[validation_global], source_clock[test_global]
    )
    monkeypatch.setattr(prepare_confirmation, "_expected_partition_clocks", lambda _row: clocks)
    monkeypatch.setattr(train_confirmation, "_expected_partition_clocks", lambda _row: clocks)
    legs = np.asarray([
        (second, first) for first in range(24) for second in range(first + 1, 24)
    ], dtype=np.int64)
    weights = np.broadcast_to(
        np.asarray([0.5, -0.5], dtype=np.float64), (176, 276, 2)
    ).copy()
    components = prepare_confirmation.Components(
        clock=pd.to_datetime(source_clock, utc=True),
        symbols=symbols,
        pair_ids=pairs,
        legs=legs,
        coin_minute_returns=np.zeros((176, 24, 60), dtype=np.float32),
        pair_weights=weights,
        pair_endpoints=np.zeros((176, 276), dtype=np.float32),
        feature_available=np.ones((176, 276), dtype=bool),
        target_cashflows=np.zeros((176, 276), dtype=np.float64),
        target_available=np.ones((176, 276), dtype=bool),
        coin_open_prices=np.ones((176, 24), dtype=np.float64),
        coin_returns=np.zeros((176, 24), dtype=np.float64),
        funding_boundary=np.zeros((176, 24), dtype=np.float64),
        coin_execution_available=np.ones((176, 24), dtype=bool),
        pair_decision_available=np.ones((176, 276), dtype=bool),
        terminal_exit=np.zeros((176, 24), dtype=bool),
    )
    output_root = tmp_path / "prepared"
    receipts = prepare_confirmation.write_blinded_inputs(
        components,
        output_root=output_root,
        source_freeze_path=freeze,
        acquisition_hashes={},
    )
    assert len(receipts) == 8
    native_path = output_root / "R1_2025_08_01" / "native_1m.blinded.npz"
    native = train_confirmation.load_blinded_fold_input(
        native_path,
        fold_name="R1_2025_08_01",
        representation="native_1m",
        source_freeze_path=freeze,
    )
    assert native.target_train.dtype == np.float64
    with np.load(native_path, allow_pickle=False) as stored:
        changed = {name: np.asarray(stored[name]) for name in stored.files}
    changed["coin_minute_returns"] = changed["coin_minute_returns"].copy()
    changed["coin_minute_returns"][0, 0, 0] = 1.0
    tampered = tmp_path / "tampered_factorized.npz"
    np.savez_compressed(tampered, **changed)
    with pytest.raises(governance.ConfirmationGateError, match="cryptographic hash"):
        train_confirmation.load_blinded_fold_input(
            tampered,
            fold_name="R1_2025_08_01",
            representation="native_1m",
            source_freeze_path=freeze,
        )


def test_actual_frozen_trainer_tiny_fit_writes_three_seeds_and_ensemble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path)
    contract = governance.load_contract()
    pairs = governance.pair_ids(contract["data"]["symbols_in_order"])
    test_clock = np.asarray([np.datetime64("2025-08-01T00:00:00", "ns").astype(np.int64)])
    original_fold_contract = governance.fold_contract
    monkeypatch.setattr(
        governance,
        "fold_contract",
        lambda frozen, name: (
            {
                **original_fold_contract(frozen, name),
                "test_end_exclusive": "2025-08-01T01:00:00Z",
                "expected_test_hours": 1,
            }
            if name == "R1_2025_08_01"
            else original_fold_contract(frozen, name)
        ),
    )
    generator = np.random.default_rng(91)
    raw_train = generator.normal(size=(4, 276, 232)).astype(np.float32)
    raw_validation = generator.normal(size=(2, 276, 232)).astype(np.float32)
    raw_test = generator.normal(size=(1, 276, 232)).astype(np.float32)
    train_mask = np.ones((4, 276), dtype=bool)
    validation_mask = np.ones((2, 276), dtype=bool)
    test_mask = np.ones((1, 276), dtype=bool)
    bundle = train_confirmation.BlindedFoldInput(
        fold="R1_2025_08_01",
        representation="native_1m",
        pair_ids=pairs,
        train_clock_ns=test_clock[0] - np.arange(6, 2, -1, dtype=np.int64) * 3_600_000_000_000,
        validation_clock_ns=test_clock[0] - np.arange(2, 0, -1, dtype=np.int64) * 3_600_000_000_000,
        test_clock_ns=test_clock,
        raw_train=raw_train,
        raw_validation=raw_validation,
        raw_test=raw_test,
        feature_train=train_mask,
        feature_validation=validation_mask,
        feature_test=test_mask,
        target_train=generator.normal(0.0, 1e-4, size=(4, 276)),
        target_validation=generator.normal(0.0, 1e-4, size=(2, 276)),
        label_train=train_mask,
        label_validation=validation_mask,
        artifact_sha256="a" * 64,
        factorized_content_sha256="b" * 64,
    )
    output_root = tmp_path / "tiny_fit"
    models = train_confirmation.load_frozen_models(freeze)
    train_confirmation.train_one_bundle(
        bundle,
        models=models,
        output_root=output_root,
        source_freeze_path=freeze,
        device="cpu",
    )
    for seed, role in (("1701", "individual_seed"), ("1702", "individual_seed"),
                       ("1703", "individual_seed"), ("all3", "seed_mean")):
        spec = governance.ForecastSpec("R1_2025_08_01", "native_1m", seed, role)
        assert governance.expected_artifact_path(output_root, spec).is_file()
        assert governance.expected_registry_path(output_root, spec).is_file()
    assert (output_root / "runtime_manifest.json").is_file()


def test_all_fold_seal_refuses_an_incomplete_registry(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    with pytest.raises(governance.ConfirmationGateError, match="forecast registry entry"):
        governance.seal_all_forecasts(
            tmp_path / "empty_forecasts",
            source_freeze_path=freeze,
            output=tmp_path / "all_fold_seal.json",
        )
    assert not (tmp_path / "all_fold_seal.json").exists()


def test_evaluation_manifest_is_not_read_when_all_fold_seal_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExpectedSealFailure(RuntimeError):
        pass

    def fail_before_evaluation(*args: object, **kwargs: object) -> None:
        raise ExpectedSealFailure("seal first")

    monkeypatch.setattr(
        evaluate_confirmation, "verify_all_forecast_seal", fail_before_evaluation
    )
    missing_manifest = tmp_path / "must_not_be_read.json"
    with pytest.raises(ExpectedSealFailure, match="seal first"):
        evaluate_confirmation.authorize_evaluation(
            forecast_root=tmp_path / "forecast",
            source_freeze_path=tmp_path / "freeze",
            forecast_seal_path=tmp_path / "seal",
            evaluation_manifest_path=missing_manifest,
            output=tmp_path / "authorization.json",
        )
    assert not missing_manifest.exists()
    assert not (tmp_path / "authorization.json").exists()


def test_training_imports_frozen_lower_level_models_not_development_runner(
    tmp_path: Path,
) -> None:
    freeze = _freeze(tmp_path)
    models = train_confirmation.load_frozen_models(freeze)
    assert tuple(models.FIXED_SEEDS) == (1701, 1702, 1703)
    config = models.transformer_config(seed=1701, device="cpu")
    assert config.raw_dimension == 232
    assert config.auxiliary_dimension == 84
    assert config.d_model == 48
    assert config.n_heads == 4
    assert "run_paired_neural" not in models.__name__
