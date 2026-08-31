from __future__ import annotations

import json
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
import history_verify_evaluate_v2 as v2
import run_history as history


def _ledger(gross: tuple[float, ...]) -> SimpleNamespace:
    clock = pd.date_range("2024-01-01T00:00:00Z", periods=len(gross), freq="1h")
    bars = pd.DataFrame(
        {
            "price_return": gross,
            "funding_return": np.zeros(len(gross)),
            "transaction_cost": np.zeros(len(gross)),
            "net_return": gross,
            "gross_return": gross,
            "turnover": np.full(len(gross), 0.5 / len(gross)),
        },
        index=clock,
    )
    coin = pd.DataFrame(
        {
            "timestamp": clock,
            "coin": "BTCUSDT",
            "price_return": gross,
            "funding_return": np.zeros(len(gross)),
            "transaction_cost": np.zeros(len(gross)),
        }
    ).set_index(["timestamp", "coin"])
    return SimpleNamespace(bars=bars, coin_ledger=coin)


def test_hourly_gross_break_even_is_not_daily_compounded_gross() -> None:
    daily, _coin = v2._calculation_daily_attribution(_ledger((0.01, 0.02)))
    row = daily.iloc[0]
    assert row["daily_compounded_gross_return"] == pytest.approx(0.0302)
    assert row["hourly_gross_return_sum"] == pytest.approx(0.03)
    assert row["gross_return"] == pytest.approx(0.03)
    corrected = core.performance_metrics(
        daily["net_return"], turnover=0.5, gross_sum=float(daily["gross_return"].sum())
    )
    old = core.performance_metrics(
        daily["net_return"],
        turnover=0.5,
        gross_sum=float(daily["daily_compounded_gross_return"].sum()),
    )
    assert corrected["linearized_break_even_cost_bps"] == pytest.approx(600.0)
    assert old["linearized_break_even_cost_bps"] == pytest.approx(604.0)


def test_daily_wrapper_uses_captured_v1_callable_after_real_module_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert v2.V1_CONTINUOUS_DAILY_ATTRIBUTION is history._continuous_daily_attribution
    monkeypatch.setattr(
        history, "_continuous_daily_attribution", v2._calculation_daily_attribution
    )
    # This is the exact call path used by the unchanged v1 evaluator.  It must
    # enter the v2 wrapper once and then call the captured, unpatched v1 helper.
    daily, _coin = history._continuous_daily_attribution(_ledger((0.01, 0.02)))
    assert daily.iloc[0]["hourly_gross_return_sum"] == pytest.approx(0.03)
    assert daily.iloc[0]["daily_compounded_gross_return"] == pytest.approx(0.0302)


def test_public_daily_table_restores_v1_gross_and_keeps_explicit_numerator() -> None:
    internal, _coin = v2._calculation_daily_attribution(_ledger((0.01, 0.02)))
    public = v2._restore_public_daily_table(internal.reset_index())
    assert "daily_compounded_gross_return" not in public
    assert public.loc[0, "gross_return"] == pytest.approx(0.0302)
    assert public.loc[0, "hourly_gross_return_sum"] == pytest.approx(0.03)
    broken = internal.reset_index()
    broken.loc[0, "gross_return"] += 1e-12
    with pytest.raises(ValueError, match="did not use hourly-gross"):
        v2._restore_public_daily_table(broken)


def test_array_digest_binds_shape_and_canonicalizes_nan() -> None:
    first = np.asarray([[1.0, np.nan]], dtype=np.float32)
    second = first.copy()
    assert v2._array_sha256(first) == v2._array_sha256(second)
    assert v2._array_sha256(first) != v2._array_sha256(first.reshape(2, 1))
    assert v2._array_sha256(first) != v2._array_sha256(first.astype(np.float64))


def test_frozen_seed_mean_uses_exact_three_seed_semantics() -> None:
    values = {
        1701: np.asarray([[1.0, np.nan]], dtype=np.float32),
        1702: np.asarray([[2.0, np.nan]], dtype=np.float32),
        1703: np.asarray([[6.0, np.nan]], dtype=np.float32),
    }
    rebuilt = history.frozen_models.mean_seed_predictions(values)
    assert rebuilt[0, 0] == np.float32(3.0)
    assert np.isnan(rebuilt[0, 1])
    altered = rebuilt.copy()
    altered[0, 0] = np.nextafter(altered[0, 0], np.float32(np.inf))
    assert not np.array_equal(rebuilt, altered, equal_nan=True)


def _manifest_fixture(root: Path) -> pd.DataFrame:
    contract = core.load_contract()
    rows: list[dict[str, str]] = []

    def add(fold: str, representation: str, model: str, seed: str) -> None:
        path = history._forecast_path(
            root,
            fold=fold,
            representation=representation,
            model=model,
            seed=seed,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        rows.append(
            {
                "fold": fold,
                "representation": representation,
                "model": model,
                "seed": seed,
                "path": path.relative_to(root).as_posix(),
                "sha256": core.sha256_file(path),
            }
        )

    for fold_spec in contract["folds"]:
        fold = str(fold_spec["name"])
        add(fold, "shared_endpoint", "endpoint_ridge", "none")
        for representation in v2.REPRESENTATIONS:
            for seed in ("1701", "1702", "1703"):
                add(fold, representation, "itransformer", seed)
            add(fold, representation, "itransformer_seed_mean", "all3")
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "forecast_manifest.csv", index=False)
    return frame


def test_manifest_tree_is_exact_117_and_rejects_extra_or_missing_npz(tmp_path: Path) -> None:
    frame = _manifest_fixture(tmp_path)
    assert len(frame) == 117
    loaded, indexed = v2._load_manifest_index(tmp_path, core.load_contract())
    assert len(loaded) == len(indexed) == 117
    extra = tmp_path / "forecasts" / "stray.npz"
    extra.touch()
    with pytest.raises(ValueError, match="missing or unmanifested"):
        v2._load_manifest_index(tmp_path, core.load_contract())
    extra.unlink()
    declared = tmp_path / frame.iloc[0]["path"]
    declared.unlink()
    with pytest.raises(ValueError, match="missing or unmanifested"):
        v2._load_manifest_index(tmp_path, core.load_contract())


def test_seed_receipt_validation_rebuilds_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "schema": v2.SEED_MEAN_RECEIPT_SCHEMA,
        "verified_group_count": 26,
        "groups": [{"fold": "2022Q2", "exact_arithmetic_mean": True}],
        "confirmation_parsed": False,
    }
    monkeypatch.setattr(v2, "build_seed_mean_receipt", lambda **_kwargs: expected)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(expected), encoding="utf-8")
    assert v2.validate_seed_mean_receipt(
        components_path=tmp_path / "components",
        forecast_output=tmp_path / "forecasts",
        receipt_path=path,
    ) == expected
    tampered = dict(expected, verified_group_count=25)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt differs"):
        v2.validate_seed_mean_receipt(
            components_path=tmp_path / "components",
            forecast_output=tmp_path / "forecasts",
            receipt_path=path,
        )


def test_staged_directory_publishes_once_and_cleans_failures(tmp_path: Path) -> None:
    destination = tmp_path / "complete"
    with v2._StagedDirectory(destination) as stage:
        (stage / "artifact.txt").write_text("complete", encoding="utf-8")
        assert not destination.exists()
    assert (destination / "artifact.txt").read_text(encoding="utf-8") == "complete"
    with pytest.raises(FileExistsError):
        with v2._StagedDirectory(destination):
            pass

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        with v2._StagedDirectory(failed) as stage:
            (stage / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("injected")
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))
    assert not (tmp_path / ".failed.publish.lock").exists()


def test_internal_evaluator_restores_original_attribution_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = history._continuous_daily_attribution

    def fail(_args: object) -> int:
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr(history, "command_evaluate", fail)
    with pytest.raises(RuntimeError, match="injected evaluation"):
        v2._run_internal_v1_evaluation(
            components_path=tmp_path / "components",
            economic_inputs=tmp_path / "economic",
            economic_manifest=tmp_path / "manifest",
            forecast_output=tmp_path / "forecast",
            pairs_research_root=tmp_path / "dependencies",
            output=tmp_path / "output",
        )
    assert history._continuous_daily_attribution is original


def test_v2_preserves_exact_v1_table_family_and_adds_only_a_receipt() -> None:
    assert len(v2.V1_CSV_TABLES) == 11
    assert v2.V1_BINARY_TABLES == ("hourly_constraint_audit.parquet",)
    assert len(v2.FINAL_TABLES) == 12
    assert len(set(v2.FINAL_TABLES)) == 12


def test_v2_evaluation_publishes_complete_bound_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = tmp_path / "components.npz"
    economic = tmp_path / "economic.npz"
    economic_manifest = tmp_path / "economic.manifest.json"
    forecast = tmp_path / "forecast"
    seed_receipt_path = tmp_path / "seed_receipt.json"
    for path, payload in (
        (components, b"components"),
        (economic, b"economic"),
        (economic_manifest, b"manifest"),
        (seed_receipt_path, b"seed-receipt"),
    ):
        path.write_bytes(payload)
    forecast.mkdir()
    (forecast / "run_manifest.json").write_text("{}", encoding="utf-8")
    (forecast / "forecast_manifest.csv").write_text("manifest\n", encoding="utf-8")
    seed_receipt = {"verified_group_count": 26}
    monkeypatch.setattr(v2, "validate_seed_mean_receipt", lambda **_kwargs: seed_receipt)

    internal_receipt = {
        "schema": history.EVALUATION_SCHEMA,
        "status": "complete_retrospective_continuous_history_evaluation",
        "constraint_violation_count": 0,
        "confirmation_parsed": False,
        "audited_dependency_source_hashes": {"execution.py": "a" * 64},
        "family": {"continuous_ledger_count": 12},
        "clock": {"first_decision": "2022-04-08T02:00:00+00:00"},
        "terminal_policy": "test terminal",
        "bootstrap_interpretation": "descriptive",
    }

    def fake_internal(**kwargs: object) -> dict[str, object]:
        output = Path(kwargs["output"])
        output.mkdir(parents=True)
        pd.DataFrame(
            {
                "gross_return": [0.03],
                "daily_compounded_gross_return": [0.0302],
                "hourly_gross_return_sum": [0.03],
            }
        ).to_csv(output / "continuous_daily_returns.csv", index=False)
        for name in v2.V1_CSV_TABLES[1:]:
            pd.DataFrame({"value": [1]}).to_csv(output / name, index=False)
        (output / v2.V1_BINARY_TABLES[0]).write_bytes(b"parquet")
        (output / "completion_receipt.json").write_text(
            json.dumps(internal_receipt), encoding="utf-8"
        )
        return internal_receipt

    monkeypatch.setattr(v2, "_run_internal_v1_evaluation", fake_internal)
    destination = tmp_path / "evaluation_v2"
    args = SimpleNamespace(
        components=components,
        economic_inputs=economic,
        economic_manifest=economic_manifest,
        forecast_output=forecast,
        seed_mean_receipt=seed_receipt_path,
        output=destination,
        pairs_research_root=tmp_path / "dependencies",
    )
    receipt = v2.command_evaluate_v2(args)
    assert destination.is_dir()
    assert {path.name for path in destination.iterdir()} == set(v2.FINAL_TABLES) | {
        "completion_receipt.json"
    }
    daily = pd.read_csv(destination / "continuous_daily_returns.csv")
    assert daily.loc[0, "gross_return"] == pytest.approx(0.0302)
    assert daily.loc[0, "hourly_gross_return_sum"] == pytest.approx(0.03)
    assert receipt["seed_mean_verified_group_count"] == 26
    assert receipt["forecast_family_revalidated_after_ledgers"] is True
    for name, digest in receipt["output_sha256"].items():
        assert core.sha256_file(destination / name) == digest
    assert not list(tmp_path.glob(".evaluation_v2.staging-*"))
    assert not (tmp_path / ".evaluation_v2.publish.lock").exists()
