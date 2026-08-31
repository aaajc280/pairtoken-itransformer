#!/usr/bin/env python3
"""Verify frozen history seed means and publish a corrected atomic evaluation.

This adapter is intentionally separate from the three files bound into the
active forecast run (``contract.json``, ``history_core.py`` and
``run_history.py``).  It performs two post-fit operations:

``verify-seed-means``
    Rebuild every N02/C02 three-seed arithmetic mean exactly and create a
    deterministic, content-addressed receipt.

``evaluate``
    Reuse the unchanged v1 evaluator inside a private staging directory while
    supplying the frozen hourly-gross break-even numerator.  The public v2
    directory appears only after every table and its completion receipt exist.

No confirmation timestamp is addressable through either command.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import history_core as core  # noqa: E402
import run_history as history  # noqa: E402


SEED_MEAN_RECEIPT_SCHEMA = "crypto_pairtoken_1m_history_seed_mean_verification_v2"
EVALUATION_RECEIPT_SCHEMA = "crypto_pairtoken_1m_history_evaluation_v2"
# Capture the bound v1 callable before the temporary evaluation monkeypatch.
# Calling through ``history._continuous_daily_attribution`` from the wrapper
# would recurse after that module attribute is replaced with the wrapper.
V1_CONTINUOUS_DAILY_ATTRIBUTION = history._continuous_daily_attribution
REPRESENTATIONS = ("native_1m", "causal_15m_control")
V1_CSV_TABLES = (
    "continuous_daily_returns.csv",
    "quarterly_metrics.csv",
    "aggregate_metrics.csv",
    "quarterly_ic.csv",
    "stability_summary.csv",
    "leave_one_quarter_out_metrics.csv",
    "paired_daily_differences.csv",
    "paired_bootstrap.csv",
    "risk_audit_summary.csv",
    "concentration_summary.csv",
    "coin_contributions.csv",
)
V1_BINARY_TABLES = ("hourly_constraint_audit.parquet",)
FINAL_TABLES = V1_CSV_TABLES + V1_BINARY_TABLES


def _adapter_sha256() -> str:
    return core.sha256_file(Path(__file__))


def _array_sha256(values: np.ndarray) -> str:
    """Hash an array with explicit dtype/shape and canonical floating NaNs."""

    array = np.asarray(values)
    if array.dtype.kind == "f":
        dtype = np.dtype(f"<f{array.dtype.itemsize}")
        canonical = np.asarray(array, dtype=dtype, order="C").copy()
        canonical[np.isnan(canonical)] = np.nan
    else:
        dtype = array.dtype.newbyteorder("<")
        canonical = np.asarray(array, dtype=dtype, order="C")
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _manifest_family_sha256(frame: pd.DataFrame) -> str:
    columns = ("fold", "representation", "model", "seed", "path", "sha256")
    if set(frame.columns) != set(columns):
        raise ValueError("history forecast manifest columns differ")
    ordered = frame.loc[:, list(columns)].sort_values(list(columns[:4]))
    rows = [
        {column: str(value) for column, value in zip(columns, row, strict=True)}
        for row in ordered.itertuples(index=False, name=None)
    ]
    return hashlib.sha256(core.canonical_json({"artifacts": rows})).hexdigest()


def _load_manifest_index(root: Path, contract: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = root / "forecast_manifest.csv"
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    expected = len(contract["folds"]) * 9
    if len(frame) != expected or frame.duplicated(
        ["fold", "representation", "model", "seed"]
    ).any():
        raise ValueError("history forecast manifest family differs")
    declared = {(root / relative).resolve() for relative in frame["path"]}
    observed = {path.resolve() for path in (root / "forecasts").rglob("*.npz")}
    if observed != declared:
        raise ValueError("history forecast tree has missing or unmanifested NPZ artifacts")
    indexed = frame.set_index(["fold", "representation", "model", "seed"])
    return frame, indexed


def build_seed_mean_receipt(
    *, components_path: Path, forecast_output: Path
) -> dict[str, Any]:
    """Return the deterministic expected receipt after exact full-family audit."""

    contract = core.load_contract()
    components, _metadata, _preflight = history.preflight_inputs(
        components_path=components_path
    )
    # This validates the run manifest, all 117 declared artifact hashes and the
    # three evaluation forecast families before the more specific mean audit.
    _targets, run_manifest = history._load_evaluation_forecasts(
        forecast_output=forecast_output,
        components=components,
        component_path=components_path,
        contract=contract,
    )
    root = Path(forecast_output).expanduser().resolve()
    manifest, indexed = _load_manifest_index(root, contract)
    fold_clocks = core.expected_fold_clocks(contract)
    component_lookup = pd.Series(
        np.arange(len(components.clock), dtype=np.int64), index=components.clock
    )
    groups: list[dict[str, Any]] = []

    for fold_spec in contract["folds"]:
        fold = str(fold_spec["name"])
        clock = fold_clocks[fold]
        locations = component_lookup.reindex(clock)
        if locations.isna().any():
            raise ValueError(f"component cache lacks seed-mean clock: {fold}")
        mask = components.feature_available[locations.to_numpy(dtype=np.int64)]
        for representation in REPRESENTATIONS:
            predictions: dict[int, np.ndarray] = {}
            seed_hashes: dict[str, str] = {}
            for seed in map(int, history.frozen_models.FIXED_SEEDS):
                key = (fold, representation, "itransformer", str(seed))
                if key not in indexed.index:
                    raise ValueError(f"history seed artifact is absent: {key}")
                path = history._forecast_path(
                    root,
                    fold=fold,
                    representation=representation,
                    model="itransformer",
                    seed=seed,
                )
                identity = history._identity(
                    contract_hash=run_manifest["history_contract_sha256"],
                    component_hash=run_manifest["component_sha256"],
                    pair_hash=run_manifest["pair_axis_sha256"],
                    frozen_bundle_hash=run_manifest["frozen_source_bundle_sha256"],
                    history_bundle_hash=run_manifest["history_source_bundle_sha256"],
                    environment_hash=run_manifest["runtime_environment_sha256"],
                    fold=fold,
                    representation=representation,
                    model="itransformer",
                    seed=seed,
                    role="individual_fixed_seed_forecast",
                )
                prediction, _pair_targets, _diagnostics = history._load_forecast(
                    path,
                    expected_clock=clock,
                    expected_mask=mask,
                    expected_identity=identity,
                    require_pair_targets=False,
                    contract=contract,
                )
                observed_hash = core.sha256_file(path)
                if observed_hash != str(indexed.loc[key, "sha256"]):
                    raise ValueError(f"history seed artifact hash differs: {key}")
                predictions[seed] = prediction
                seed_hashes[str(seed)] = observed_hash

            rebuilt = history.frozen_models.mean_seed_predictions(predictions)
            ensemble_key = (
                fold,
                representation,
                "itransformer_seed_mean",
                "all3",
            )
            if ensemble_key not in indexed.index:
                raise ValueError(f"history seed-mean artifact is absent: {ensemble_key}")
            ensemble_path = history._forecast_path(
                root,
                fold=fold,
                representation=representation,
                model="itransformer_seed_mean",
                seed="all3",
            )
            ensemble_identity = history._identity(
                contract_hash=run_manifest["history_contract_sha256"],
                component_hash=run_manifest["component_sha256"],
                pair_hash=run_manifest["pair_axis_sha256"],
                frozen_bundle_hash=run_manifest["frozen_source_bundle_sha256"],
                history_bundle_hash=run_manifest["history_source_bundle_sha256"],
                environment_hash=run_manifest["runtime_environment_sha256"],
                fold=fold,
                representation=representation,
                model="itransformer_seed_mean",
                seed="all3",
                role="fixed_three_seed_arithmetic_mean",
            )
            observed, pair_targets, diagnostics = history._load_forecast(
                ensemble_path,
                expected_clock=clock,
                expected_mask=mask,
                expected_identity=ensemble_identity,
                require_pair_targets=True,
                contract=contract,
            )
            if not np.array_equal(rebuilt, observed, equal_nan=True):
                raise ValueError(
                    f"history seed mean is not the exact arithmetic mean: {fold} {representation}"
                )
            if diagnostics.get("seeds") != list(history.frozen_models.FIXED_SEEDS):
                raise ValueError("history seed-mean diagnostics name another seed family")
            if diagnostics.get("aggregation") != "arithmetic_mean_never_best_seed":
                raise ValueError("history seed-mean aggregation policy differs")
            assert pair_targets is not None
            ensemble_hash = core.sha256_file(ensemble_path)
            if ensemble_hash != str(indexed.loc[ensemble_key, "sha256"]):
                raise ValueError(f"history seed-mean artifact hash differs: {ensemble_key}")
            expected_digest = _array_sha256(rebuilt)
            observed_digest = _array_sha256(observed)
            if expected_digest != observed_digest:
                raise ValueError("equal history seed means have different canonical digests")
            groups.append(
                {
                    "fold": fold,
                    "representation": representation,
                    "seed_artifact_sha256": seed_hashes,
                    "ensemble_artifact_sha256": ensemble_hash,
                    "rebuilt_prediction_sha256": expected_digest,
                    "observed_prediction_sha256": observed_digest,
                    "pair_targets_sha256": _array_sha256(pair_targets),
                    "exact_arithmetic_mean": True,
                    "exact_ranked_target_rebuild": True,
                }
            )

    expected_groups = len(contract["folds"]) * len(REPRESENTATIONS)
    if len(groups) != expected_groups:
        raise AssertionError("history seed-mean group count differs")
    frozen_hashes = history.validate_frozen_sources(contract)
    history_hashes = history.history_source_hashes()
    return {
        "schema": SEED_MEAN_RECEIPT_SCHEMA,
        "status": "all_26_neural_seed_means_exact_and_full_forecast_family_hash_valid",
        "sample_role": contract["claim_boundary"]["sample_role"],
        "not_confirmatory": True,
        "contract_sha256": core.sha256_file(core.CONTRACT_PATH),
        "components_sha256": core.sha256_file(components_path),
        "forecast_run_manifest_sha256": core.sha256_file(root / "run_manifest.json"),
        "forecast_manifest_sha256": core.sha256_file(root / "forecast_manifest.csv"),
        "forecast_artifact_family_sha256": _manifest_family_sha256(manifest),
        "forecast_artifact_count": int(len(manifest)),
        "pair_axis_sha256": history.pair_axis_sha256(components.pair_ids),
        "frozen_source_sha256": frozen_hashes,
        "frozen_source_bundle_sha256": history.bundle_sha256(frozen_hashes),
        "history_source_sha256": history_hashes,
        "history_source_bundle_sha256": history.bundle_sha256(history_hashes),
        "verification_adapter_sha256": _adapter_sha256(),
        "fixed_seeds": list(history.frozen_models.FIXED_SEEDS),
        "verified_group_count": expected_groups,
        "groups": groups,
        "performance_metrics_computed": False,
        "confirmation_parsed": False,
    }


def write_seed_mean_receipt(
    *, components_path: Path, forecast_output: Path, output: Path
) -> dict[str, Any]:
    receipt = build_seed_mean_receipt(
        components_path=components_path, forecast_output=forecast_output
    )
    history._atomic_json(output, receipt)
    return receipt


def validate_seed_mean_receipt(
    *, components_path: Path, forecast_output: Path, receipt_path: Path
) -> dict[str, Any]:
    try:
        observed = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid history seed-mean receipt") from error
    expected = build_seed_mean_receipt(
        components_path=components_path, forecast_output=forecast_output
    )
    if observed != expected:
        differing = sorted(
            key for key in set(observed) | set(expected) if observed.get(key) != expected.get(key)
        )
        raise ValueError(f"history seed-mean receipt differs: {differing}")
    return observed


def _calculation_daily_attribution(ledger: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expose the frozen hourly-gross sum while retaining compounded gross."""

    daily, daily_coin = V1_CONTINUOUS_DAILY_ATTRIBUTION(ledger)
    clock = pd.DatetimeIndex(ledger.bars.index).tz_convert("UTC")
    keys = clock.normalize()
    raw_gross = ledger.bars["gross_return"].groupby(keys).sum().reindex(daily.index)
    if raw_gross.isna().any() or not np.isfinite(raw_gross.to_numpy(dtype=float)).all():
        raise ValueError("hourly-gross daily numerator is incomplete")
    daily["daily_compounded_gross_return"] = daily["gross_return"].to_numpy(dtype=float)
    daily["hourly_gross_return_sum"] = raw_gross.to_numpy(dtype=float)
    # The unchanged evaluator passes this column only to the break-even helper;
    # return, volatility, Sharpe and drawdown continue to use daily net return.
    daily["gross_return"] = daily["hourly_gross_return_sum"]
    return daily, daily_coin


def _restore_public_daily_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "gross_return",
        "daily_compounded_gross_return",
        "hourly_gross_return_sum",
    }
    if not required.issubset(frame.columns):
        raise ValueError("internal daily table lacks v2 break-even columns")
    result = frame.copy()
    if not np.allclose(
        result["gross_return"], result["hourly_gross_return_sum"], rtol=0.0, atol=0.0
    ):
        raise ValueError("internal evaluator did not use hourly-gross break-even input")
    result["gross_return"] = result["daily_compounded_gross_return"]
    result = result.drop(columns=["daily_compounded_gross_return"])
    return result


def _atomic_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Rename a directory without replacing a concurrently created target."""

    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            at_fdcwd = -100
            rename_noreplace = 1
            status = renameat2(
                at_fdcwd,
                os.fsencode(source),
                at_fdcwd,
                os.fsencode(destination),
                rename_noreplace,
            )
            if status == 0:
                return
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(destination)
            if error not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
                raise OSError(error, os.strerror(error), destination)
    # Portable fallback.  The create-only lock excludes another v2 publisher;
    # the immediate existence check also refuses a pre-existing destination.
    if destination.exists():
        raise FileExistsError(destination)
    os.rename(source, destination)


class _StagedDirectory:
    """Publish a directory with one same-filesystem rename after full success."""

    def __init__(self, destination: Path):
        self.destination = Path(destination).expanduser().resolve()
        self.scratch: Path | None = None
        self.publish: Path | None = None
        self.lock: Path | None = None

    def __enter__(self) -> Path:
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.destination.parent / f".{self.destination.name}.publish.lock"
        try:
            descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise FileExistsError(f"history evaluation publication is locked: {self.lock}") from error
        else:
            os.close(descriptor)
        try:
            self.scratch = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.destination.name}.staging-", dir=self.destination.parent
                )
            )
            self.publish = self.scratch / "publish"
            self.publish.mkdir()
        except BaseException:
            if self.scratch is not None:
                shutil.rmtree(self.scratch, ignore_errors=True)
            self.lock.unlink(missing_ok=True)
            raise
        return self.publish

    def __exit__(self, error_type: object, error: object, traceback: object) -> bool:
        assert self.scratch is not None and self.publish is not None and self.lock is not None
        try:
            if error_type is None:
                if self.destination.exists():
                    raise FileExistsError(self.destination)
                _rename_directory_no_replace(self.publish, self.destination)
        finally:
            shutil.rmtree(self.scratch, ignore_errors=True)
            self.lock.unlink(missing_ok=True)
        return False


def _run_internal_v1_evaluation(
    *,
    components_path: Path,
    economic_inputs: Path,
    economic_manifest: Path,
    forecast_output: Path,
    pairs_research_root: Path,
    output: Path,
) -> dict[str, Any]:
    original = history._continuous_daily_attribution
    history._continuous_daily_attribution = _calculation_daily_attribution
    try:
        arguments = SimpleNamespace(
            components=components_path,
            economic_inputs=economic_inputs,
            economic_manifest=economic_manifest,
            forecast_output=forecast_output,
            output=output,
            pairs_research_root=pairs_research_root,
        )
        with redirect_stdout(io.StringIO()):
            status = history.command_evaluate(arguments)
        if status != 0:
            raise RuntimeError(f"internal history evaluation returned {status}")
    finally:
        history._continuous_daily_attribution = original
    receipt_path = output / "completion_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("internal history evaluation receipt is invalid") from error
    if (
        receipt.get("schema") != history.EVALUATION_SCHEMA
        or receipt.get("status") != "complete_retrospective_continuous_history_evaluation"
        or receipt.get("constraint_violation_count") != 0
        or receipt.get("confirmation_parsed") is not False
    ):
        raise ValueError("internal history evaluation did not complete cleanly")
    expected_files = set(FINAL_TABLES) | {"completion_receipt.json"}
    if {path.name for path in output.iterdir() if path.is_file()} != expected_files:
        raise ValueError("internal history evaluation file family differs")
    return receipt


def command_evaluate_v2(args: argparse.Namespace) -> dict[str, Any]:
    seed_receipt = validate_seed_mean_receipt(
        components_path=args.components,
        forecast_output=args.forecast_output,
        receipt_path=args.seed_mean_receipt,
    )
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    with _StagedDirectory(destination) as publish:
        assert publish.parent.name.startswith(f".{destination.name}.staging-")
        internal = publish.parent / "internal_v1"
        internal_receipt = _run_internal_v1_evaluation(
            components_path=args.components,
            economic_inputs=args.economic_inputs,
            economic_manifest=args.economic_manifest,
            forecast_output=args.forecast_output,
            pairs_research_root=args.pairs_research_root,
            output=internal,
        )
        # Close the input/output TOCTOU window before publication.  This second
        # full reconstruction re-hashes all 117 artifacts and rechecks all 26
        # arithmetic means after the ledgers have finished.
        final_seed_receipt = validate_seed_mean_receipt(
            components_path=args.components,
            forecast_output=args.forecast_output,
            receipt_path=args.seed_mean_receipt,
        )
        if final_seed_receipt != seed_receipt:
            raise ValueError("history seed-mean family changed during evaluation")
        daily = pd.read_csv(internal / "continuous_daily_returns.csv")
        public_daily = _restore_public_daily_table(daily)
        history._atomic_csv(publish / "continuous_daily_returns.csv", public_daily)
        for name in V1_CSV_TABLES[1:] + V1_BINARY_TABLES:
            _atomic_copy(internal / name, publish / name)
        output_hashes = {name: core.sha256_file(publish / name) for name in FINAL_TABLES}
        contract = core.load_contract()
        frozen_hashes = history.validate_frozen_sources(contract)
        history_hashes = history.history_source_hashes()
        receipt = {
            "schema": EVALUATION_RECEIPT_SCHEMA,
            "status": "complete_atomic_retrospective_continuous_history_evaluation_v2",
            "sample_role": contract["claim_boundary"]["sample_role"],
            "not_confirmatory": True,
            "contract_sha256": core.sha256_file(core.CONTRACT_PATH),
            "components_sha256": core.sha256_file(args.components),
            "economic_inputs_sha256": core.sha256_file(args.economic_inputs),
            "economic_manifest_sha256": core.sha256_file(args.economic_manifest),
            "forecast_run_manifest_sha256": core.sha256_file(
                Path(args.forecast_output) / "run_manifest.json"
            ),
            "forecast_manifest_sha256": core.sha256_file(
                Path(args.forecast_output) / "forecast_manifest.csv"
            ),
            "seed_mean_verification_receipt_sha256": core.sha256_file(
                args.seed_mean_receipt
            ),
            "seed_mean_verified_group_count": seed_receipt["verified_group_count"],
            "evaluation_adapter_sha256": _adapter_sha256(),
            "history_source_sha256": history_hashes,
            "history_source_bundle_sha256": history.bundle_sha256(history_hashes),
            "frozen_source_sha256": frozen_hashes,
            "frozen_source_bundle_sha256": history.bundle_sha256(frozen_hashes),
            "audited_dependency_source_hashes": internal_receipt[
                "audited_dependency_source_hashes"
            ],
            "family": internal_receipt["family"],
            "clock": internal_receipt["clock"],
            "terminal_policy": internal_receipt["terminal_policy"],
            "constraint_violation_count": internal_receipt[
                "constraint_violation_count"
            ],
            "bootstrap_interpretation": internal_receipt[
                "bootstrap_interpretation"
            ],
            "break_even_formula": (
                "10000 * sum(hourly gross_return) / sum(hourly turnover); "
                "daily net returns remain the inputs to return, volatility, Sharpe and drawdown"
            ),
            "continuous_daily_added_column": "hourly_gross_return_sum",
            "forecast_family_revalidated_after_ledgers": True,
            "v1_output_table_family_preserved": list(FINAL_TABLES),
            "atomic_publication": "same_filesystem_directory_rename_after_all_hashes_and_receipt",
            "output_sha256": output_hashes,
            "confirmation_authorized": False,
            "confirmation_parsed": False,
        }
        history._atomic_json(publish / "completion_receipt.json", receipt)
    return receipt


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-seed-means")
    verify.add_argument("--components", type=Path, required=True)
    verify.add_argument("--forecast-output", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--components", type=Path, required=True)
    evaluate.add_argument("--economic-inputs", type=Path, required=True)
    evaluate.add_argument("--economic-manifest", type=Path, required=True)
    evaluate.add_argument("--forecast-output", type=Path, required=True)
    evaluate.add_argument("--seed-mean-receipt", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--pairs-research-root",
        type=Path,
        default=history.frozen_evaluation.DEFAULT_AUDITED_ROOT,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    if args.command == "verify-seed-means":
        result = write_seed_mean_receipt(
            components_path=args.components,
            forecast_output=args.forecast_output,
            output=args.output,
        )
    else:
        result = command_evaluate_v2(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
