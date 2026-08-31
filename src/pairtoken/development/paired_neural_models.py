"""Matched endpoint, LSTM, and iTransformer models for the paired experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paired_neural_core import (
    ENDPOINT_HISTORY_HOURS,
    PAIR_PATH_DIMENSION,
    RAW_DIMENSION,
    PairedComponents,
)


HERE = Path(__file__).resolve().parent
PARENT_MODEL_PATH = HERE.parent / "model" / "pairtoken_model.py"
EXPECTED_PARENT_MODEL_SHA256 = (
    "649846a74e14058d9e764ccea1de482edad57087053c59aed8b1066fd05c4da7"
)
AUXILIARY_DIMENSION = 84
FIXED_SEEDS = (1701, 1702, 1703)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_parent_model() -> Any:
    if sha256_file(PARENT_MODEL_PATH) != EXPECTED_PARENT_MODEL_SHA256:
        raise RuntimeError("frozen parent cycle-model source hash differs")
    name = "paired_neural_frozen_cycle_pilot"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PARENT_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(PARENT_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cycle = _load_parent_model()


@dataclass(frozen=True)
class MaterializedPanel:
    """One fold and one representation in the parent trainer interface."""

    clock: pd.DatetimeIndex
    pair_ids: tuple[str, ...]
    raw_features: np.ndarray
    zero_auxiliary: np.ndarray
    feature_available: np.ndarray
    target_cashflows: np.ndarray
    target_available: np.ndarray

    @property
    def label_mask(self) -> np.ndarray:
        return self.feature_available & self.target_available

    def auxiliary(self, arm: str) -> np.ndarray:
        if arm != "raw":
            raise KeyError("paired experiment exposes only the raw model arm")
        return self.zero_auxiliary


@dataclass(frozen=True)
class LocalFold:
    name: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    source_indices: np.ndarray


def make_walk_forward_fold(
    components: PairedComponents,
    *,
    name: str,
    test_start: object,
    test_end_exclusive: object,
    training_days: int = 365,
    validation_days: int = 90,
    purge_hours: int = 1,
) -> Any:
    return cycle.make_pilot_fold(
        components.clock,
        name=name,
        test_start=test_start,
        test_end_exclusive=test_end_exclusive,
        training_days=int(training_days),
        validation_days=int(validation_days),
        purge_hours=int(purge_hours),
    )


def materialize_fold(
    components: PairedComponents,
    fold: Any,
    *,
    representation: str,
) -> tuple[MaterializedPanel, Any, LocalFold]:
    """Materialize only the rows needed by one fold.

    This is the scale boundary: the persistent cache stays coin-factorized;
    pair features exist only for a single fit and can be released afterward.
    """

    source = np.concatenate(
        (fold.train_indices, fold.validation_indices, fold.test_indices)
    ).astype(np.int64)
    if len(np.unique(source)) != len(source) or np.any(np.diff(source) <= 0):
        raise ValueError("fold source rows must be disjoint and chronological")
    raw = components.features(source, representation=representation)
    zeros = np.zeros(
        (len(source), len(components.pair_ids), AUXILIARY_DIMENSION),
        dtype=np.float32,
    )
    n_train = len(fold.train_indices)
    n_validation = len(fold.validation_indices)
    train = np.arange(0, n_train, dtype=np.int64)
    validation = np.arange(n_train, n_train + n_validation, dtype=np.int64)
    test = np.arange(n_train + n_validation, len(source), dtype=np.int64)
    local_fold = cycle.PilotFold(str(fold.name), train, validation, test)
    panel = MaterializedPanel(
        clock=components.clock.take(source),
        pair_ids=components.pair_ids,
        raw_features=raw,
        zero_auxiliary=zeros,
        feature_available=components.feature_available[source],
        target_cashflows=components.target_cashflows[source],
        target_available=components.target_available[source],
    )
    return panel, local_fold, LocalFold(str(fold.name), train, validation, test, source)


@dataclass(frozen=True)
class FittedEndpointRidge:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float
    coefficients: np.ndarray
    ridge_penalty: float

    def predict(self, raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float64)[..., PAIR_PATH_DIMENSION:]
        standardized = (values - self.feature_mean) / self.feature_scale
        prediction = standardized @ self.coefficients
        prediction = prediction * self.target_scale + self.target_mean
        return np.where(np.asarray(mask, dtype=bool), prediction, np.nan)


def fit_endpoint_ridge(
    panel: MaterializedPanel,
    fold: Any,
    *,
    ridge_penalty: float = 10.0,
    chunk_rows: int = 512,
) -> FittedEndpointRidge:
    """Fit the shared order-free benchmark using streaming sufficient stats."""

    train = np.asarray(fold.train_indices, dtype=np.int64)
    feature_dimension = RAW_DIMENSION - PAIR_PATH_DIMENSION
    sum_x = np.zeros(feature_dimension, dtype=np.float64)
    sum_x2 = np.zeros(feature_dimension, dtype=np.float64)
    sum_y = 0.0
    sum_y2 = 0.0
    count_features = 0
    count_labels = 0
    for first in range(0, len(train), int(chunk_rows)):
        rows = train[first : first + int(chunk_rows)]
        x = panel.raw_features[rows, :, PAIR_PATH_DIMENSION:]
        feature_mask = panel.feature_available[rows]
        observed_x = x[feature_mask].astype(np.float64)
        sum_x += observed_x.sum(axis=0)
        sum_x2 += np.square(observed_x).sum(axis=0)
        count_features += len(observed_x)
        label_mask = panel.label_mask[rows]
        observed_y = panel.target_cashflows[rows][label_mask].astype(np.float64)
        sum_y += float(observed_y.sum())
        sum_y2 += float(np.square(observed_y).sum())
        count_labels += len(observed_y)
    if count_features < 2 or count_labels < 2:
        raise ValueError("endpoint ridge requires training features and labels")
    mean_x = sum_x / count_features
    variance_x = np.maximum(sum_x2 / count_features - np.square(mean_x), 0.0)
    scale_x = np.where(variance_x > 1e-16, np.sqrt(variance_x), 1.0)
    mean_y = sum_y / count_labels
    variance_y = max(sum_y2 / count_labels - mean_y * mean_y, 0.0)
    scale_y = np.sqrt(variance_y) if variance_y > 1e-16 else 1.0
    xtx = np.zeros((feature_dimension, feature_dimension), dtype=np.float64)
    xty = np.zeros(feature_dimension, dtype=np.float64)
    for first in range(0, len(train), int(chunk_rows)):
        rows = train[first : first + int(chunk_rows)]
        label_mask = panel.label_mask[rows]
        x = panel.raw_features[rows, :, PAIR_PATH_DIMENSION:][label_mask].astype(np.float64)
        y = panel.target_cashflows[rows][label_mask].astype(np.float64)
        standardized_x = (x - mean_x) / scale_x
        standardized_y = (y - mean_y) / scale_y
        xtx += standardized_x.T @ standardized_x
        xty += standardized_x.T @ standardized_y
    penalty = float(ridge_penalty)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("ridge penalty must be positive")
    coefficients = np.linalg.solve(xtx + penalty * np.eye(feature_dimension), xty)
    return FittedEndpointRidge(
        mean_x.astype(np.float32),
        scale_x.astype(np.float32),
        float(mean_y),
        float(scale_y),
        coefficients.astype(np.float32),
        penalty,
    )


@dataclass(frozen=True)
class LSTMConfig:
    path_hidden_size: int = 16
    path_layers: int = 1
    common_tail_projection: int = 32
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    maximum_epochs: int = 18
    early_stopping_patience: int = 4
    minimum_improvement: float = 1e-6
    seed: int = 1701
    device: str = "cpu"


class SharedPairPathLSTM(nn.Module):
    """Small shared per-pair LSTM with the common endpoint/state tail."""

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.path_encoder = nn.LSTM(
            input_size=1,
            hidden_size=int(config.path_hidden_size),
            num_layers=int(config.path_layers),
            batch_first=True,
            dropout=0.0,
        )
        tail_dimension = RAW_DIMENSION - PAIR_PATH_DIMENSION
        self.tail_projection = nn.Sequential(
            nn.Linear(tail_dimension, int(config.common_tail_projection)),
            nn.GELU(),
            nn.LayerNorm(int(config.common_tail_projection)),
        )
        self.output_head = nn.Linear(
            int(config.path_hidden_size) + int(config.common_tail_projection), 1
        )

    def forward(self, raw: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 3 or raw.shape[-1] != RAW_DIMENSION:
            raise ValueError("raw tensor must have shape (batch,pairs,232)")
        if token_mask.shape != raw.shape[:2]:
            raise ValueError("token mask has wrong axes")
        batch, pairs, _ = raw.shape
        path = raw[:, :, :PAIR_PATH_DIMENSION].reshape(batch * pairs, PAIR_PATH_DIMENSION, 1)
        _, (hidden, _) = self.path_encoder(path)
        path_state = hidden[-1].reshape(batch, pairs, -1)
        tail_state = self.tail_projection(raw[:, :, PAIR_PATH_DIMENSION:])
        prediction = self.output_head(torch.cat((path_state, tail_state), dim=-1)).squeeze(-1)
        return prediction.masked_fill(~token_mask.to(dtype=torch.bool), 0.0)


@dataclass(frozen=True)
class FittedLSTM:
    model: SharedPairPathLSTM
    scaler: Any
    config: LSTMConfig
    epochs_trained: int
    best_validation_loss: float
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]

    def predict(self, raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
        zeros = np.zeros((*raw.shape[:2], AUXILIARY_DIMENSION), dtype=np.float32)
        scaled, _ = self.scaler.transform_features(raw, zeros, mask)
        device = torch.device(self.config.device)
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for first in range(0, len(scaled), self.config.batch_size):
                stop = min(first + self.config.batch_size, len(scaled))
                value = self.model(
                    torch.from_numpy(scaled[first:stop]).to(device),
                    torch.from_numpy(np.asarray(mask[first:stop], dtype=bool)).to(device),
                )
                outputs.append(value.cpu().numpy())
        standardized = np.concatenate(outputs, axis=0)
        predictions = self.scaler.inverse_targets(standardized)
        return np.where(mask, predictions, np.nan)


def fit_lstm(panel: MaterializedPanel, fold: Any, *, config: LSTMConfig) -> FittedLSTM:
    train = fold.train_indices
    validation = fold.validation_indices
    scaler = cycle.FeatureScaler.fit(
        panel.raw_features[train],
        panel.zero_auxiliary[train],
        panel.target_cashflows[train],
        panel.feature_available[train],
        panel.label_mask[train],
    )

    def tensors(indices: np.ndarray) -> tuple[torch.Tensor, ...]:
        raw, _ = scaler.transform_features(
            panel.raw_features[indices],
            panel.zero_auxiliary[indices],
            panel.feature_available[indices],
        )
        target = scaler.transform_targets(panel.target_cashflows[indices], panel.label_mask[indices])
        return (
            torch.from_numpy(raw),
            torch.from_numpy(panel.feature_available[indices]),
            torch.from_numpy(target),
            torch.from_numpy(panel.label_mask[indices]),
        )

    cycle.set_reproducible_seed(int(config.seed))
    device = torch.device(config.device)
    model = SharedPairPathLSTM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(int(config.seed))
    train_loader = DataLoader(
        TensorDataset(*tensors(train)),
        batch_size=int(config.batch_size),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    validation_loader = DataLoader(
        TensorDataset(*tensors(validation)),
        batch_size=int(config.batch_size),
        shuffle=False,
        drop_last=False,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    training_losses: list[float] = []
    validation_losses: list[float] = []
    for _ in range(int(config.maximum_epochs)):
        model.train()
        loss_sum = 0.0
        labels = 0
        for raw, token_mask, target, label_mask in train_loader:
            count = int(label_mask.sum().item())
            if count == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(raw.to(device), token_mask.to(device))
            loss = cycle.masked_smooth_l1(
                prediction, target.to(device), label_mask.to(device)
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * count
            labels += count
        if labels == 0:
            raise ValueError("LSTM training epoch contains no labels")
        training_losses.append(loss_sum / labels)
        model.eval()
        validation_sum = 0.0
        validation_labels = 0
        with torch.no_grad():
            for raw, token_mask, target, label_mask in validation_loader:
                count = int(label_mask.sum().item())
                if count == 0:
                    continue
                prediction = model(raw.to(device), token_mask.to(device))
                loss = cycle.masked_smooth_l1(
                    prediction, target.to(device), label_mask.to(device)
                )
                validation_sum += float(loss.detach().cpu()) * count
                validation_labels += count
        if validation_labels == 0:
            raise ValueError("LSTM validation epoch contains no labels")
        validation_loss = validation_sum / validation_labels
        validation_losses.append(validation_loss)
        if validation_loss < best_loss - float(config.minimum_improvement):
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(config.early_stopping_patience):
                break
    if best_state is None:
        raise RuntimeError("LSTM training produced no validation state")
    model.load_state_dict(best_state)
    model.eval()
    return FittedLSTM(
        model,
        scaler,
        config,
        len(training_losses),
        float(best_loss),
        tuple(training_losses),
        tuple(validation_losses),
    )


def transformer_config(
    *,
    seed: int,
    device: str,
    maximum_epochs: int = 18,
    early_stopping_patience: int = 4,
    batch_size: int = 32,
) -> Any:
    return cycle.TransformerConfig(
        raw_dimension=RAW_DIMENSION,
        auxiliary_dimension=AUXILIARY_DIMENSION,
        d_model=48,
        n_heads=4,
        encoder_layers=1,
        feedforward_dimension=96,
        dropout=0.1,
        learning_rate=8e-4,
        weight_decay=1e-4,
        batch_size=int(batch_size),
        maximum_epochs=int(maximum_epochs),
        early_stopping_patience=int(early_stopping_patience),
        minimum_improvement=1e-6,
        seed=int(seed),
        device=str(device),
    )


def fit_itransformer(panel: MaterializedPanel, fold: Any, *, config: Any) -> Any:
    return cycle.fit_transformer(panel, fold, arm="raw", config=config)


def mean_seed_predictions(predictions: Mapping[int, np.ndarray]) -> np.ndarray:
    if tuple(sorted(map(int, predictions))) != FIXED_SEEDS:
        raise ValueError(f"predictions must contain the exact fixed seeds {FIXED_SEEDS}")
    shapes = {np.asarray(value).shape for value in predictions.values()}
    if len(shapes) != 1:
        raise ValueError("seed prediction axes differ")
    stacked = np.stack([np.asarray(predictions[seed], dtype=np.float64) for seed in FIXED_SEEDS])
    all_nan = np.isnan(stacked).all(axis=0)
    if np.any(np.isnan(stacked) != all_nan[None, ...]):
        raise ValueError("seed missingness differs")
    result = np.where(all_nan, 0.0, np.where(np.isnan(stacked), 0.0, stacked).sum(axis=0) / 3.0)
    return np.where(all_nan, np.nan, result).astype(np.float32)


def decision_ic(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return one cross-sectional Spearman IC per decision (NaN if <3 pairs)."""

    values = np.asarray(predictions, dtype=np.float64)
    response = np.asarray(targets, dtype=np.float64)
    observed = np.asarray(mask, dtype=bool)
    result = np.full(len(values), np.nan, dtype=np.float64)
    for row in range(len(values)):
        chosen = observed[row] & np.isfinite(values[row]) & np.isfinite(response[row])
        if int(chosen.sum()) < 3:
            continue
        x = pd.Series(values[row, chosen]).rank(method="average").to_numpy()
        y = pd.Series(response[row, chosen]).rank(method="average").to_numpy()
        if np.std(x) > 0.0 and np.std(y) > 0.0:
            result[row] = float(np.corrcoef(x, y)[0, 1])
    return result


__all__ = [
    "FIXED_SEEDS",
    "FittedEndpointRidge",
    "LSTMConfig",
    "MaterializedPanel",
    "SharedPairPathLSTM",
    "decision_ic",
    "fit_endpoint_ridge",
    "fit_itransformer",
    "fit_lstm",
    "make_walk_forward_fold",
    "materialize_fold",
    "mean_seed_predictions",
    "transformer_config",
]
