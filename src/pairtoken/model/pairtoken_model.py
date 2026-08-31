"""Pure components for the governed eight-hour crypto development pilot.

The module deliberately contains no exchange client and no hidden path to the
older research repository.  It accepts already constructed causal pair
episodes and current-decision hedge/availability state, then builds features,
fits the matched pair-token models and produces pair targets.  The runner is
responsible for the audited pair-to-coin projection and execution ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

BARS_PER_CYCLE = 32
CYCLE_HOURS = 8
SIGNATURE_LEVEL = 3
SIGNATURE_DIMENSION = 84
CONVENTIONAL_DIMENSION = 84
AUXILIARY_ARMS = (
    "raw",
    "conventional",
    "signature",
    "permuted_signature",
)
_SCALE_FLOOR = 1e-8


def _utc_index(values: Sequence[object] | pd.DatetimeIndex, *, name: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    index = index.rename(name)
    return index


def _readonly(values: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def normalized_time_residual_paths(increments: np.ndarray) -> np.ndarray:
    """Create time-augmented paths after safe pathwise quadratic scaling.

    The completed cycle itself supplies the scale, which is known at the next
    cycle's decision.  The raw branch retains the unscaled increments, while
    the Signature branch focuses on path shape.  A zero path remains zero.
    """

    values = np.asarray(increments, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != BARS_PER_CYCLE:
        raise ValueError("increments must have shape (episodes, 32)")
    if not np.isfinite(values).all():
        raise ValueError("increments must be finite")
    quadratic_scale = np.sqrt(np.square(values).sum(axis=1))
    safe_scale = np.where(quadratic_scale > _SCALE_FLOOR, quadratic_scale, 1.0)
    normalized = values / safe_scale[:, None]
    cumulative = np.concatenate(
        (np.zeros((len(values), 1), dtype=np.float64), np.cumsum(normalized, axis=1)),
        axis=1,
    )
    time = np.linspace(0.0, 1.0, BARS_PER_CYCLE + 1, dtype=np.float64)
    paths = np.empty((len(values), BARS_PER_CYCLE + 1, 2), dtype=np.float64)
    paths[:, :, 0] = time[None, :]
    paths[:, :, 1] = cumulative
    return paths


def signature_features(increments: np.ndarray, *, chunk_size: int = 4096) -> np.ndarray:
    """Return the 84 non-constant level-three Hoff lead-lag coordinates."""

    values = np.asarray(increments, dtype=np.float64)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    try:
        import iisignature as fast_signature  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - exercised when the extension is absent
        fast_signature = None
        # Keep the model-training path independent of the larger thesis stack.
        # This transparent fallback is imported only when path features are
        # actually constructed, never when a frozen feature cache is trained.
        from src.allocation.signature_kernel import (  # noqa: PLC0415
            signature_feature_matrix as reference_signature_feature_matrix,
        )
    blocks: list[np.ndarray] = []
    for start in range(0, len(values), int(chunk_size)):
        paths = normalized_time_residual_paths(values[start : start + chunk_size])
        if fast_signature is None:
            full_block, _, _ = reference_signature_feature_matrix(
                paths, level=SIGNATURE_LEVEL
            )
            block = full_block[:, 1:]
        else:
            # Exact same canonical Hoff lift as signature_kernel.py.  The C
            # extension evaluates the batch efficiently and, in the focused
            # equivalence test, agrees with the transparent Chen recursion to
            # approximately 1e-14.
            repeated = np.repeat(paths, repeats=4, axis=1)
            lifted = np.concatenate((repeated[:, :-5], repeated[:, 5:]), axis=2)
            block = fast_signature.sig(lifted, SIGNATURE_LEVEL)
        if block.shape[1] != SIGNATURE_DIMENSION:
            raise AssertionError("the frozen Hoff level-three map must have 84 coordinates")
        blocks.append(np.asarray(block, dtype=np.float32))
    if not blocks:
        return np.empty((0, SIGNATURE_DIMENSION), dtype=np.float32)
    result = np.concatenate(blocks, axis=0)
    if not np.isfinite(result).all():
        raise FloatingPointError("Signature features are non-finite")
    return result


def stable_endpoint_preserving_permutations(
    increments: np.ndarray,
    *,
    pair_ids: Sequence[str],
    anchors: Sequence[object] | pd.DatetimeIndex,
    seed: int,
) -> np.ndarray:
    """Stably shuffle each observed cycle without changing its endpoint.

    The key contains only the declared seed, pair and already completed source
    anchor.  The transformation therefore cannot consult a target cycle.
    """

    values = np.asarray(increments, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != BARS_PER_CYCLE:
        raise ValueError("increments must have shape (episodes, 32)")
    names = tuple(str(value) for value in pair_ids)
    times = _utc_index(anchors, name="anchor")
    if len(names) != len(values) or len(times) != len(values):
        raise ValueError("permutation keys must align with increments")
    result = np.empty_like(values)
    for row, (pair_id, anchor) in enumerate(zip(names, times, strict=True)):
        payload = f"{int(seed)}|{pair_id}|{int(anchor.value)}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        row_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        permutation = np.random.default_rng(row_seed).permutation(BARS_PER_CYCLE)
        result[row] = values[row, permutation]
    if not np.allclose(result.sum(axis=1), values.sum(axis=1), atol=1e-14, rtol=1e-13):
        raise AssertionError("control permutation changed a path endpoint")
    return result


def conventional_path_features(increments: np.ndarray) -> np.ndarray:
    """Return an 84-coordinate transparent, order-aware comparison map.

    Coordinates are 32 reverse-chronological raw increments, 32
    reverse-chronological cumulative states, four summaries at each of the
    4/8/16/32-bar horizons, and four full-cycle autocorrelations.  The map has
    the same nominal dimension as the Signature branch and no fitted values.
    """

    values = np.asarray(increments, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != BARS_PER_CYCLE:
        raise ValueError("increments must have shape (episodes, 32)")
    if not np.isfinite(values).all():
        raise ValueError("increments must be finite")
    cumulative = np.cumsum(values, axis=1)
    blocks: list[np.ndarray] = [values[:, ::-1], cumulative[:, ::-1]]
    for horizon in (4, 8, 16, 32):
        window = values[:, -horizon:]
        momentum = window.sum(axis=1)
        realized = np.sqrt(np.square(window).sum(axis=1))
        downside = np.sqrt(np.square(np.minimum(window, 0.0)).sum(axis=1))
        path = np.concatenate(
            (np.zeros((len(window), 1)), np.cumsum(window, axis=1)), axis=1
        )
        running_peak = np.maximum.accumulate(path, axis=1)
        maximum_drawdown = (running_peak - path).max(axis=1)
        blocks.append(np.column_stack((momentum, realized, downside, maximum_drawdown)))
    centered = values - values.mean(axis=1, keepdims=True)
    denominator = np.square(centered).sum(axis=1)
    autocorrelations: list[np.ndarray] = []
    for lag in (1, 2, 4, 8):
        numerator = (centered[:, lag:] * centered[:, :-lag]).sum(axis=1)
        autocorrelations.append(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > _SCALE_FLOOR,
            )
        )
    blocks.append(np.column_stack(autocorrelations))
    result = np.column_stack(blocks)
    if result.shape != (len(values), CONVENTIONAL_DIMENSION):
        raise AssertionError("the frozen conventional map must have 84 coordinates")
    if not np.isfinite(result).all():
        raise FloatingPointError("conventional features are non-finite")
    return np.asarray(result, dtype=np.float32)


@dataclass(frozen=True)
class CyclePanel:
    """Dense decision panel with separate feature and response availability."""

    clock: pd.DatetimeIndex
    pair_ids: tuple[str, ...]
    raw_features: np.ndarray
    conventional_features: np.ndarray
    signature_features: np.ndarray
    permuted_signature_features: np.ndarray
    feature_available: np.ndarray
    target_cashflows: np.ndarray
    target_available: np.ndarray

    def __post_init__(self) -> None:
        clock = _utc_index(self.clock, name="decision_at")
        pairs = tuple(str(value) for value in self.pair_ids)
        if not clock.is_monotonic_increasing or not clock.is_unique:
            raise ValueError("clock must be unique and increasing")
        if len(clock) and (clock.asi8 % pd.Timedelta(hours=CYCLE_HOURS).value != 0).any():
            raise ValueError("clock must lie on the eight-hour UTC grid")
        if not pairs or len(pairs) != len(set(pairs)):
            raise ValueError("pair_ids must be non-empty and unique")
        rows, pair_count = len(clock), len(pairs)
        raw = np.asarray(self.raw_features, dtype=np.float32)
        conventional = np.asarray(self.conventional_features, dtype=np.float32)
        ordinary = np.asarray(self.signature_features, dtype=np.float32)
        permuted = np.asarray(self.permuted_signature_features, dtype=np.float32)
        feature_available = np.asarray(self.feature_available, dtype=bool)
        targets = np.asarray(self.target_cashflows, dtype=np.float32)
        target_available = np.asarray(self.target_available, dtype=bool)
        if raw.ndim != 3 or raw.shape[:2] != (rows, pair_count):
            raise ValueError("raw_features do not share the panel axes")
        expected_aux = (rows, pair_count, SIGNATURE_DIMENSION)
        if any(value.shape != expected_aux for value in (conventional, ordinary, permuted)):
            raise ValueError("auxiliary feature arrays do not share the frozen axes")
        if feature_available.shape != (rows, pair_count):
            raise ValueError("feature_available does not share the panel axes")
        if targets.shape != (rows, pair_count) or target_available.shape != targets.shape:
            raise ValueError("target arrays do not share the panel axes")
        for values, label in (
            (raw, "raw"),
            (conventional, "conventional"),
            (ordinary, "signature"),
            (permuted, "permuted signature"),
        ):
            if not np.isfinite(values[feature_available]).all():
                raise ValueError(f"available {label} features must be finite")
        if not np.isfinite(targets[target_available]).all():
            raise ValueError("available targets must be finite")
        # Target availability is outcome-side and may be false even when a
        # causal decision was available.  It must never widen the feature mask.
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "pair_ids", pairs)
        object.__setattr__(self, "raw_features", _readonly(raw, dtype=np.float32))
        object.__setattr__(self, "conventional_features", _readonly(conventional, dtype=np.float32))
        object.__setattr__(self, "signature_features", _readonly(ordinary, dtype=np.float32))
        object.__setattr__(self, "permuted_signature_features", _readonly(permuted, dtype=np.float32))
        object.__setattr__(self, "feature_available", _readonly(feature_available, dtype=bool))
        object.__setattr__(self, "target_cashflows", _readonly(targets, dtype=np.float32))
        object.__setattr__(self, "target_available", _readonly(target_available, dtype=bool))

    @property
    def label_mask(self) -> np.ndarray:
        return self.feature_available & self.target_available

    def auxiliary(self, arm: str) -> np.ndarray:
        if arm == "raw":
            return np.zeros_like(self.signature_features)
        if arm == "conventional":
            return self.conventional_features
        if arm == "signature":
            return self.signature_features
        if arm == "permuted_signature":
            return self.permuted_signature_features
        raise KeyError(f"unknown pilot arm {arm!r}")


def build_cycle_panel(
    *,
    pair_ids: Sequence[str],
    episode_pair_ids: Sequence[str],
    episode_anchors: Sequence[object] | pd.DatetimeIndex,
    residual_increments: np.ndarray,
    target_cashflows: np.ndarray,
    clock: Sequence[object] | pd.DatetimeIndex,
    current_pair_available: np.ndarray,
    current_pair_weights: np.ndarray,
    history_cycles: int = 21,
    control_seed: int = 0x13527D6,
    signature_chunk_size: int = 4096,
) -> CyclePanel:
    """Build decisions at `t` from complete cycles ending at `t`.

    `target_cashflows` belongs to the episode identified by the same
    `(pair_id, anchor)` row.  Features at anchor `t` use episodes `t-h,...,t-1`
    and the response uses episode `t`.  Current availability and hedge weights
    are supplied separately because conditioning them on a complete future
    response episode would create delisting/coverage look-ahead.
    """

    pairs = tuple(str(value) for value in pair_ids)
    if history_cycles < 2:
        raise ValueError("history_cycles must be at least two")
    decisions = _utc_index(clock, name="decision_at")
    if len(decisions) < history_cycles + 1:
        raise ValueError("decision clock is too short for the cycle history")
    if not decisions.is_unique or not decisions.is_monotonic_increasing:
        raise ValueError("decision clock must be unique and increasing")
    expected = pd.date_range(decisions[0], decisions[-1], freq="8h", tz="UTC")
    if not decisions.equals(expected.rename("decision_at")):
        raise ValueError("decision clock must be a dense eight-hour grid")
    episode_names = tuple(str(value) for value in episode_pair_ids)
    anchors = _utc_index(episode_anchors, name="anchor")
    increments = np.asarray(residual_increments, dtype=np.float64)
    cashflows = np.asarray(target_cashflows, dtype=np.float64)
    if increments.shape != (len(episode_names), BARS_PER_CYCLE):
        raise ValueError("episode residuals must have shape (episodes, 32)")
    if cashflows.shape != (len(episode_names),):
        raise ValueError("target_cashflows must have one value per episode")
    if len(anchors) != len(episode_names):
        raise ValueError("episode anchors do not align")
    if not np.isfinite(increments).all() or not np.isfinite(cashflows).all():
        raise ValueError("complete episode inputs must be finite")
    pair_lookup = {pair_id: position for position, pair_id in enumerate(pairs)}
    if set(episode_names).difference(pair_lookup):
        raise ValueError("episode family contains undeclared pairs")
    time_lookup = {value: position for position, value in enumerate(decisions.asi8)}
    keys = list(zip(episode_names, anchors.asi8, strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("episode pair/anchor keys must be unique")

    rows, pair_count = len(decisions), len(pairs)
    dense_residual = np.full((rows, pair_count, BARS_PER_CYCLE), np.nan, dtype=np.float32)
    dense_cashflow = np.full((rows, pair_count), np.nan, dtype=np.float32)
    episode_complete = np.zeros((rows, pair_count), dtype=bool)
    inside_episode_rows: list[int] = []
    inside_episode_pairs: list[int] = []
    inside_source_rows: list[int] = []
    for source_row, (pair_id, anchor_ns) in enumerate(keys):
        time_row = time_lookup.get(int(anchor_ns))
        if time_row is None:
            continue
        pair_row = pair_lookup[pair_id]
        dense_residual[time_row, pair_row] = increments[source_row]
        dense_cashflow[time_row, pair_row] = cashflows[source_row]
        episode_complete[time_row, pair_row] = True
        inside_episode_rows.append(time_row)
        inside_episode_pairs.append(pair_row)
        inside_source_rows.append(source_row)

    available = np.asarray(current_pair_available, dtype=bool)
    weights = np.asarray(current_pair_weights, dtype=np.float32)
    if available.shape != (rows, pair_count):
        raise ValueError("current_pair_available does not share the panel axes")
    if weights.shape != (rows, pair_count, 2):
        raise ValueError("current_pair_weights must have signal/hedge coordinates")
    if not np.isfinite(weights[available]).all():
        raise ValueError("available current pair weights must be finite")

    endpoint = np.nansum(dense_residual, axis=2)
    base_dimension = BARS_PER_CYCLE + history_cycles + 4
    raw = np.zeros((rows, pair_count, base_dimension), dtype=np.float32)
    feature_available = np.zeros((rows, pair_count), dtype=bool)
    source_path_row = np.full((rows, pair_count), -1, dtype=np.int64)
    for row in range(history_cycles, rows):
        history_complete = episode_complete[row - history_cycles : row].all(axis=0)
        usable = history_complete & available[row]
        feature_available[row] = usable
        if not usable.any():
            continue
        raw[row, usable, :BARS_PER_CYCLE] = dense_residual[row - 1, usable]
        raw[
            row,
            usable,
            BARS_PER_CYCLE : BARS_PER_CYCLE + history_cycles,
        ] = endpoint[row - history_cycles : row, usable].T
        offset = BARS_PER_CYCLE + history_cycles
        raw[row, usable, offset : offset + 2] = weights[row, usable]
        phase = int((decisions[row].hour // CYCLE_HOURS) % 3)
        angle = 2.0 * math.pi * phase / 3.0
        raw[row, usable, offset + 2] = math.sin(angle)
        raw[row, usable, offset + 3] = math.cos(angle)
        source_path_row[row, usable] = row - 1

    conventional = np.zeros((rows, pair_count, CONVENTIONAL_DIMENSION), dtype=np.float32)
    ordinary = np.zeros((rows, pair_count, SIGNATURE_DIMENSION), dtype=np.float32)
    permuted = np.zeros_like(ordinary)
    feature_rows, feature_pairs = np.nonzero(feature_available)
    previous_paths = dense_residual[feature_rows - 1, feature_pairs].astype(np.float64)
    conventional_values = conventional_path_features(previous_paths)
    ordinary_values = signature_features(previous_paths, chunk_size=signature_chunk_size)
    previous_pair_names = tuple(pairs[position] for position in feature_pairs)
    previous_anchors = decisions.take(feature_rows - 1)
    shuffled_paths = stable_endpoint_preserving_permutations(
        previous_paths,
        pair_ids=previous_pair_names,
        anchors=previous_anchors,
        seed=int(control_seed),
    )
    permuted_values = signature_features(shuffled_paths, chunk_size=signature_chunk_size)
    conventional[feature_rows, feature_pairs] = conventional_values
    ordinary[feature_rows, feature_pairs] = ordinary_values
    permuted[feature_rows, feature_pairs] = permuted_values

    target_available = episode_complete.copy()
    target_values = np.where(target_available, dense_cashflow, 0.0)
    return CyclePanel(
        clock=decisions,
        pair_ids=pairs,
        raw_features=raw,
        conventional_features=conventional,
        signature_features=ordinary,
        permuted_signature_features=permuted,
        feature_available=feature_available,
        target_cashflows=target_values,
        target_available=target_available,
    )


@dataclass(frozen=True)
class PilotFold:
    name: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray

    def __post_init__(self) -> None:
        train = np.asarray(self.train_indices, dtype=np.int64)
        validation = np.asarray(self.validation_indices, dtype=np.int64)
        test = np.asarray(self.test_indices, dtype=np.int64)
        if min(len(train), len(validation), len(test)) < 1:
            raise ValueError("every fold partition must be non-empty")
        if train[-1] >= validation[0] or validation[-1] >= test[0]:
            raise ValueError("fold partitions must be strictly chronological")
        object.__setattr__(self, "train_indices", _readonly(train, dtype=np.int64))
        object.__setattr__(self, "validation_indices", _readonly(validation, dtype=np.int64))
        object.__setattr__(self, "test_indices", _readonly(test, dtype=np.int64))


def make_pilot_fold(
    clock: pd.DatetimeIndex,
    *,
    name: str,
    test_start: object,
    test_end_exclusive: object,
    training_days: int = 365,
    validation_days: int = 90,
    purge_hours: int = 8,
) -> PilotFold:
    times = _utc_index(clock, name="decision_at")
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end_exclusive)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    purge = pd.Timedelta(hours=int(purge_hours))
    validation_end = start - purge
    validation_start = validation_end - pd.Timedelta(days=int(validation_days))
    training_end = validation_start - purge
    training_start = training_end - pd.Timedelta(days=int(training_days))

    def locations(first: pd.Timestamp, stop: pd.Timestamp) -> np.ndarray:
        return np.flatnonzero(np.asarray((times >= first) & (times < stop))).astype(np.int64)

    return PilotFold(
        name=str(name),
        train_indices=locations(training_start, training_end),
        validation_indices=locations(validation_start, validation_end),
        test_indices=locations(start, end),
    )


@dataclass(frozen=True)
class FeatureScaler:
    raw_mean: np.ndarray
    raw_scale: np.ndarray
    auxiliary_mean: np.ndarray
    auxiliary_scale: np.ndarray
    target_mean: float
    target_scale: float

    @classmethod
    def fit(
        cls,
        raw: np.ndarray,
        auxiliary: np.ndarray,
        targets: np.ndarray,
        feature_mask: np.ndarray,
        label_mask: np.ndarray,
    ) -> "FeatureScaler":
        feature = np.asarray(feature_mask, dtype=bool)
        labels = np.asarray(label_mask, dtype=bool)
        if not feature.any() or not labels.any():
            raise ValueError("training scaler requires features and labels")
        raw_rows = np.asarray(raw, dtype=np.float64)[feature]
        auxiliary_rows = np.asarray(auxiliary, dtype=np.float64)[feature]
        target_rows = np.asarray(targets, dtype=np.float64)[labels]

        def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mean = values.mean(axis=0)
            scale = values.std(axis=0, ddof=0)
            return mean, np.where(scale > _SCALE_FLOOR, scale, 1.0)

        raw_mean, raw_scale = moments(raw_rows)
        auxiliary_mean, auxiliary_scale = moments(auxiliary_rows)
        target_mean = float(target_rows.mean())
        raw_target_scale = float(target_rows.std(ddof=0))
        target_scale = raw_target_scale if raw_target_scale > _SCALE_FLOOR else 1.0
        return cls(
            raw_mean=_readonly(raw_mean, dtype=np.float32),
            raw_scale=_readonly(raw_scale, dtype=np.float32),
            auxiliary_mean=_readonly(auxiliary_mean, dtype=np.float32),
            auxiliary_scale=_readonly(auxiliary_scale, dtype=np.float32),
            target_mean=target_mean,
            target_scale=target_scale,
        )

    def transform_features(
        self,
        raw: np.ndarray,
        auxiliary: np.ndarray,
        feature_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = np.asarray(feature_mask, dtype=bool)
        raw_scaled = (np.asarray(raw, dtype=np.float32) - self.raw_mean) / self.raw_scale
        auxiliary_scaled = (
            np.asarray(auxiliary, dtype=np.float32) - self.auxiliary_mean
        ) / self.auxiliary_scale
        raw_scaled = np.where(mask[..., None], raw_scaled, 0.0).astype(np.float32)
        auxiliary_scaled = np.where(mask[..., None], auxiliary_scaled, 0.0).astype(np.float32)
        return raw_scaled, auxiliary_scaled

    def transform_targets(self, targets: np.ndarray, label_mask: np.ndarray) -> np.ndarray:
        values = (np.asarray(targets, dtype=np.float32) - self.target_mean) / self.target_scale
        return np.where(np.asarray(label_mask, dtype=bool), values, 0.0).astype(np.float32)

    def inverse_targets(self, predictions: np.ndarray) -> np.ndarray:
        return np.asarray(predictions, dtype=np.float32) * self.target_scale + self.target_mean


@dataclass(frozen=True)
class TransformerConfig:
    raw_dimension: int
    auxiliary_dimension: int = SIGNATURE_DIMENSION
    d_model: int = 48
    n_heads: int = 4
    encoder_layers: int = 1
    feedforward_dimension: int = 96
    dropout: float = 0.10
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    maximum_epochs: int = 18
    early_stopping_patience: int = 4
    minimum_improvement: float = 1e-6
    seed: int = 0x13527D6
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.raw_dimension < 1 or self.auxiliary_dimension < 1:
            raise ValueError("feature dimensions must be positive")
        if self.d_model < 4 or self.d_model % self.n_heads:
            raise ValueError("n_heads must divide d_model")
        if self.encoder_layers < 1 or self.feedforward_dimension < 2:
            raise ValueError("encoder dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")


class PairTokenITransformer(nn.Module):
    """Permutation-equivariant Transformer with pairs as attention tokens."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.raw_projection = nn.Linear(config.raw_dimension, config.d_model)
        self.auxiliary_projection = nn.Linear(
            config.auxiliary_dimension, config.d_model, bias=False
        )
        self.input_norm = nn.LayerNorm(config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward_dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )
        self.output_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        raw_features: torch.Tensor,
        auxiliary_features: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if raw_features.ndim != 3 or auxiliary_features.ndim != 3:
            raise ValueError("feature tensors must have shape (batch,pairs,features)")
        if raw_features.shape[:2] != auxiliary_features.shape[:2]:
            raise ValueError("raw and auxiliary pair axes differ")
        if token_mask.shape != raw_features.shape[:2]:
            raise ValueError("token_mask does not share the batch/pair axes")
        mask = token_mask.to(device=raw_features.device, dtype=torch.bool)
        clean_raw = torch.where(mask[..., None], raw_features, torch.zeros_like(raw_features))
        clean_auxiliary = torch.where(
            mask[..., None], auxiliary_features, torch.zeros_like(auxiliary_features)
        )
        tokens = self.input_norm(
            self.raw_projection(clean_raw) + self.auxiliary_projection(clean_auxiliary)
        )
        padding = ~mask
        all_missing = padding.all(dim=1)
        if bool(all_missing.any()):
            padding = padding.clone()
            padding[all_missing, 0] = False
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        predictions = self.output_head(encoded).squeeze(-1)
        return predictions.masked_fill(~mask, 0.0)


def masked_smooth_l1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    if predictions.shape != targets.shape or predictions.shape != label_mask.shape:
        raise ValueError("predictions, targets and label_mask must share a shape")
    mask = label_mask.to(device=predictions.device, dtype=torch.bool)
    if int(mask.sum().item()) == 0:
        return predictions.sum() * 0.0
    return F.smooth_l1_loss(predictions[mask], targets.to(predictions.device)[mask])


@dataclass(frozen=True)
class FittedTransformer:
    model: PairTokenITransformer
    scaler: FeatureScaler
    config: TransformerConfig
    arm: str
    epochs_trained: int
    best_validation_loss: float
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]

    def predict(
        self,
        raw: np.ndarray,
        auxiliary: np.ndarray,
        feature_mask: np.ndarray,
    ) -> np.ndarray:
        raw_scaled, auxiliary_scaled = self.scaler.transform_features(
            raw, auxiliary, feature_mask
        )
        mask = np.asarray(feature_mask, dtype=bool)
        device = torch.device(self.config.device)
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(raw_scaled), self.config.batch_size):
                stop = min(start + self.config.batch_size, len(raw_scaled))
                value = self.model(
                    torch.from_numpy(raw_scaled[start:stop]).to(device),
                    torch.from_numpy(auxiliary_scaled[start:stop]).to(device),
                    torch.from_numpy(mask[start:stop]).to(device),
                )
                outputs.append(value.cpu().numpy())
        standardized = np.concatenate(outputs, axis=0) if outputs else np.empty(mask.shape)
        prediction = self.scaler.inverse_targets(standardized)
        return np.where(mask, prediction, np.nan)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.cuda.is_available():
        # The fused CUDA attention kernels are faster but are not guaranteed to
        # have a deterministic backward pass.  The pilot is small enough that
        # the deterministic math implementation is the appropriate default.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def fit_transformer(
    panel: CyclePanel,
    fold: PilotFold,
    *,
    arm: str,
    config: TransformerConfig,
) -> FittedTransformer:
    if arm not in AUXILIARY_ARMS:
        raise KeyError(f"unknown pilot arm {arm!r}")
    auxiliary = panel.auxiliary(arm)
    train = fold.train_indices
    validation = fold.validation_indices
    scaler = FeatureScaler.fit(
        panel.raw_features[train],
        auxiliary[train],
        panel.target_cashflows[train],
        panel.feature_available[train],
        panel.label_mask[train],
    )

    def tensors(indices: np.ndarray) -> tuple[torch.Tensor, ...]:
        raw, aux = scaler.transform_features(
            panel.raw_features[indices], auxiliary[indices], panel.feature_available[indices]
        )
        targets = scaler.transform_targets(
            panel.target_cashflows[indices], panel.label_mask[indices]
        )
        return (
            torch.from_numpy(raw),
            torch.from_numpy(aux),
            torch.from_numpy(panel.feature_available[indices]),
            torch.from_numpy(targets),
            torch.from_numpy(panel.label_mask[indices]),
        )

    set_reproducible_seed(config.seed)
    device = torch.device(config.device)
    model = PairTokenITransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        TensorDataset(*tensors(train)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    validation_loader = DataLoader(
        TensorDataset(*tensors(validation)),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    training_losses: list[float] = []
    validation_losses: list[float] = []
    for _ in range(config.maximum_epochs):
        model.train()
        loss_sum = 0.0
        labels = 0
        for raw, aux, token_mask, target, label_mask in train_loader:
            count = int(label_mask.sum().item())
            if count == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(raw.to(device), aux.to(device), token_mask.to(device))
            loss = masked_smooth_l1(prediction, target.to(device), label_mask.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * count
            labels += count
        if labels == 0:
            raise ValueError("training epoch contained no labels")
        training_losses.append(loss_sum / labels)

        model.eval()
        validation_sum = 0.0
        validation_labels = 0
        with torch.no_grad():
            for raw, aux, token_mask, target, label_mask in validation_loader:
                count = int(label_mask.sum().item())
                if count == 0:
                    continue
                prediction = model(raw.to(device), aux.to(device), token_mask.to(device))
                loss = masked_smooth_l1(
                    prediction, target.to(device), label_mask.to(device)
                )
                validation_sum += float(loss.detach().cpu()) * count
                validation_labels += count
        if validation_labels == 0:
            raise ValueError("validation epoch contained no labels")
        validation_loss = validation_sum / validation_labels
        validation_losses.append(validation_loss)
        if validation_loss < best_loss - config.minimum_improvement:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a validation state")
    model.load_state_dict(best_state)
    model.eval()
    return FittedTransformer(
        model=model,
        scaler=scaler,
        config=config,
        arm=arm,
        epochs_trained=len(training_losses),
        best_validation_loss=float(best_loss),
        training_losses=tuple(training_losses),
        validation_losses=tuple(validation_losses),
    )


def rank_pair_targets(
    predictions: np.ndarray,
    available: np.ndarray,
    *,
    long_pairs: int = 8,
    short_pairs: int = 8,
    absolute_pair_target: float = 0.05,
) -> np.ndarray:
    """Map continuous forecasts to a fixed, arm-independent active basket."""

    values = np.asarray(predictions, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("predictions and availability must be time-by-pair")
    if long_pairs < 1 or short_pairs < 1:
        raise ValueError("long_pairs and short_pairs must be positive")
    if not np.isfinite(absolute_pair_target) or absolute_pair_target <= 0.0:
        raise ValueError("absolute_pair_target must be positive")
    result = np.zeros_like(values)
    required = long_pairs + short_pairs
    for row in range(len(values)):
        eligible = np.flatnonzero(mask[row] & np.isfinite(values[row]))
        if len(eligible) < required:
            continue
        row_values = values[row, eligible]
        if float(np.ptp(row_values)) <= 1e-12:
            continue
        order = eligible[np.argsort(row_values, kind="mergesort")]
        result[row, order[:short_pairs]] = -float(absolute_pair_target)
        result[row, order[-long_pairs:]] = float(absolute_pair_target)
    return result


def cross_sectional_spearman(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean cycle-wise Spearman IC without treating pairs as independent."""

    values = np.asarray(predictions, dtype=float)
    response = np.asarray(targets, dtype=float)
    observed = np.asarray(mask, dtype=bool)
    correlations: list[float] = []
    for row in range(len(values)):
        chosen = observed[row] & np.isfinite(values[row]) & np.isfinite(response[row])
        if chosen.sum() < 3:
            continue
        x = pd.Series(values[row, chosen]).rank(method="average").to_numpy()
        y = pd.Series(response[row, chosen]).rank(method="average").to_numpy()
        if np.std(x) <= 0.0 or np.std(y) <= 0.0:
            continue
        correlations.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(correlations)) if correlations else float("nan")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
