"""Leakage-safe pooled path-signature core for the multi-coin experiment.

The multi-coin contract differs from the earlier single-pair pilot in one
important respect: one coefficient vector is learned across pairs, while each
pair keeps its own residual scale.  This module implements that statistical
core without making any portfolio or execution assumptions.

The expensive object is cached once.  For every complete eight-hour episode
we store stopped signatures of the *canonical* path

``(t / 32, cumulative additive hedge residual)``.

At fit time, a training-only pair scale ``s_i`` and a declared path multiplier
``gamma`` are applied analytically.  A signature word containing ``k``
residual letters is multiplied by ``(gamma / s_i) ** k``.  Consequently the
same canonical cache can be reused across rolling outer and inner folds; no
validation or test observation participates in scaling.

Pooling uses equal pair-month weight.  Within a pair-month, episodes receive
equal weight, so a pair with a few missing episodes cannot be silently
downweighted and a pair with unusually many episodes cannot dominate the
fit.  Turnover Grams are likewise averaged over pair-month paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .signature_rkhs import (
    SignatureMeanVarianceModel,
    SignatureRKHSConfig,
    cashflow_attribution,
    feature_turnover_gram,
    signature_feature_dimension,
    signature_words,
    stopped_signature_features,
)


FeatureSet = Literal["full", "higher_only"]
_BARS_PER_EPISODE = 32
_EPISODE_DURATION = pd.Timedelta(hours=8)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


@dataclass(frozen=True)
class PooledPairEpisode:
    """One complete 8h/15m path and its linear economic unit cashflows.

    ``residual_increments`` must be additive hedge-residual increments based
    on information available at the corresponding decision time.
    ``economic_cashflows`` are the subsequently earned, per-unit cashflows
    (executable leg PnL plus funding, if applicable).  Prefer supplying these
    directly on the decision clock: eight fixed-quantity segment cashflows for
    hourly decisions.  A 32-vector is also accepted as an explicitly additive
    observation-clock input and is aggregated only when the cache is built.
    Nonlinear costs belong in the coin-netted execution ledger, not in this
    linear calibration object.
    """

    pair_id: str
    anchor: pd.Timestamp
    residual_increments: np.ndarray
    economic_cashflows: np.ndarray

    def __post_init__(self) -> None:
        pair_id = str(self.pair_id).strip()
        if not pair_id:
            raise ValueError("pair_id must not be empty")
        anchor = _utc(self.anchor)
        if anchor.minute != 0 or anchor.second != 0 or anchor.microsecond != 0:
            raise ValueError("episode anchor must lie on a whole UTC hour")
        if anchor.hour not in (0, 8, 16):
            raise ValueError("episode anchor must lie on the 00/08/16 UTC grid")
        residual = np.array(self.residual_increments, dtype=float, copy=True)
        cashflows = np.array(self.economic_cashflows, dtype=float, copy=True)
        expected = (_BARS_PER_EPISODE,)
        if residual.shape != expected:
            raise ValueError("an 8h/15m episode must contain exactly 32 increments")
        if cashflows.ndim != 1 or cashflows.size < 1 or cashflows.size > 32:
            raise ValueError("economic_cashflows must be a vector of length 1..32")
        if not np.isfinite(residual).all() or not np.isfinite(cashflows).all():
            raise ValueError("episode increments and cashflows must be finite")
        # Make the episode a true immutable input.  In particular, mutating a
        # caller-owned future array cannot alter an already built fit/cache.
        residual.setflags(write=False)
        cashflows.setflags(write=False)
        object.__setattr__(self, "pair_id", pair_id)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "residual_increments", residual)
        object.__setattr__(self, "economic_cashflows", cashflows)


@dataclass(frozen=True, order=True)
class PooledSignatureCandidate:
    """One declared pooled-signature hyperparameter candidate."""

    order: int
    gamma: float
    ridge_ratio: float
    turnover_penalty: float
    feature_set: FeatureSet = "full"

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        if not np.isfinite(self.ridge_ratio) or self.ridge_ratio < 0:
            raise ValueError("ridge_ratio must be finite and non-negative")
        if not np.isfinite(self.turnover_penalty) or self.turnover_penalty < 0:
            raise ValueError("turnover_penalty must be finite and non-negative")
        if self.feature_set not in ("full", "higher_only"):
            raise ValueError("feature_set must be 'full' or 'higher_only'")
        if self.feature_set == "higher_only" and self.order == 1:
            raise ValueError("higher_only requires signature order 2 or 3")

    @property
    def model_name(self) -> str:
        if self.order == 1:
            return "order1"
        return f"order{self.order}_{self.feature_set}"


@dataclass(frozen=True)
class PooledSignatureCache:
    """Canonical, scale-free stopped features reusable across time folds."""

    pair_ids: tuple[str, ...]
    anchors: pd.DatetimeIndex
    pair_codes: np.ndarray
    residual_increments: np.ndarray
    decision_cashflows: np.ndarray
    canonical_features: np.ndarray
    decision_indices: np.ndarray
    max_order: int
    rebalance_every_bars: int

    @property
    def episode_count(self) -> int:
        return int(len(self.anchors))

    @property
    def decision_count(self) -> int:
        return int(len(self.decision_indices))


@dataclass(frozen=True)
class PooledTrainingDesign:
    """A gamma-specific fit design shared by ridge/order candidates."""

    cache: PooledSignatureCache
    training_indices: np.ndarray
    attributions: np.ndarray
    sample_weights: np.ndarray
    turnover_gram: np.ndarray
    pair_scales: tuple[tuple[str, float], ...]
    pair_month_episode_counts: tuple[tuple[str, str, int], ...]
    pair_cashflow_statistics: tuple["PairCashflowStatistic", ...]
    sample_weight_sum: float
    sample_weight_squared_sum: float
    effective_sample_size: float
    gamma: float
    training_start: pd.Timestamp
    training_end_exclusive: pd.Timestamp
    purge_before_end: pd.Timedelta

    def scale_for_pair(self, pair_id: str) -> float:
        for name, value in self.pair_scales:
            if name == pair_id:
                return float(value)
        raise KeyError(f"pair {pair_id!r} was not present in the training fit")


@dataclass(frozen=True)
class FittedPooledSignature:
    """Shared theta plus the exact pair-specific training scales it uses."""

    candidate: PooledSignatureCandidate
    model: SignatureMeanVarianceModel
    selected_feature_indices: np.ndarray
    pair_scales: tuple[tuple[str, float], ...]
    training_start: pd.Timestamp
    training_end_exclusive: pd.Timestamp
    purge_before_end: pd.Timedelta
    training_episodes: int
    pair_month_groups: int
    effective_sample_size: float
    sample_weight_sum: float
    sample_weight_squared_sum: float

    def scale_for_pair(self, pair_id: str) -> float:
        for name, value in self.pair_scales:
            if name == pair_id:
                return float(value)
        raise KeyError(f"pair {pair_id!r} was not present in the training fit")


@dataclass(frozen=True)
class PooledInnerFold:
    """One chronological inner fit/validation split preceding an outer fold."""

    fit_start: pd.Timestamp
    fit_end_exclusive: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end_exclusive: pd.Timestamp
    purge_before_validation: pd.Timedelta


@dataclass(frozen=True)
class PairCashflowStatistic:
    """Train-only CONST diagnostic with equal calendar-month weighting."""

    pair_id: str
    training_episodes: int
    pair_months: int
    mean_episode_cashflow: float
    sign: int


@dataclass(frozen=True)
class PooledTrainScoreScale:
    """Frozen train-only multiplier for pooled raw pair scores."""

    multiplier: float
    unscaled_weighted_mean_absolute: float
    unscaled_weighted_absolute_quantile: float
    target_weighted_mean_absolute: float
    maximum_absolute: float
    quantile_probability: float
    binding_constraint: str


def build_canonical_state_path(increments: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return ``(normalized time, cumulative additive residual)`` for 8h."""

    values = np.asarray(increments, dtype=float)
    if values.shape != (_BARS_PER_EPISODE,) or not np.isfinite(values).all():
        raise ValueError("increments must be a finite 32-vector")
    time = np.arange(_BARS_PER_EPISODE + 1, dtype=float) / _BARS_PER_EPISODE
    cumulative = np.concatenate([np.zeros(1), np.cumsum(values)])
    return np.column_stack([time, cumulative])


def pooled_candidate_grid(
    *,
    orders: Sequence[int],
    gammas: Sequence[float],
    ridge_ratios: Sequence[float],
    turnover_penalties: Sequence[float],
    include_higher_only: bool = True,
) -> tuple[PooledSignatureCandidate, ...]:
    """Build the deterministic family: order 1, full 2/3 and higher-only."""

    candidates: list[PooledSignatureCandidate] = []
    for order in orders:
        feature_sets: tuple[FeatureSet, ...] = (
            ("full", "higher_only")
            if include_higher_only and int(order) >= 2
            else ("full",)
        )
        for gamma in gammas:
            for ridge in ridge_ratios:
                for turnover in turnover_penalties:
                    for feature_set in feature_sets:
                        candidates.append(
                            PooledSignatureCandidate(
                                order=int(order),
                                gamma=float(gamma),
                                ridge_ratio=float(ridge),
                                turnover_penalty=float(turnover),
                                feature_set=feature_set,
                            )
                        )
    result = tuple(candidates)
    if len(result) != len(set(result)):
        raise ValueError("candidate grid contains duplicates")
    return result


def build_pooled_signature_cache(
    episodes: Iterable[PooledPairEpisode],
    *,
    max_order: int = 3,
    rebalance_every_bars: int = 4,
) -> PooledSignatureCache:
    """Precompute canonical signatures once for all rolling/nested fits.

    ``rebalance_every_bars=4`` is the contract's primary hourly decision
    clock; ``1`` supports the declared 15-minute secondary diagnostic.
    """

    if max_order not in (1, 2, 3):
        raise ValueError("max_order must be 1, 2 or 3")
    if (
        isinstance(rebalance_every_bars, (bool, np.bool_))
        or not isinstance(rebalance_every_bars, (int, np.integer))
        or rebalance_every_bars < 1
        or _BARS_PER_EPISODE % int(rebalance_every_bars)
    ):
        raise ValueError("rebalance_every_bars must divide the 32-bar episode")
    ordered = sorted(tuple(episodes), key=lambda item: (item.anchor, item.pair_id))
    if not ordered:
        raise ValueError("at least one episode is required")
    keys = [(item.pair_id, item.anchor.value) for item in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("pair/anchor episode keys must be unique")

    pair_ids = tuple(sorted({item.pair_id for item in ordered}))
    pair_lookup = {pair_id: code for code, pair_id in enumerate(pair_ids)}
    decision_indices = np.arange(
        0, _BARS_PER_EPISODE, int(rebalance_every_bars), dtype=np.int64
    )
    dimension = signature_feature_dimension(max_order)
    feature_rows = np.empty(
        (len(ordered), len(decision_indices), dimension), dtype=float
    )
    residuals = np.empty((len(ordered), _BARS_PER_EPISODE), dtype=float)
    decision_cashflows = np.empty((len(ordered), len(decision_indices)), dtype=float)
    pair_codes = np.empty(len(ordered), dtype=np.int64)
    anchors: list[pd.Timestamp] = []
    for row, episode in enumerate(ordered):
        path = build_canonical_state_path(episode.residual_increments)
        stopped = stopped_signature_features(path, max_order)
        feature_rows[row] = stopped[decision_indices]
        residuals[row] = episode.residual_increments
        if episode.economic_cashflows.size == len(decision_indices):
            # Primary route: exact fixed-quantity segment economics were
            # calculated by the runner on the actual decision clock.
            decision_cashflows[row] = episode.economic_cashflows
        elif episode.economic_cashflows.size == _BARS_PER_EPISODE:
            # Secondary route for a genuinely additive observation-clock
            # attribution (or the native 15m decision clock).
            decision_cashflows[row] = np.add.reduceat(
                episode.economic_cashflows, decision_indices
            )
        else:
            raise ValueError(
                "economic_cashflows must have one value per decision or per 15m bar"
            )
        pair_codes[row] = pair_lookup[episode.pair_id]
        anchors.append(episode.anchor)

    # Cache arrays are deliberately read-only: feature reuse must never turn
    # into hidden state carried from one nested fit to the next.
    for array in (
        pair_codes,
        residuals,
        decision_cashflows,
        feature_rows,
        decision_indices,
    ):
        array.setflags(write=False)
    return PooledSignatureCache(
        pair_ids=pair_ids,
        anchors=pd.DatetimeIndex(anchors),
        pair_codes=pair_codes,
        residual_increments=residuals,
        decision_cashflows=decision_cashflows,
        canonical_features=feature_rows,
        decision_indices=decision_indices,
        max_order=int(max_order),
        rebalance_every_bars=int(rebalance_every_bars),
    )


def select_cache_indices(
    cache: PooledSignatureCache,
    *,
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    purge_before_end: pd.Timedelta = pd.Timedelta(0),
    pair_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Select complete episodes in a causal half-open time window."""

    first = _utc(start)
    end = _utc(end_exclusive)
    purge = pd.Timedelta(purge_before_end)
    if purge < pd.Timedelta(0):
        raise ValueError("purge_before_end must be non-negative")
    stop = end - purge
    if first >= stop:
        raise ValueError("selection window is empty after purge")
    mask = (cache.anchors >= first) & (
        cache.anchors + _EPISODE_DURATION <= stop
    )
    if pair_ids is not None:
        requested = tuple(str(item) for item in pair_ids)
        unknown = sorted(set(requested).difference(cache.pair_ids))
        if unknown:
            raise KeyError(f"unknown pairs: {unknown}")
        allowed_codes = np.asarray(
            [cache.pair_ids.index(pair_id) for pair_id in requested], dtype=int
        )
        mask &= np.isin(cache.pair_codes, allowed_codes)
    indices = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)
    if indices.size == 0:
        raise ValueError("cache selection contains no complete episodes")
    indices.setflags(write=False)
    return indices


def equal_pair_month_sample_weights(
    pair_ids: Sequence[str], anchors: Sequence[pd.Timestamp] | pd.DatetimeIndex
) -> np.ndarray:
    """Give every observed pair-month total weight one."""

    names = np.asarray([str(value) for value in pair_ids], dtype=object)
    dates = pd.DatetimeIndex(anchors)
    if names.ndim != 1 or names.size != len(dates) or names.size < 1:
        raise ValueError("pair_ids and anchors must be aligned non-empty vectors")
    if dates.tz is None:
        dates = dates.tz_localize("UTC")
    else:
        dates = dates.tz_convert("UTC")
    # Deterministic Python tuples are clear and fast enough for roughly 10^5
    # episodes.
    tuple_keys = [(str(names[i]), int(dates[i].year), int(dates[i].month)) for i in range(len(dates))]
    counts: dict[tuple[str, int, int], int] = {}
    for key in tuple_keys:
        counts[key] = counts.get(key, 0) + 1
    group_count = len(counts)
    weights = np.asarray(
        [1.0 / (group_count * counts[key]) for key in tuple_keys], dtype=float
    )
    weights /= float(weights.sum())
    return weights


def pair_month_episode_counts(
    pair_ids: Sequence[str], anchors: Sequence[pd.Timestamp] | pd.DatetimeIndex
) -> tuple[tuple[str, str, int], ...]:
    """Return deterministic auditing counts behind equal pair-month weights."""

    names = tuple(str(value) for value in pair_ids)
    dates = pd.DatetimeIndex(anchors)
    if len(names) != len(dates) or not names:
        raise ValueError("pair_ids and anchors must be aligned non-empty vectors")
    if dates.tz is None:
        dates = dates.tz_localize("UTC")
    else:
        dates = dates.tz_convert("UTC")
    counts: dict[tuple[str, str], int] = {}
    for name, date in zip(names, dates, strict=True):
        key = (name, f"{date.year:04d}-{date.month:02d}")
        counts[key] = counts.get(key, 0) + 1
    return tuple((pair, month, counts[(pair, month)]) for pair, month in sorted(counts))


def pair_training_cashflow_statistics(
    cache: PooledSignatureCache,
    training_indices: Sequence[int] | np.ndarray,
) -> tuple[PairCashflowStatistic, ...]:
    """Compute per-pair CONST mean/sign from training economics only.

    An episode is first reduced to its total unit economic cashflow.  Episodes
    are averaged inside each calendar month and those monthly means are then
    averaged equally for the pair.  Thus CONST follows the same missing-data
    convention as the pooled signature fit.
    """

    indices = np.asarray(training_indices, dtype=int)
    if indices.ndim != 1 or indices.size < 1:
        raise ValueError("training_indices must be a non-empty vector")
    results: list[PairCashflowStatistic] = []
    for code, pair_id in enumerate(cache.pair_ids):
        chosen = indices[cache.pair_codes[indices] == code]
        if chosen.size == 0:
            continue
        months: dict[tuple[int, int], list[float]] = {}
        for cache_index in chosen:
            anchor = cache.anchors[cache_index]
            months.setdefault((int(anchor.year), int(anchor.month)), []).append(
                float(cache.decision_cashflows[cache_index].sum())
            )
        monthly_means = np.asarray(
            [np.mean(months[key]) for key in sorted(months)], dtype=float
        )
        mean = float(monthly_means.mean())
        sign = int(np.sign(mean))
        results.append(
            PairCashflowStatistic(
                pair_id=pair_id,
                training_episodes=int(chosen.size),
                pair_months=int(len(months)),
                mean_episode_cashflow=mean,
                sign=sign,
            )
        )
    if not results:
        raise ValueError("no per-pair training cashflow statistic is available")
    return tuple(results)


def robust_pair_scales(
    cache: PooledSignatureCache,
    training_indices: Sequence[int] | np.ndarray,
    *,
    floor: float = 1e-8,
) -> tuple[tuple[str, float], ...]:
    """Fit Gaussian-consistent terminal-state MAD separately for each pair."""

    indices = np.asarray(training_indices, dtype=int)
    if indices.ndim != 1 or indices.size < 2:
        raise ValueError("at least two training episodes are required")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be finite and positive")
    result: list[tuple[str, float]] = []
    for code, pair_id in enumerate(cache.pair_ids):
        selected = indices[cache.pair_codes[indices] == code]
        if selected.size == 0:
            continue
        if selected.size < 2:
            raise ValueError(f"pair {pair_id!r} has fewer than two training episodes")
        terminal = cache.residual_increments[selected].sum(axis=1)
        median = float(np.median(terminal))
        scale = float(1.4826 * np.median(np.abs(terminal - median)))
        if not np.isfinite(scale) or scale <= floor:
            scale = float(np.std(terminal, ddof=1))
        if not np.isfinite(scale) or scale <= floor:
            scale = float(floor)
        result.append((pair_id, scale))
    if not result:
        raise ValueError("no pair scale could be estimated")
    return tuple(result)


def _residual_letter_counts(order: int) -> np.ndarray:
    return np.asarray([word.count(1) for word in signature_words(order)], dtype=int)


def _pair_scale_lookup(scales: tuple[tuple[str, float], ...]) -> dict[str, float]:
    return {pair_id: float(scale) for pair_id, scale in scales}


def _scaled_feature_rows(
    cache: PooledSignatureCache,
    indices: np.ndarray,
    *,
    pair_scales: tuple[tuple[str, float], ...],
    gamma: float,
) -> np.ndarray:
    dimension = signature_feature_dimension(cache.max_order)
    counts = _residual_letter_counts(cache.max_order)
    lookup = _pair_scale_lookup(pair_scales)
    factors = np.empty((len(indices), dimension), dtype=float)
    for row, cache_index in enumerate(indices):
        pair_id = cache.pair_ids[int(cache.pair_codes[cache_index])]
        if pair_id not in lookup:
            raise KeyError(f"pair {pair_id!r} has no frozen training scale")
        factors[row] = (float(gamma) / lookup[pair_id]) ** counts
    return cache.canonical_features[indices] * factors[:, None, :]


def _equal_pair_month_turnover_gram(
    features: np.ndarray,
    cache: PooledSignatureCache,
    indices: np.ndarray,
) -> np.ndarray:
    grouped: dict[tuple[str, int, int], list[int]] = {}
    for local, cache_index in enumerate(indices):
        anchor = cache.anchors[cache_index]
        pair_id = cache.pair_ids[int(cache.pair_codes[cache_index])]
        key = (pair_id, int(anchor.year), int(anchor.month))
        grouped.setdefault(key, []).append(local)
    grams: list[np.ndarray] = []
    for key in sorted(grouped):
        rows = grouped[key]
        # Cache order is chronological inside each pair.  Concatenating the
        # decision paths captures genuine jumps at 8h signature resets.
        path = np.vstack([features[row] for row in rows])
        grams.append(
            feature_turnover_gram(
                (path,), include_entry_exit=True, drop_terminal_feature=False
            )
        )
    return np.mean(np.stack(grams, axis=0), axis=0)


def prepare_pooled_training_design(
    cache: PooledSignatureCache,
    *,
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    gamma: float,
    purge_before_end: pd.Timedelta = pd.Timedelta(0),
    pair_ids: Sequence[str] | None = None,
    scale_floor: float = 1e-8,
) -> PooledTrainingDesign:
    """Prepare one leakage-safe design reusable across candidate fits."""

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    indices = select_cache_indices(
        cache,
        start=start,
        end_exclusive=end_exclusive,
        purge_before_end=purge_before_end,
        pair_ids=pair_ids,
    )
    scales = robust_pair_scales(cache, indices, floor=scale_floor)
    features = _scaled_feature_rows(
        cache, indices, pair_scales=scales, gamma=float(gamma)
    )
    attributions = cashflow_attribution(features, cache.decision_cashflows[indices])
    names = [cache.pair_ids[int(cache.pair_codes[index])] for index in indices]
    weights = equal_pair_month_sample_weights(names, cache.anchors[indices])
    counts = pair_month_episode_counts(names, cache.anchors[indices])
    cashflow_statistics = pair_training_cashflow_statistics(cache, indices)
    turnover = _equal_pair_month_turnover_gram(features, cache, indices)
    for array in (indices, attributions, weights, turnover):
        array.setflags(write=False)
    return PooledTrainingDesign(
        cache=cache,
        training_indices=indices,
        attributions=attributions,
        sample_weights=weights,
        turnover_gram=turnover,
        pair_scales=scales,
        pair_month_episode_counts=counts,
        pair_cashflow_statistics=cashflow_statistics,
        sample_weight_sum=float(weights.sum()),
        sample_weight_squared_sum=float(weights @ weights),
        effective_sample_size=float(1.0 / (weights @ weights)),
        gamma=float(gamma),
        training_start=_utc(start),
        training_end_exclusive=_utc(end_exclusive),
        purge_before_end=pd.Timedelta(purge_before_end),
    )


def selected_feature_indices(candidate: PooledSignatureCandidate) -> np.ndarray:
    """Return active coordinates; higher-only contains no levels 0 or 1."""

    words = signature_words(candidate.order)
    if candidate.feature_set == "full":
        result = np.arange(len(words), dtype=np.int64)
    else:
        result = np.asarray(
            [index for index, word in enumerate(words) if len(word) >= 2],
            dtype=np.int64,
        )
    result.setflags(write=False)
    return result


def fit_pooled_signature_from_design(
    design: PooledTrainingDesign,
    candidate: PooledSignatureCandidate,
    *,
    numerical_floor: float = 1e-12,
) -> FittedPooledSignature:
    """Fit shared theta; excluded feature levels are exactly zero."""

    if not np.isclose(candidate.gamma, design.gamma, atol=0.0, rtol=0.0):
        raise ValueError("candidate gamma does not match the prepared design")
    if candidate.order > design.cache.max_order:
        raise ValueError("candidate order exceeds cache max_order")
    dimension = signature_feature_dimension(candidate.order)
    active = selected_feature_indices(candidate)
    values = design.attributions[:, :dimension][:, active]
    gram = design.turnover_gram[:dimension, :dimension][np.ix_(active, active)]

    # Solve in the compact active space, then embed exactly into the declared
    # tensor order.  The compact order is not generally a signature order, so
    # perform the weighted algebra here and build the public full model below.
    weights = np.asarray(design.sample_weights, dtype=float)
    weights = weights / weights.sum()
    mean_active = weights @ values
    centered = values - mean_active
    correction = 1.0 - float(weights @ weights)
    if correction <= numerical_floor:
        raise ValueError("candidate has fewer than two effective samples")
    covariance_active = (centered * weights[:, None]).T @ centered / correction
    covariance_active = 0.5 * (covariance_active + covariance_active.T)
    gram = 0.5 * (gram + gram.T)
    active_dimension = len(active)
    ridge_scale = max(
        float(np.trace(covariance_active) / active_dimension), numerical_floor
    )
    regularized_active = covariance_active + (
        candidate.ridge_ratio * ridge_scale * np.eye(active_dimension)
    )
    turnover_scale = float(np.trace(gram) / active_dimension)
    if candidate.turnover_penalty > 0 and turnover_scale > numerical_floor:
        regularized_active += (
            candidate.turnover_penalty * ridge_scale * gram / turnover_scale
        )
    regularized_active = 0.5 * (regularized_active + regularized_active.T)
    eigenvalues, eigenvectors = np.linalg.eigh(regularized_active)
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = max(float(eigenvalues[-1]), numerical_floor)
    solve_floor = max(numerical_floor, numerical_floor * maximum_eigenvalue)
    stable = np.maximum(eigenvalues, solve_floor)
    coefficients_active = 0.5 * eigenvectors @ (
        (eigenvectors.T @ mean_active) / stable
    )
    if not np.isfinite(coefficients_active).all():
        raise FloatingPointError("pooled mean-variance solve is non-finite")

    coefficients = np.zeros(dimension, dtype=float)
    mean = np.zeros(dimension, dtype=float)
    covariance = np.zeros((dimension, dimension), dtype=float)
    regularized = np.zeros((dimension, dimension), dtype=float)
    coefficients[active] = coefficients_active
    mean[active] = mean_active
    covariance[np.ix_(active, active)] = covariance_active
    regularized[np.ix_(active, active)] = regularized_active
    variance = max(
        float(coefficients_active @ covariance_active @ coefficients_active), 0.0
    )
    config = SignatureRKHSConfig(
        order=candidate.order,
        ridge_ratio=candidate.ridge_ratio,
        turnover_penalty=candidate.turnover_penalty,
        variance_target=None,
        numerical_floor=numerical_floor,
    )
    model = SignatureMeanVarianceModel(
        config=config,
        coefficients=coefficients,
        mean_attribution=mean,
        covariance=covariance,
        regularized_matrix=regularized,
        ridge_scale=float(ridge_scale),
        turnover_scale=float(turnover_scale),
        variance_before_scaling=float(variance),
        normalization=1.0,
        n_samples=int(len(values)),
        condition_number=float(stable[-1] / stable[0]),
        effective_rank=int(np.count_nonzero(eigenvalues > solve_floor)),
        min_eigenvalue=minimum_eigenvalue,
    )
    group_keys = {
        (
            design.cache.pair_ids[int(design.cache.pair_codes[index])],
            int(design.cache.anchors[index].year),
            int(design.cache.anchors[index].month),
        )
        for index in design.training_indices
    }
    effective_sample_size = 1.0 / float(design.sample_weights @ design.sample_weights)
    return FittedPooledSignature(
        candidate=candidate,
        model=model,
        selected_feature_indices=active,
        pair_scales=design.pair_scales,
        training_start=design.training_start,
        training_end_exclusive=design.training_end_exclusive,
        purge_before_end=design.purge_before_end,
        training_episodes=int(len(design.training_indices)),
        pair_month_groups=int(len(group_keys)),
        effective_sample_size=float(effective_sample_size),
        sample_weight_sum=float(design.sample_weight_sum),
        sample_weight_squared_sum=float(design.sample_weight_squared_sum),
    )


def fit_pooled_signature(
    cache: PooledSignatureCache,
    candidate: PooledSignatureCandidate,
    *,
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    purge_before_end: pd.Timedelta = pd.Timedelta(0),
    pair_ids: Sequence[str] | None = None,
    scale_floor: float = 1e-8,
) -> FittedPooledSignature:
    """Convenience wrapper for one arbitrary chronological fit window."""

    design = prepare_pooled_training_design(
        cache,
        start=start,
        end_exclusive=end_exclusive,
        gamma=candidate.gamma,
        purge_before_end=purge_before_end,
        pair_ids=pair_ids,
        scale_floor=scale_floor,
    )
    return fit_pooled_signature_from_design(design, candidate)


def fit_pooled_candidate_family(
    cache: PooledSignatureCache,
    candidates: Sequence[PooledSignatureCandidate],
    *,
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    purge_before_end: pd.Timedelta = pd.Timedelta(0),
    pair_ids: Sequence[str] | None = None,
    scale_floor: float = 1e-8,
) -> dict[PooledSignatureCandidate, FittedPooledSignature]:
    """Fit a family while preparing each gamma-specific design only once."""

    family = tuple(candidates)
    if not family:
        raise ValueError("candidate family must not be empty")
    designs: dict[float, PooledTrainingDesign] = {}
    result: dict[PooledSignatureCandidate, FittedPooledSignature] = {}
    for candidate in family:
        if candidate.gamma not in designs:
            designs[candidate.gamma] = prepare_pooled_training_design(
                cache,
                start=start,
                end_exclusive=end_exclusive,
                gamma=candidate.gamma,
                purge_before_end=purge_before_end,
                pair_ids=pair_ids,
                scale_floor=scale_floor,
            )
        result[candidate] = fit_pooled_signature_from_design(
            designs[candidate.gamma], candidate
        )
    return result


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / float(sorted_weights.sum())
    location = min(int(np.searchsorted(cumulative, probability, side="left")), len(values) - 1)
    return float(sorted_values[location])


def calibrate_train_only_pooled_score_scale(
    design: PooledTrainingDesign,
    fitted: FittedPooledSignature,
    *,
    target_weighted_mean_absolute: float = 0.10,
    maximum_absolute: float = 0.25,
    quantile_probability: float = 0.99,
    numerical_floor: float = 1e-12,
) -> PooledTrainScoreScale:
    """Calibrate raw pair-score magnitude with fit observations only.

    Episode weights retain equal pair-month pooling and are divided equally
    over decision rows.  The returned immutable object can be passed to
    :func:`score_pooled_signature`; that scoring function has no way to refit
    it on validation or test data.
    """

    if not 0 < target_weighted_mean_absolute < maximum_absolute <= 1:
        raise ValueError("score targets must satisfy 0 < mean < cap <= 1")
    if not 0.5 <= quantile_probability < 1:
        raise ValueError("quantile_probability must be in [0.5, 1)")
    if fitted.candidate.gamma != design.gamma or fitted.pair_scales != design.pair_scales:
        raise ValueError("fitted model and training design do not match")
    features = _scaled_feature_rows(
        design.cache,
        design.training_indices,
        pair_scales=design.pair_scales,
        gamma=design.gamma,
    )
    dimension = signature_feature_dimension(fitted.candidate.order)
    raw = features[:, :, :dimension] @ fitted.model.coefficients
    row_weights = np.repeat(
        design.sample_weights / raw.shape[1], raw.shape[1]
    )
    absolute = np.abs(raw).reshape(-1)
    mean_absolute = float(row_weights @ absolute)
    high_quantile = _weighted_quantile(
        absolute, row_weights, float(quantile_probability)
    )
    if mean_absolute <= numerical_floor or high_quantile <= numerical_floor:
        raise ValueError("raw pooled training scores are numerically zero")
    constraints = {
        "target_weighted_mean_absolute": (
            target_weighted_mean_absolute / mean_absolute
        ),
        "maximum_absolute_quantile": maximum_absolute / high_quantile,
    }
    binding = min(constraints, key=constraints.get)
    return PooledTrainScoreScale(
        multiplier=float(constraints[binding]),
        unscaled_weighted_mean_absolute=mean_absolute,
        unscaled_weighted_absolute_quantile=high_quantile,
        target_weighted_mean_absolute=float(target_weighted_mean_absolute),
        maximum_absolute=float(maximum_absolute),
        quantile_probability=float(quantile_probability),
        binding_constraint=binding,
    )


def score_pooled_signature(
    cache: PooledSignatureCache,
    fitted: FittedPooledSignature,
    *,
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    pair_ids: Sequence[str] | None = None,
    expand_to_15m_bars: bool = True,
    position_scale: PooledTrainScoreScale | None = None,
    no_trade_absolute: float = 0.0,
) -> dict[str, dict[pd.Timestamp, np.ndarray]]:
    """Score each pair using only the scales frozen in ``fitted``.

    No scale-estimation option is accepted here by design.  Validation and
    test scores therefore cannot accidentally recalibrate on their own path.
    With the default expansion, each hourly decision score is held over its
    four constituent 15-minute bars.
    """

    indices = select_cache_indices(
        cache,
        start=start,
        end_exclusive=end_exclusive,
        pair_ids=pair_ids,
    )
    fitted_pairs = {pair_id for pair_id, _ in fitted.pair_scales}
    requested_pairs = {
        cache.pair_ids[int(cache.pair_codes[index])] for index in indices
    }
    missing = sorted(requested_pairs.difference(fitted_pairs))
    if missing:
        raise KeyError(f"validation/test pairs lack frozen training scales: {missing}")
    features = _scaled_feature_rows(
        cache,
        indices,
        pair_scales=fitted.pair_scales,
        gamma=fitted.candidate.gamma,
    )
    dimension = signature_feature_dimension(fitted.candidate.order)
    decisions = features[:, :, :dimension] @ fitted.model.coefficients
    if position_scale is not None:
        if not 0 <= no_trade_absolute < position_scale.maximum_absolute:
            raise ValueError("no_trade_absolute must be below the frozen score cap")
        decisions = np.clip(
            decisions * position_scale.multiplier,
            -position_scale.maximum_absolute,
            position_scale.maximum_absolute,
        )
        decisions[np.abs(decisions) < no_trade_absolute] = 0.0
    elif no_trade_absolute != 0.0:
        raise ValueError("no_trade_absolute requires a frozen position_scale")
    result: dict[str, dict[pd.Timestamp, np.ndarray]] = {}
    for local, cache_index in enumerate(indices):
        pair_id = cache.pair_ids[int(cache.pair_codes[cache_index])]
        values = np.asarray(decisions[local], dtype=float)
        if expand_to_15m_bars:
            values = np.repeat(values, cache.rebalance_every_bars)[:_BARS_PER_EPISODE]
        values = np.array(values, copy=True)
        values.setflags(write=False)
        result.setdefault(pair_id, {})[cache.anchors[cache_index]] = values
    return result


def build_rolling_inner_folds(
    outer_start: str | pd.Timestamp,
    *,
    validation_blocks: int = 3,
    validation_days: int = 30,
    training_days: int = 365,
    purge_hours: int = 8,
    available_start: str | pd.Timestamp | None = None,
) -> tuple[PooledInnerFold, ...]:
    """Construct fully chronological inner folds before any outer test month.

    The function is deliberately independent of an outer-fold count.  For an
    outer start ``T``, validation blocks tile the interval immediately before
    ``T`` and every block receives its own trailing training window.
    """

    if validation_blocks < 1 or validation_days < 1 or training_days < 1:
        raise ValueError("fold lengths and count must be positive")
    if purge_hours < 0:
        raise ValueError("purge_hours must be non-negative")
    outer = _utc(outer_start)
    validation_span = pd.Timedelta(days=int(validation_days))
    training_span = pd.Timedelta(days=int(training_days))
    purge = pd.Timedelta(hours=int(purge_hours))
    folds: list[PooledInnerFold] = []
    for block in range(int(validation_blocks), 0, -1):
        validation_start = outer - block * validation_span
        validation_end = validation_start + validation_span
        fit_end = validation_start - purge
        # Preserve a full usable training span *before* the separate purge.
        fit_start = fit_end - training_span
        folds.append(
            PooledInnerFold(
                fit_start=fit_start,
                fit_end_exclusive=fit_end,
                validation_start=validation_start,
                validation_end_exclusive=validation_end,
                purge_before_validation=purge,
            )
        )
    if available_start is not None and folds[0].fit_start < _utc(available_start):
        raise ValueError("outer fold lacks a complete nested training history")
    return tuple(folds)


def prepare_pooled_inner_training_design(
    cache: PooledSignatureCache,
    fold: PooledInnerFold,
    *,
    gamma: float,
    pair_ids: Sequence[str] | None = None,
    scale_floor: float = 1e-8,
) -> PooledTrainingDesign:
    """Prepare a fold without risking a second application of its purge.

    ``PooledInnerFold.fit_end_exclusive`` is already eight hours (or the
    declared alternative) before validation.  This helper therefore passes a
    zero additional purge to the general design builder.
    """

    return prepare_pooled_training_design(
        cache,
        start=fold.fit_start,
        end_exclusive=fold.fit_end_exclusive,
        gamma=gamma,
        purge_before_end=pd.Timedelta(0),
        pair_ids=pair_ids,
        scale_floor=scale_floor,
    )
