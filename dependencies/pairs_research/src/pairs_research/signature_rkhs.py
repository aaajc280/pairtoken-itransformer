"""Finite signature-RKHS mean-variance model for scalar residual paths.

This module implements the finite-dimensional version of the signature
trading construction used by Futter and Wiese.  A linear functional on the
truncated signature is a function in the RKHS induced by the corresponding
finite signature kernel.  The implementation is deliberately explicit:

* a Daily residual return is treated as an increment of one tradable path;
* the state path is ``(time, cumulative standardized residual)``;
* the empty word is retained, so a constant position is part of the model;
* calibration can use either the project's exact next-increment discrete PnL
  attribution or the terminal Hoff lead-lag coordinates ``w f`` from the
  authors' reference fast method; and
* ridge and turnover regularisation are expressed in the same variance units.

There is no scale estimation inside :func:`build_residual_path`.  ``scale``
must be frozen from information available before the path starts (or omitted
when the increments are already standardized).  This is important: using the
terminal sample volatility to rescale stopped paths would leak future data.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial
from typing import Iterable, Sequence

import numpy as np

try:  # Optional C++ acceleration; transparent Chen fallbacks remain below.
    import iisignature as _iisignature
except ImportError:  # pragma: no cover - exercised on minimal installations
    _iisignature = None


_ORIGINAL_DIMENSION = 2  # time, cumulative residual
_HOFF_DIMENSION = 2 * _ORIGINAL_DIMENSION
_HOFF_TRADABLE_LEAD_COORDINATE = 3


@dataclass(frozen=True)
class SignatureRKHSConfig:
    """Hyperparameters of the finite signature mean-variance problem.

    ``ridge_ratio`` is relative to the average diagonal of the empirical PnL
    covariance.  ``turnover_penalty`` is dimensionless as well: a supplied
    turnover Gram matrix is normalized to that same average variance before
    entering the quadratic objective.  ``variance_target=None`` leaves the
    unconstrained mean-variance solution untouched; a positive value rescales
    it to the requested in-sample PnL variance, as in the reference code.
    """

    order: int = 2
    ridge_ratio: float = 1e-3
    turnover_penalty: float = 0.0
    variance_target: float | None = 1.0
    numerical_floor: float = 1e-12

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        if not np.isfinite(self.ridge_ratio) or self.ridge_ratio < 0:
            raise ValueError("ridge_ratio must be finite and non-negative")
        if not np.isfinite(self.turnover_penalty) or self.turnover_penalty < 0:
            raise ValueError("turnover_penalty must be finite and non-negative")
        if self.variance_target is not None and (
            not np.isfinite(self.variance_target) or self.variance_target <= 0
        ):
            raise ValueError("variance_target must be positive or None")
        if not np.isfinite(self.numerical_floor) or self.numerical_floor <= 0:
            raise ValueError("numerical_floor must be finite and positive")


@dataclass(frozen=True)
class SignatureMeanVarianceModel:
    """Fitted coefficients and auditable calibration diagnostics."""

    config: SignatureRKHSConfig
    coefficients: np.ndarray
    mean_attribution: np.ndarray
    covariance: np.ndarray
    regularized_matrix: np.ndarray
    ridge_scale: float
    turnover_scale: float
    variance_before_scaling: float
    normalization: float
    n_samples: int
    condition_number: float = float("nan")
    effective_rank: int = 0
    min_eigenvalue: float = float("nan")

    @property
    def feature_dimension(self) -> int:
        return int(self.coefficients.size)

    def score_path(self, path: np.ndarray) -> np.ndarray:
        """Return the position at every stopped time of ``path``."""

        return score_signature_path(path, self)


def signature_feature_dimension(order: int) -> int:
    """Dimension of ``T^order(R^2)`` including the empty word."""

    _validate_order(order)
    return int(sum(_ORIGINAL_DIMENSION**level for level in range(order + 1)))


def signature_words(
    order: int, *, dimension: int = _ORIGINAL_DIMENSION
) -> tuple[tuple[int, ...], ...]:
    """Return tensor words in flattened signature order, including ``()``."""

    _validate_order(order)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    words: list[tuple[int, ...]] = [()]
    for level in range(1, order + 1):
        words.extend(product(range(dimension), repeat=level))
    return tuple(words)


def tensor_word_index(
    word: Sequence[int], dimension: int, *, include_scalar: bool = True
) -> int:
    """Map a tensor word to its level-major, lexicographic flat coordinate.

    This is the ordering used by ``iisignature`` and by the official
    ``word_to_loc`` helper.  With a scalar term the empty word has index zero.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    letters = tuple(int(letter) for letter in word)
    if not letters and not include_scalar:
        raise ValueError("the empty word has no coordinate without a scalar term")
    if any(letter < 0 or letter >= dimension for letter in letters):
        raise ValueError("word contains a letter outside the tensor alphabet")
    level = len(letters)
    offset = sum(dimension**k for k in range(level)) if include_scalar else sum(
        dimension**k for k in range(1, level)
    )
    encoded = sum(
        letter * dimension ** (level - position - 1)
        for position, letter in enumerate(letters)
    )
    return int(offset + encoded)


def build_residual_path(
    increments: np.ndarray | Sequence[float],
    *,
    scale: float | None = None,
    horizon: int = 60,
) -> np.ndarray:
    """Build a causal ``(time, cumulative residual)`` path with a basepoint.

    Parameters
    ----------
    increments:
        Already-observed Daily factor-residual returns.
    scale:
        A strictly positive volatility estimate frozen before this path.  If
        omitted, increments are interpreted as already standardized.
    horizon:
        Fixed block horizon.  Time is ``j / horizon`` and residual increments
        are divided by ``sqrt(horizon)``.  Keeping this value fixed for every
        prefix makes stopped paths nested and their magnitudes comparable.

    Notes
    -----
    The increments are not demeaned.  Demeaning by their terminal sample mean
    would both erase a possible drift and introduce look-ahead bias.
    """

    residuals = np.asarray(increments, dtype=float)
    if residuals.ndim != 1 or residuals.size < 1:
        raise ValueError("increments must be a non-empty one-dimensional array")
    if not np.isfinite(residuals).all():
        raise ValueError("increments must be finite")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if residuals.size > horizon:
        raise ValueError("the number of increments cannot exceed horizon")
    frozen_scale = 1.0 if scale is None else float(scale)
    if not np.isfinite(frozen_scale) or frozen_scale <= 0:
        raise ValueError("scale must be finite and strictly positive")

    time = np.arange(residuals.size + 1, dtype=float) / float(horizon)
    standardized = residuals / (frozen_scale * np.sqrt(float(horizon)))
    cumulative = np.concatenate([np.zeros(1), np.cumsum(standardized)])
    return np.column_stack([time, cumulative])


def stopped_signature_features(path: np.ndarray, order: int) -> np.ndarray:
    """Compute ordinary signature features at every stopped path prefix.

    Row zero is the tensor identity ``(1, 0, ..., 0)`` at the basepoint.  Row
    ``t`` therefore only depends on path points ``0..t``.  The returned column
    order is empty word, level one, level two, and (if requested) level three.
    """

    x = _validate_original_path(path)
    _validate_order(order)
    dimension = signature_feature_dimension(order)
    identity = np.zeros((1, dimension), dtype=float)
    identity[0, 0] = 1.0
    if x.shape[0] == 1:
        return identity

    if _iisignature is not None:
        prefixes = np.asarray(_iisignature.sig(x, order, 2), dtype=float)
        prefixes = np.column_stack([np.ones(prefixes.shape[0]), prefixes])
    else:  # pragma: no cover - normally the optional accelerator is installed
        prefixes = np.vstack(
            [_signature_with_scalar(x[: stop + 1], order) for stop in range(1, len(x))]
        )
    return np.vstack([identity, prefixes])


def hoff_lead_lag(path: np.ndarray) -> np.ndarray:
    """Apply the exact Hoff lead-lag lift used by the reference fast method.

    Each point is repeated four times, then the original copy shifted by five
    rows is concatenated channel-wise.  For the original coordinates
    ``(time, residual)`` the resulting alphabet is
    ``(time_lag, residual_lag, time_lead, residual_lead)``.  The tradable lead
    coordinate is consequently letter ``3``.
    """

    x = np.asarray(path, dtype=float)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None, ...]
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] != _ORIGINAL_DIMENSION:
        raise ValueError("path must have shape (time, 2) or (batch, time, 2)")
    if not np.isfinite(x).all():
        raise ValueError("path must be finite")
    repeated = np.repeat(x, repeats=4, axis=1)
    lifted = np.concatenate([repeated[:, :-5], repeated[:, 5:]], axis=2)
    return lifted[0] if squeeze else lifted


def pnl_attribution_selector_indices(order: int) -> np.ndarray:
    """Indices of ``w + (3,)`` in the 4-D Hoff signature through order M+1."""

    words = signature_words(order)
    return np.asarray(
        [
            tensor_word_index(
                word + (_HOFF_TRADABLE_LEAD_COORDINATE,),
                _HOFF_DIMENSION,
                include_scalar=True,
            )
            for word in words
        ],
        dtype=int,
    )


def terminal_pnl_attribution(path: np.ndarray, order: int) -> np.ndarray:
    r"""Return the reference repository's fast-clock PnL coordinates.

    For every position word ``w`` this selects

    ``Sig(Hoff(path))^[w, residual_lead]``.

    Their empirical mean and covariance are precisely the quantities used in
    the official repository's fast Futter/Wiese calibration.  Its repeat-four,
    shift-five Hoff clock contains an additional observation lag: on an
    ordinary stopped-feature path this equals

    ``features[:-2].T @ diff(residual_path)[1:]``.

    It must therefore not be mixed silently with a standard Daily strategy in
    which the position observed at close ``t`` earns the immediately following
    increment.  Use :func:`discrete_pnl_attribution` for that execution clock.
    A two-dimensional input returns one vector; a batched input returns one
    vector per path.
    """

    _validate_order(order)
    x = np.asarray(path, dtype=float)
    squeeze = x.ndim == 2
    lifted = hoff_lead_lag(x)
    if squeeze:
        lifted_batch = lifted[None, ...]
    else:
        lifted_batch = lifted
    terminal = _batch_signature_with_scalar(lifted_batch, order + 1)
    selected = terminal[:, pnl_attribution_selector_indices(order)]
    return selected[0] if squeeze else selected


def discrete_pnl_attribution(path: np.ndarray, order: int) -> np.ndarray:
    r"""Return the causal close-``t`` to next-increment signature integral.

    For stopped features ``phi_t`` and residual-path increments ``Delta X``,
    this computes

    ``sum_t phi_t * Delta X_(t+1)``.

    Equivalently, for each path it is
    ``stopped_signature_features(path, order)[:-1].T @ diff(path[:, 1])``.
    This is the attribution matching the project's Daily execution ledger.
    A two-dimensional input returns one vector and a three-dimensional input
    returns a matrix with one row per path.
    """

    _validate_order(order)
    x = np.asarray(path, dtype=float)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None, ...]
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] != _ORIGINAL_DIMENSION:
        raise ValueError("path must have shape (time, 2) or (batch, time, 2)")
    if not np.isfinite(x).all():
        raise ValueError("path must be finite")

    rows = []
    for sample in x:
        features = stopped_signature_features(sample, order)
        increments = np.diff(sample[:, 1])
        rows.append(features[:-1].T @ increments)
    result = np.asarray(rows, dtype=float)
    return result[0] if squeeze else result


def clocked_signature_design(
    path: np.ndarray,
    order: int,
    *,
    execution_lag_bars: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Align stopped features with subsequently earned path increments.

    With ``T`` path increments and lag ``L``, the returned arrays are

    ``features = Phi[: T - L]`` and ``increments = Delta X[L:]``.

    Thus lag zero pairs ``Phi_t`` with the immediately following increment,
    while lag one pairs it with the increment after one complete execution
    bar.  A two-dimensional path returns shapes ``(T-L, p)`` and ``(T-L,)``;
    a batched path returns ``(batch, T-L, p)`` and ``(batch, T-L)``.

    The second output is the residual coordinate of ``path``.  When actual
    executable price PnL or funding differs from that state increment, retain
    the first output and pass the separately aligned economic cashflows to
    :func:`cashflow_attribution`.
    """

    _validate_order(order)
    if (
        isinstance(execution_lag_bars, (bool, np.bool_))
        or not isinstance(execution_lag_bars, (int, np.integer))
        or execution_lag_bars < 0
    ):
        raise ValueError("execution_lag_bars must be a non-negative integer")
    lag = int(execution_lag_bars)
    x = np.asarray(path, dtype=float)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None, ...]
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] != _ORIGINAL_DIMENSION:
        raise ValueError("path must have shape (time, 2) or (batch, time, 2)")
    if not np.isfinite(x).all():
        raise ValueError("path must be finite")
    increment_count = x.shape[1] - 1
    if lag >= increment_count:
        raise ValueError("execution lag leaves no earned path increment")

    feature_stop = increment_count - lag
    feature_rows = np.stack(
        [stopped_signature_features(sample, order)[:feature_stop] for sample in x]
    )
    increments = np.diff(x[:, :, 1], axis=1)[:, lag:]
    if feature_rows.shape[1] != increments.shape[1]:
        raise AssertionError("clocked signature features and increments misalign")
    if squeeze:
        return feature_rows[0], increments[0]
    return feature_rows, increments


def cashflow_attribution(features: np.ndarray, cashflows: np.ndarray) -> np.ndarray:
    r"""Return the linear PnL attribution ``sum_t Phi_t * cashflow_t``.

    This deliberately accepts already aligned *economic* cashflows.  They may
    combine executable leg PnL, funding, or other linear per-unit payments; no
    claim is made that they equal a coordinate increment of the state path.
    Two-dimensional features require a one-dimensional cashflow vector and
    return one attribution vector.  Batched three-dimensional features require
    a matching two-dimensional cashflow matrix and return one vector per path.
    """

    phi = np.asarray(features, dtype=float)
    payments = np.asarray(cashflows, dtype=float)
    if not np.isfinite(phi).all() or not np.isfinite(payments).all():
        raise ValueError("features and cashflows must be finite")
    if phi.ndim == 2:
        if phi.shape[0] < 1 or phi.shape[1] < 1 or payments.shape != (phi.shape[0],):
            raise ValueError(
                "two-dimensional features require one cashflow per feature row"
            )
        return phi.T @ payments
    if phi.ndim == 3:
        if (
            phi.shape[0] < 1
            or phi.shape[1] < 1
            or phi.shape[2] < 1
            or payments.shape != phi.shape[:2]
        ):
            raise ValueError(
                "batched features require cashflows with shape (batch, time)"
            )
        return np.einsum("btf,bt->bf", phi, payments, optimize=True)
    raise ValueError("features must have shape (time, p) or (batch, time, p)")


def fit_signature_mean_variance(
    attributions: np.ndarray,
    config: SignatureRKHSConfig | None = None,
    *,
    turnover_gram: np.ndarray | None = None,
) -> SignatureMeanVarianceModel:
    r"""Fit a regularized signature mean-variance functional.

    Let ``g`` be a supplied PnL-attribution vector.  For the project's primary
    Daily clock it comes from :func:`discrete_pnl_attribution`; the separate
    repository-clock diagnostic can use
    :func:`terminal_pnl_attribution`.  We estimate
    ``mu = E[g]`` and ``Sigma = Cov(g)`` and maximize

    ``mu' theta - theta' (Sigma + ridge + turnover) theta``.

    The unconstrained optimizer is one half of the regularized linear solve.
    If ``variance_target`` is set, only its direction matters and it is scaled
    to that empirical PnL variance afterwards.
    """

    cfg = config or SignatureRKHSConfig()
    values = np.asarray(attributions, dtype=float)
    expected_dimension = signature_feature_dimension(cfg.order)
    if values.ndim != 2 or values.shape[1] != expected_dimension:
        raise ValueError(
            f"attributions must have shape (samples, {expected_dimension})"
        )
    if values.shape[0] < 2:
        raise ValueError("at least two attribution samples are required")
    if not np.isfinite(values).all():
        raise ValueError("attributions must be finite")

    mean = values.mean(axis=0)
    covariance = np.asarray(np.cov(values, rowvar=False, ddof=1), dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    average_variance = float(np.trace(covariance) / expected_dimension)
    ridge_scale = max(average_variance, cfg.numerical_floor)
    regularized = covariance + (
        cfg.ridge_ratio * ridge_scale * np.eye(expected_dimension)
    )

    turnover_scale = 0.0
    if turnover_gram is not None:
        gram = np.asarray(turnover_gram, dtype=float)
        if gram.shape != (expected_dimension, expected_dimension):
            raise ValueError(
                "turnover_gram must be square with the signature feature dimension"
            )
        if not np.isfinite(gram).all():
            raise ValueError("turnover_gram must be finite")
        gram = 0.5 * (gram + gram.T)
        eigen_floor = float(np.linalg.eigvalsh(gram).min())
        if eigen_floor < -10.0 * cfg.numerical_floor:
            raise ValueError("turnover_gram must be positive semidefinite")
        turnover_scale = float(np.trace(gram) / expected_dimension)
        if turnover_scale > cfg.numerical_floor and cfg.turnover_penalty > 0:
            regularized = regularized + (
                cfg.turnover_penalty
                * ridge_scale
                * gram
                / turnover_scale
            )

    regularized = 0.5 * (regularized + regularized.T)
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    min_eigenvalue = float(eigenvalues[0])
    max_eigenvalue = max(float(eigenvalues[-1]), cfg.numerical_floor)
    # Clip only for the solve; retain the raw minimum eigenvalue above so an
    # ill-conditioned calibration remains visible in the diagnostics.
    solve_floor = max(
        cfg.numerical_floor,
        cfg.numerical_floor * max_eigenvalue,
    )
    stable_eigenvalues = np.maximum(eigenvalues, solve_floor)
    projected_mean = eigenvectors.T @ mean
    direction = 0.5 * eigenvectors @ (projected_mean / stable_eigenvalues)
    if not np.isfinite(direction).all():
        raise FloatingPointError("mean-variance solve produced non-finite coefficients")

    condition_number = float(stable_eigenvalues[-1] / stable_eigenvalues[0])
    effective_rank = int(np.count_nonzero(eigenvalues > solve_floor))

    variance_before_scaling = max(float(direction @ covariance @ direction), 0.0)
    normalization = 1.0
    coefficients = direction
    if (
        cfg.variance_target is not None
        and variance_before_scaling > cfg.numerical_floor
    ):
        normalization = float(
            np.sqrt(variance_before_scaling / cfg.variance_target)
        )
        coefficients = direction / normalization

    return SignatureMeanVarianceModel(
        config=cfg,
        coefficients=np.asarray(coefficients, dtype=float),
        mean_attribution=mean,
        covariance=covariance,
        regularized_matrix=regularized,
        ridge_scale=ridge_scale,
        turnover_scale=turnover_scale,
        variance_before_scaling=variance_before_scaling,
        normalization=normalization,
        n_samples=int(values.shape[0]),
        condition_number=condition_number,
        effective_rank=effective_rank,
        min_eigenvalue=min_eigenvalue,
    )


def score_signature_path(
    path: np.ndarray,
    model_or_coefficients: SignatureMeanVarianceModel | np.ndarray,
    *,
    order: int | None = None,
) -> np.ndarray:
    """Evaluate a fitted linear signature functional on every stopped prefix."""

    if isinstance(model_or_coefficients, SignatureMeanVarianceModel):
        if order is not None and order != model_or_coefficients.config.order:
            raise ValueError("order conflicts with the fitted model")
        resolved_order = model_or_coefficients.config.order
        coefficients = model_or_coefficients.coefficients
    else:
        if order is None:
            raise ValueError("order is required when raw coefficients are supplied")
        resolved_order = order
        coefficients = np.asarray(model_or_coefficients, dtype=float)
    expected_dimension = signature_feature_dimension(resolved_order)
    if coefficients.shape != (expected_dimension,) or not np.isfinite(
        coefficients
    ).all():
        raise ValueError(
            f"coefficients must be a finite vector of length {expected_dimension}"
        )
    return stopped_signature_features(path, resolved_order) @ coefficients


def feature_turnover_gram(
    feature_paths: Iterable[np.ndarray],
    *,
    include_entry_exit: bool = False,
    drop_terminal_feature: bool = False,
) -> np.ndarray:
    """Estimate ``E[Delta phi Delta phi']`` from stopped feature paths.

    This helper supplies the quadratic approximation to position turnover used
    by :func:`fit_signature_mean_variance`.  Each input must be a two-dimensional
    stopped-feature array with the same number of columns.

    With ``drop_terminal_feature=True``, only ``phi[:-1]`` is considered
    tradeable, matching the close-``t`` to next-increment clock.  With
    ``include_entry_exit=True``, the difference matrix additionally includes
    entry from zero and liquidation back to zero:

    ``D = [phi_0, diff(phi), -phi_T]``.

    The defaults retain the earlier internal-rebalancing-only behaviour.
    """

    differences: list[np.ndarray] = []
    feature_dimension: int | None = None
    for features in feature_paths:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError("each feature path must be finite with at least two rows")
        if feature_dimension is None:
            feature_dimension = values.shape[1]
        elif values.shape[1] != feature_dimension:
            raise ValueError("all feature paths must have the same feature dimension")
        traded = values[:-1] if drop_terminal_feature else values
        if traded.shape[0] < 1:
            raise ValueError("a feature path must contain a tradeable row")
        internal = np.diff(traded, axis=0)
        if include_entry_exit:
            path_differences = np.vstack([traded[0], internal, -traded[-1]])
        else:
            path_differences = internal
        if path_differences.shape[0] > 0:
            differences.append(path_differences)
    if not differences or feature_dimension is None:
        raise ValueError("at least one feature difference is required")
    stacked = np.vstack(differences)
    return np.asarray(stacked.T @ stacked / stacked.shape[0], dtype=float)


def _validate_order(order: int) -> None:
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2 or 3")


def _validate_original_path(path: np.ndarray) -> np.ndarray:
    x = np.asarray(path, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != _ORIGINAL_DIMENSION:
        raise ValueError("path must have shape (time, 2) with at least one point")
    if not np.isfinite(x).all():
        raise ValueError("path must be finite")
    return x


def _signature_with_scalar(path: np.ndarray, order: int) -> np.ndarray:
    """Transparent Chen signature used when ``iisignature`` is unavailable."""

    x = np.asarray(path, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("path must have at least two points and one channel")
    dimension = x.shape[1]
    levels = [np.ones(1)] + [
        np.zeros(dimension**level, dtype=float) for level in range(1, order + 1)
    ]
    for increment in np.diff(x, axis=0):
        segment = [np.ones(1)]
        tensor_power = np.ones(1)
        for level in range(1, order + 1):
            tensor_power = np.kron(tensor_power, increment)
            segment.append(tensor_power / factorial(level))
        old = [level.copy() for level in levels]
        for level in range(order + 1):
            combined = np.zeros(dimension**level if level else 1, dtype=float)
            for split in range(level + 1):
                combined += np.kron(old[level - split], segment[split])
            levels[level] = combined
    return np.concatenate(levels)


def _batch_signature_with_scalar(paths: np.ndarray, order: int) -> np.ndarray:
    x = np.asarray(paths, dtype=float)
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] < 1:
        raise ValueError("paths must have shape (batch, time, channels)")
    if _iisignature is not None:
        raw = np.asarray(_iisignature.sig(x, order), dtype=float)
        return np.column_stack([np.ones(x.shape[0]), raw])
    return np.vstack([_signature_with_scalar(path, order) for path in x])
