"""Walk-forward plumbing for the frozen multi-coin signature experiment.

This module contains only deterministic transformations shared by validation
and outer-test runs.  Model fitting lives in :mod:`crypto_multicoin_signature`
and accounting lives in :mod:`crypto_multicoin_execution`.

The primary execution clock is hourly.  Fifteen-minute bars still define the
state path, but a target is changed only on the declared 60-minute decision
clock.  A funding event stamped at an hourly boundary is passed unchanged to
the execution ledger, where the old carried position pays before the trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .crypto_multicoin_data import (
    MultiCoinDataPanel,
    MultiCoinPairHedge,
    synchronized_hourly_log_returns,
)
from .crypto_multicoin_control import (
    shuffle_observed_prefixes,
    stable_prefix_control_seeds,
)
from .crypto_multicoin_execution import PortfolioConstraints
from .crypto_multicoin_signature import PairCashflowStatistic
from .crypto_multicoin_signature import PooledSignatureCache
from .signature_rkhs import signature_feature_dimension, stopped_signature_features

try:
    import iisignature as _iisignature
except ImportError:  # pragma: no cover - optional acceleration
    _iisignature = None


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


@dataclass(frozen=True)
class HourlyExecutionInputs:
    """Exact hourly inputs consumed by the coin-level ledger."""

    open_prices: pd.DataFrame
    open_to_open_returns: pd.DataFrame
    funding_rates_at_boundary: pd.DataFrame
    availability: pd.DataFrame


@dataclass(frozen=True)
class DailyRiskInputs:
    """Causal daily covariance and BTC-beta maps plus fit diagnostics."""

    covariance_matrices: Mapping[pd.Timestamp, pd.DataFrame]
    btc_beta_loadings: Mapping[pd.Timestamp, pd.Series]
    diagnostics: pd.DataFrame

    def beta_frame(self, clock: pd.DatetimeIndex) -> pd.DataFrame:
        """Expand daily beta estimates to ``clock`` with a causal as-of join."""

        if not self.btc_beta_loadings:
            raise ValueError("BTC-beta mapping is empty")
        effective = pd.DatetimeIndex(sorted(self.btc_beta_loadings))
        locations = effective.searchsorted(clock, side="right") - 1
        if (locations < 0).any():
            raise ValueError("BTC-beta mapping has no causal first value")
        columns = tuple(self.btc_beta_loadings[effective[0]].index.astype(str))
        matrix = np.vstack(
            [
                self.btc_beta_loadings[effective[position]]
                .reindex(columns)
                .to_numpy(dtype=float)
                for position in locations
            ]
        )
        return pd.DataFrame(matrix, index=clock, columns=columns)


@dataclass(frozen=True)
class FastTargetProjection:
    """Numerical equivalent of the audited DataFrame projection core."""

    targets: np.ndarray
    raw_targets: np.ndarray
    effective_pair_targets: np.ndarray
    projection_scale: np.ndarray
    disabled_pair_count: np.ndarray


def build_prefix_permutation_control_cache(
    cache: PooledSignatureCache,
) -> PooledSignatureCache:
    """Stably shuffle each *observed prefix* while preserving its endpoint.

    The shuffle is keyed separately by pair, eight-hour episode and current
    prefix length.  At decision ``j`` it uses only increments strictly before
    ``j``; later increments never enter either the key or the path.  Level one
    is therefore unchanged while temporal interaction terms lose their
    original ordering.

    Unlike one global reversal, this sample-specific transformation is not a
    fixed invertible coordinate relabelling of the complete tensor basis.  A
    full refit cannot undo the negative control with one common coefficient
    transformation.
    """

    observations = cache.episode_count
    decisions = cache.decision_indices
    dimension = signature_feature_dimension(cache.max_order)
    features = np.empty((observations, len(decisions), dimension), dtype=float)
    episode_pair_ids = tuple(cache.pair_ids[int(code)] for code in cache.pair_codes)
    episode_seeds = stable_prefix_control_seeds(episode_pair_ids, tuple(cache.anchors))
    for decision_number, stop in enumerate(decisions):
        stop = int(stop)
        if stop == 0:
            features[:, decision_number] = 0.0
            features[:, decision_number, 0] = 1.0
            continue
        increments = shuffle_observed_prefixes(
            cache.residual_increments[:, :stop], episode_seeds
        )
        cumulative = np.concatenate(
            [np.zeros((observations, 1)), np.cumsum(increments, axis=1)], axis=1
        )
        time = np.broadcast_to(
            np.arange(stop + 1, dtype=float) / 32.0,
            cumulative.shape,
        )
        paths = np.stack([time, cumulative], axis=2)
        if _iisignature is not None:
            signature = np.asarray(
                _iisignature.sig(paths, cache.max_order), dtype=float
            )
            features[:, decision_number, 0] = 1.0
            features[:, decision_number, 1:] = signature
        else:  # pragma: no cover - normal research environment is accelerated
            for row in range(observations):
                features[row, decision_number] = stopped_signature_features(
                    paths[row], cache.max_order
                )[-1]
    features.setflags(write=False)
    return PooledSignatureCache(
        pair_ids=cache.pair_ids,
        anchors=cache.anchors,
        pair_codes=cache.pair_codes,
        residual_increments=cache.residual_increments,
        decision_cashflows=cache.decision_cashflows,
        canonical_features=features,
        decision_indices=cache.decision_indices,
        max_order=cache.max_order,
        rebalance_every_bars=cache.rebalance_every_bars,
    )


def build_hourly_execution_inputs(
    panel: MultiCoinDataPanel,
    *,
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
) -> HourlyExecutionInputs:
    """Build the primary clock without filling a missing price.

    A row at ``t`` normally earns the exact open-to-open return ``t -> t+1h``.
    Decision availability uses only the previous completed 15-minute mask and
    the current execution open; it never consults the next open.  If the next
    hourly open is absent after a terminal source/delisting boundary, the
    realized return is marked to the last observed close inside that hour.
    This outcome-side terminal mark cannot suppress the earlier decision.
    """

    first = _utc(start)
    stop = _utc(end_exclusive)
    if stop <= first:
        raise ValueError("end_exclusive must follow start")
    if first.minute or stop.minute or first.second or stop.second:
        raise ValueError("hourly execution boundaries must lie on whole hours")
    clock = pd.date_range(first, stop, freq="1h", inclusive="left")
    symbols = list(panel.universe.symbols)
    opens = panel.open_prices.reindex(clock).loc[:, symbols].astype(float)
    next_clock = clock + pd.Timedelta(hours=1)
    next_opens = panel.open_prices.reindex(next_clock).loc[:, symbols].astype(float)
    next_opens.index = clock
    hourly_last_close = (
        panel.close_prices.loc[:, symbols]
        .resample("1h", label="left", closed="left")
        .last()
        .reindex(clock)
    )
    realized_exit = next_opens.combine_first(hourly_last_close)
    returns = realized_exit.div(opens).sub(1.0)

    causal_components = (
        "bar_available",
        "coverage_mask",
        "liquidity_mask",
        "archive_available",
    )
    if all(hasattr(panel, name) for name in causal_components):
        # The last bar's close/volume is first known at the current open.  The
        # rolling coverage and daily-liquidity masks are already shifted in
        # the panel builder, so they may be used at t without an unnecessary
        # extra 15-minute lag.
        lagged_bar = (
            panel.bar_available.shift(1).reindex(clock).loc[:, symbols].eq(True)
        )
        current_context = pd.DataFrame(True, index=clock, columns=symbols)
        for name in ("coverage_mask", "liquidity_mask", "archive_available"):
            current_context &= (
                getattr(panel, name).reindex(clock).loc[:, symbols].eq(True)
            )
        causal_tradable = lagged_bar & current_context
    else:
        # Small synthetic callers may expose only the already-composed mask.
        causal_tradable = (
            panel.tradable_mask.shift(1).reindex(clock).loc[:, symbols].eq(True)
        )
    availability = causal_tradable & opens.notna() & opens.gt(0.0)
    unmarkable = availability & (realized_exit.isna() | realized_exit.le(0.0))
    if unmarkable.any().any():
        first_bad = np.argwhere(unmarkable.to_numpy())[0]
        raise ValueError(
            "a causally available position has no next-open or terminal-close mark: "
            f"{clock[first_bad[0]]} {symbols[first_bad[1]]}"
        )
    returns = returns.where(availability)
    if ((returns <= -1.0) & availability).any().any():
        raise ValueError("available hourly returns must be greater than -100%")

    funding = panel.funding_rates.reindex(columns=symbols)
    funding = funding.loc[funding.index.intersection(clock)].sort_index()
    return HourlyExecutionInputs(
        open_prices=opens,
        open_to_open_returns=returns,
        funding_rates_at_boundary=funding,
        availability=availability,
    )


def build_hedge_matrix_mapping(
    estimates: Sequence[MultiCoinPairHedge],
    *,
    coin_symbols: Sequence[str],
    pair_ids: Sequence[str],
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Convert explicit pair-week rows to causal coin-by-pair matrices."""

    coins = tuple(str(value) for value in coin_symbols)
    pairs = tuple(str(value) for value in pair_ids)
    if len(coins) != len(set(coins)) or len(pairs) != len(set(pairs)):
        raise ValueError("coin and pair labels must be unique")
    if not estimates:
        raise ValueError("hedge estimates must not be empty")
    pair_lookup = {pair_id: position for position, pair_id in enumerate(pairs)}
    coin_lookup = {symbol: position for position, symbol in enumerate(coins)}
    grouped: dict[pd.Timestamp, list[MultiCoinPairHedge]] = {}
    for estimate in estimates:
        grouped.setdefault(_utc(estimate.effective_at), []).append(estimate)

    result: dict[pd.Timestamp, pd.DataFrame] = {}
    for effective in sorted(grouped):
        matrix = np.zeros((len(coins), len(pairs)), dtype=float)
        seen: set[str] = set()
        for estimate in grouped[effective]:
            pair_id = estimate.pair.pair_id
            if pair_id not in pair_lookup:
                raise ValueError(f"unknown hedge pair {pair_id!r}")
            if pair_id in seen:
                raise ValueError(f"duplicate hedge pair {pair_id!r} at {effective}")
            seen.add(pair_id)
            if estimate.tradable:
                signal = estimate.pair.signal_symbol
                hedge = estimate.pair.hedge_symbol
                if signal not in coin_lookup or hedge not in coin_lookup:
                    raise ValueError("hedge contains a coin outside the matrix labels")
                column = pair_lookup[pair_id]
                matrix[coin_lookup[signal], column] = float(estimate.signal_weight)
                matrix[coin_lookup[hedge], column] = float(estimate.hedge_weight)
        if seen != set(pairs):
            raise ValueError(f"hedge family is incomplete at {effective}")
        result[effective] = pd.DataFrame(matrix, index=coins, columns=pairs)
    return result


def score_paths_to_pair_frame(
    scores: Mapping[str, Mapping[pd.Timestamp, np.ndarray]],
    *,
    pair_ids: Sequence[str],
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    decision_minutes: int = 60,
) -> pd.DataFrame:
    """Place non-overlapping episode score vectors on one dense pair clock."""

    if decision_minutes < 1 or 60 % decision_minutes:
        raise ValueError("decision_minutes must be a positive divisor of 60")
    first = _utc(start)
    stop = _utc(end_exclusive)
    clock = pd.date_range(first, stop, freq=f"{decision_minutes}min", inclusive="left")
    pairs = tuple(str(value) for value in pair_ids)
    frame = pd.DataFrame(0.0, index=clock, columns=pairs)
    expected_size = 8 * 60 // decision_minutes
    for pair_id, by_anchor in scores.items():
        if pair_id not in frame.columns:
            raise KeyError(f"score map contains unknown pair {pair_id!r}")
        for raw_anchor, raw_values in by_anchor.items():
            anchor = _utc(raw_anchor)
            values = np.asarray(raw_values, dtype=float)
            if values.shape != (expected_size,) or not np.isfinite(values).all():
                raise ValueError("episode score vector has the wrong decision clock")
            times = pd.date_range(
                anchor, periods=expected_size, freq=f"{decision_minutes}min"
            )
            inside = (times >= first) & (times < stop)
            chosen = times[inside]
            if len(chosen) and not np.allclose(
                frame.loc[chosen, pair_id].to_numpy(dtype=float), 0.0
            ):
                raise ValueError("score episodes overlap")
            frame.loc[chosen, pair_id] = values[inside]
    frame.index.name = "timestamp"
    return frame


def constant_pair_target_frame(
    statistics: Sequence[PairCashflowStatistic],
    *,
    pair_ids: Sequence[str],
    clock: pd.DatetimeIndex,
    absolute_target: float = 0.10,
) -> pd.DataFrame:
    """Build the causal per-pair training-mean CONST control."""

    if not np.isfinite(absolute_target) or absolute_target <= 0.0:
        raise ValueError("absolute_target must be finite and positive")
    pairs = tuple(str(value) for value in pair_ids)
    signs = {item.pair_id: int(item.sign) for item in statistics}
    missing = sorted(set(pairs).difference(signs))
    if missing:
        raise ValueError(f"CONST training statistics are missing pairs: {missing}")
    values = np.asarray([absolute_target * signs[pair] for pair in pairs])
    return pd.DataFrame(
        np.broadcast_to(values, (len(clock), len(values))).copy(),
        index=clock,
        columns=pairs,
    )


def project_pair_targets_array(
    pair_targets: np.ndarray,
    hedge_tensor: np.ndarray,
    availability: np.ndarray,
    *,
    constraints: PortfolioConstraints,
    btc_beta_loadings: np.ndarray | None = None,
    covariance_matrices: np.ndarray | None = None,
    covariance_annualization: float = 365.0 * 24.0,
) -> FastTargetProjection:
    """Vectorized, label-free form of ``build_netted_coin_targets``.

    The runner uses this only after all labeled inputs have been aligned once.
    Tests compare it directly with the public audited execution function.
    """

    q = np.asarray(pair_targets, dtype=float)
    h = np.asarray(hedge_tensor, dtype=float)
    available = np.asarray(availability, dtype=bool)
    if q.ndim != 2 or h.shape != (q.shape[0], available.shape[1], q.shape[1]):
        raise ValueError("pair targets, hedge tensor and availability do not align")
    if available.shape[0] != q.shape[0]:
        raise ValueError("availability does not share the target clock")
    if not np.isfinite(q).all() or not np.isfinite(h).all():
        raise ValueError("targets and hedges must be finite")
    beta = (
        None
        if btc_beta_loadings is None
        else np.asarray(btc_beta_loadings, dtype=float)
    )
    covariance = (
        None
        if covariance_matrices is None
        else np.asarray(covariance_matrices, dtype=float)
    )
    if constraints.maximum_absolute_btc_beta is not None:
        if beta is None or beta.shape != available.shape or not np.isfinite(beta).all():
            raise ValueError("BTC-beta constraint requires aligned finite loadings")
    if constraints.target_annualized_volatility is not None:
        expected = (q.shape[0], available.shape[1], available.shape[1])
        if (
            covariance is None
            or covariance.shape != expected
            or not np.isfinite(covariance).all()
        ):
            raise ValueError("volatility target requires aligned finite covariances")
    if not np.isfinite(covariance_annualization) or covariance_annualization <= 0.0:
        raise ValueError("covariance annualization must be positive")

    broken = np.any((np.abs(h) > 0.0) & ~available[:, :, None], axis=1)
    effective_q = np.where(broken, 0.0, q)
    raw = np.einsum("tcp,tp->tc", h, effective_q)
    raw = np.where(available, raw, 0.0)
    gross = np.abs(raw).sum(axis=1)
    maximum_coin = np.abs(raw).max(axis=1)
    dollar_net = raw.sum(axis=1)

    scales: list[np.ndarray] = []
    if constraints.target_annualized_volatility is not None:
        variance = np.einsum("tc,tcd,td->t", raw, covariance, raw)
        if (variance < -1e-12).any():
            raise ValueError("covariance implies a materially negative variance")
        volatility = np.sqrt(np.maximum(variance, 0.0) * covariance_annualization)
        scales.append(
            np.divide(
                constraints.target_annualized_volatility,
                volatility,
                out=np.ones_like(volatility),
                where=volatility > 0.0,
            )
        )
    else:
        scales.append(np.ones(len(q), dtype=float))
    scales.extend(
        [
            np.divide(
                constraints.maximum_total_gross,
                gross,
                out=np.full_like(gross, np.inf),
                where=gross > 0.0,
            ),
            np.divide(
                constraints.maximum_absolute_coin_weight,
                maximum_coin,
                out=np.full_like(maximum_coin, np.inf),
                where=maximum_coin > 0.0,
            ),
            np.divide(
                constraints.maximum_absolute_dollar_net,
                np.abs(dollar_net),
                out=np.full_like(dollar_net, np.inf),
                where=np.abs(dollar_net) > 0.0,
            ),
        ]
    )
    if constraints.maximum_absolute_btc_beta is not None:
        exposure = np.einsum("tc,tc->t", raw, beta)
        scales.append(
            np.divide(
                constraints.maximum_absolute_btc_beta,
                np.abs(exposure),
                out=np.full_like(exposure, np.inf),
                where=np.abs(exposure) > 0.0,
            )
        )
    scale = np.maximum(0.0, np.min(np.vstack(scales), axis=0))
    targets = raw * scale[:, None]
    return FastTargetProjection(
        targets=np.asarray(targets, dtype=float),
        raw_targets=np.asarray(raw, dtype=float),
        effective_pair_targets=np.asarray(effective_q, dtype=float),
        projection_scale=np.asarray(scale, dtype=float),
        disabled_pair_count=broken.sum(axis=1).astype(int),
    )


def estimate_daily_risk_inputs(
    open_prices: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    lookback_days: int = 60,
    minimum_complete_observations: int = 720,
    btc_symbol: str = "BTCUSDT",
) -> DailyRiskInputs:
    """Fit daily Ledoit-Wolf hourly covariance and BTC betas causally.

    The return stamped at an effective boundary is excluded.  After a terminal
    symbol disappearance, the estimate is fit on the surviving columns and
    embedded back into the frozen universe.  Inactive coordinates receive a
    conservative median active variance and zero cross-covariance; execution
    availability independently forces their targets to zero.
    """

    if lookback_days < 1 or minimum_complete_observations < 2:
        raise ValueError("risk lookback and observation floor must be positive")
    values = open_prices.astype(float)
    if btc_symbol not in values.columns:
        raise ValueError("BTC symbol is absent from open prices")
    returns = synchronized_hourly_log_returns(values)
    symbols = tuple(str(column) for column in values.columns)
    first = _utc(start).normalize()
    stop = _utc(end_exclusive)
    effective_times = pd.date_range(first, stop, freq="1D", inclusive="left")
    covariance_map: dict[pd.Timestamp, pd.DataFrame] = {}
    beta_map: dict[pd.Timestamp, pd.Series] = {}
    diagnostics: list[dict[str, object]] = []
    lookback = pd.Timedelta(days=int(lookback_days))

    for effective in effective_times:
        window = returns.loc[
            (returns.index >= effective - lookback) & (returns.index < effective)
        ]
        active = tuple(
            symbol
            for symbol in symbols
            if int(window[symbol].notna().sum()) >= minimum_complete_observations
        )
        if btc_symbol not in active:
            raise ValueError(f"BTC lacks a causal risk fit at {effective}")
        complete = window.loc[:, list(active)].dropna()
        if len(complete) < minimum_complete_observations:
            # Iteratively remove the sparsest non-BTC series until the common
            # sample is adequate.  This only removes risk coordinates; the
            # execution availability mask decides whether they can trade.
            mutable = list(active)
            while len(complete) < minimum_complete_observations and len(mutable) > 1:
                removable = [symbol for symbol in mutable if symbol != btc_symbol]
                if not removable:
                    break
                worst = min(
                    removable,
                    key=lambda symbol: int(window[symbol].notna().sum()),
                )
                mutable.remove(worst)
                complete = window.loc[:, mutable].dropna()
            active = tuple(mutable)
        if len(complete) < minimum_complete_observations:
            raise ValueError(f"risk fit has only {len(complete)} rows at {effective}")

        estimator = LedoitWolf(assume_centered=False).fit(
            complete.loc[:, list(active)].to_numpy(dtype=float)
        )
        active_covariance = np.asarray(estimator.covariance_, dtype=float)
        if not np.isfinite(active_covariance).all():
            raise ValueError("Ledoit-Wolf covariance is non-finite")
        positive_diagonal = np.diag(active_covariance)
        fallback_variance = float(np.median(positive_diagonal[positive_diagonal > 0.0]))
        if not np.isfinite(fallback_variance) or fallback_variance <= 0.0:
            raise ValueError("risk fit has no positive variance")

        covariance = np.zeros((len(symbols), len(symbols)), dtype=float)
        np.fill_diagonal(covariance, fallback_variance)
        positions = [symbols.index(symbol) for symbol in active]
        covariance[np.ix_(positions, positions)] = active_covariance
        btc_position = active.index(btc_symbol)
        btc_variance = float(active_covariance[btc_position, btc_position])
        beta = np.zeros(len(symbols), dtype=float)
        for local, symbol in enumerate(active):
            beta[symbols.index(symbol)] = (
                float(active_covariance[local, btc_position]) / btc_variance
            )
        beta[symbols.index(btc_symbol)] = 1.0

        covariance_map[effective] = pd.DataFrame(
            covariance, index=symbols, columns=symbols
        )
        beta_map[effective] = pd.Series(beta, index=symbols, name=effective)
        diagnostics.append(
            {
                "effective_at": effective,
                "sample_start": effective - lookback,
                "sample_end_exclusive": effective,
                "complete_observations": int(len(complete)),
                "active_coin_count": int(len(active)),
                "active_coins": "|".join(active),
                "minimum_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
                "btc_variance": btc_variance,
            }
        )

    if not covariance_map:
        raise ValueError("risk calendar is empty")
    return DailyRiskInputs(
        covariance_matrices=covariance_map,
        btc_beta_loadings=beta_map,
        diagnostics=pd.DataFrame(diagnostics).set_index("effective_at"),
    )
