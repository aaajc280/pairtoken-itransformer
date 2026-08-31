"""Coin-netted, fixed-quantity execution for multi-pair crypto portfolios.

This module implements the execution convention frozen in
``replications/futter_wiese_crypto_multicoin/RESEARCH_CONTRACT.md``.  Pair
scores are *not* backtested as independent two-leg ledgers.  At every target
time they are first mapped to one coin vector,

``raw_coin_target[t] = hedge_matrix[t] @ pair_target[t]``,

and only actual changes in the constrained, net coin vector trade and pay
costs.  Between those changes the USD notionals drift exactly as fixed USD-M
contract quantities do.

The timestamp convention is deliberately explicit.  A row indexed by ``t``
uses ``funding_rates_at_boundary[t]`` on quantities carried *into* the open at
``t``.  Funding is booked first, the target trade executes at that open, and
``coin_open_to_open_returns[t]`` is then earned from open ``t`` to open
``t + 1``.  This is the contract's ``old position pays, then boundary trade``
rule.  It must not be mixed with a bar-end funding convention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConstraints:
    """Radial coin-portfolio constraints.

    Gross, per-coin and dollar-net limits are always available.  BTC-beta and
    volatility scaling are optional hooks because their inputs must be fitted
    on training data by the walk-forward runner.  Supplying a corresponding
    limit without its causal risk input is an error, rather than silently
    disabling the constraint.
    """

    maximum_total_gross: float = 1.0
    maximum_absolute_coin_weight: float = 0.15
    maximum_absolute_dollar_net: float = 0.02
    maximum_absolute_btc_beta: float | None = None
    target_annualized_volatility: float | None = None

    def __post_init__(self) -> None:
        required = (
            self.maximum_total_gross,
            self.maximum_absolute_coin_weight,
            self.maximum_absolute_dollar_net,
        )
        if any(not np.isfinite(value) or value < 0 for value in required):
            raise ValueError("gross, coin and dollar-net limits must be finite and non-negative")
        optional = (
            self.maximum_absolute_btc_beta,
            self.target_annualized_volatility,
        )
        if any(value is not None and (not np.isfinite(value) or value < 0) for value in optional):
            raise ValueError("optional beta and volatility limits must be finite and non-negative")


@dataclass(frozen=True)
class TargetProjectionResult:
    """Auditable result of pair-to-coin aggregation and radial projection."""

    targets: pd.DataFrame
    raw_targets: pd.DataFrame
    effective_pair_targets: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class MultiCoinLedgerResult:
    """Aggregate bars plus long-form coin and exact daily attribution ledgers."""

    bars: pd.DataFrame
    coin_ledger: pd.DataFrame
    daily_returns: pd.Series
    daily_attribution: pd.DataFrame
    daily_coin_attribution: pd.DataFrame
    metrics: dict[str, float]
    projection: TargetProjectionResult | None = None


HedgeMatrices = Mapping[pd.Timestamp, pd.DataFrame] | np.ndarray
CovarianceMatrices = Mapping[pd.Timestamp, pd.DataFrame] | np.ndarray


def _validate_clock(index: pd.Index, *, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex")
    if index.tz is None:
        raise ValueError(f"{name} index must be timezone-aware")
    result = index.tz_convert("UTC")
    if not result.is_monotonic_increasing or result.has_duplicates:
        raise ValueError(f"{name} index must be strictly increasing and unique")
    return result


def _validate_named_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} columns must be unique")
    result = frame.copy()
    result.index = _validate_clock(result.index, name=name)
    return result


def _asof_frame_tensor(
    values: Mapping[pd.Timestamp, pd.DataFrame],
    clock: pd.DatetimeIndex,
    *,
    rows: Sequence[str],
    columns: Sequence[str],
    name: str,
) -> np.ndarray:
    if not values:
        raise ValueError(f"{name} mapping cannot be empty")
    canonical: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for raw_time, raw_frame in values.items():
        effective = pd.Timestamp(raw_time)
        effective = (
            effective.tz_localize("UTC")
            if effective.tzinfo is None
            else effective.tz_convert("UTC")
        )
        frame = raw_frame.copy()
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError(f"each {name} frame must have unique labels")
        if set(frame.index) != set(rows) or set(frame.columns) != set(columns):
            raise ValueError(f"each {name} frame must contain the exact requested labels")
        frame = frame.loc[list(rows), list(columns)].astype(float)
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} values must be finite")
        canonical.append((effective, frame))
    canonical.sort(key=lambda item: item[0])
    effective_times = pd.DatetimeIndex([item[0] for item in canonical])
    if effective_times.has_duplicates:
        raise ValueError(f"{name} effective timestamps must be unique")
    locations = effective_times.searchsorted(clock, side="right") - 1
    if (locations < 0).any():
        raise ValueError(f"{name} has no causal value at the first target timestamp")
    stack = np.stack([item[1].to_numpy(dtype=float) for item in canonical])
    return stack[locations]


def _coerce_hedges(
    hedge_matrices: HedgeMatrices,
    clock: pd.DatetimeIndex,
    pair_names: Sequence[str],
    *,
    coin_symbols: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(hedge_matrices, np.ndarray):
        tensor = np.asarray(hedge_matrices, dtype=float)
        if tensor.ndim != 3 or tensor.shape[0] != len(clock) or tensor.shape[2] != len(pair_names):
            raise ValueError("hedge_matrices array must have shape (timestamps, coins, pairs)")
        if coin_symbols is None or len(coin_symbols) != tensor.shape[1]:
            raise ValueError("coin_symbols must label every coin in an array hedge tensor")
        coins = tuple(str(symbol) for symbol in coin_symbols)
        if len(set(coins)) != len(coins):
            raise ValueError("coin_symbols must be unique")
        if not np.isfinite(tensor).all():
            raise ValueError("hedge_matrices values must be finite")
        return tensor, coins

    first = next(iter(hedge_matrices.values()), None)
    if first is None:
        raise ValueError("hedge_matrices mapping cannot be empty")
    coins = tuple(str(symbol) for symbol in first.index)
    if coin_symbols is not None and tuple(coin_symbols) != coins:
        raise ValueError("coin_symbols disagree with hedge matrix row labels")
    tensor = _asof_frame_tensor(
        hedge_matrices,
        clock,
        rows=coins,
        columns=pair_names,
        name="hedge_matrices",
    )
    return tensor, coins


def _coerce_availability(
    availability: pd.DataFrame | None,
    clock: pd.DatetimeIndex,
    coins: Sequence[str],
) -> np.ndarray:
    if availability is None:
        return np.ones((len(clock), len(coins)), dtype=bool)
    frame = availability.copy()
    frame.index = _validate_clock(frame.index, name="availability")
    if not frame.index.equals(clock) or set(frame.columns) != set(coins):
        raise ValueError("availability must exactly match the target clock and coins")
    if frame.isna().any().any():
        raise ValueError("availability cannot contain missing values")
    return frame.loc[:, list(coins)].astype(bool).to_numpy()


def _coerce_beta_loadings(
    values: pd.DataFrame | pd.Series | np.ndarray | None,
    clock: pd.DatetimeIndex,
    coins: Sequence[str],
) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, pd.Series):
        if set(values.index) != set(coins):
            raise ValueError("btc_beta_loadings Series must label every coin")
        array = np.broadcast_to(
            values.loc[list(coins)].to_numpy(dtype=float),
            (len(clock), len(coins)),
        ).copy()
    elif isinstance(values, pd.DataFrame):
        frame = values.copy()
        frame.index = _validate_clock(frame.index, name="btc_beta_loadings")
        if not frame.index.equals(clock) or set(frame.columns) != set(coins):
            raise ValueError("btc_beta_loadings must exactly match target clock and coins")
        array = frame.loc[:, list(coins)].to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)
        if array.shape != (len(clock), len(coins)):
            raise ValueError("btc_beta_loadings array has the wrong shape")
    if not np.isfinite(array).all():
        raise ValueError("btc_beta_loadings must be finite")
    return array


def _coerce_covariances(
    values: CovarianceMatrices | None,
    clock: pd.DatetimeIndex,
    coins: Sequence[str],
) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=float)
        if array.shape != (len(clock), len(coins), len(coins)):
            raise ValueError("covariance_matrices array has the wrong shape")
    else:
        array = _asof_frame_tensor(
            values,
            clock,
            rows=coins,
            columns=coins,
            name="covariance_matrices",
        )
    if not np.isfinite(array).all():
        raise ValueError("covariance_matrices must be finite")
    if not np.allclose(array, np.swapaxes(array, 1, 2), rtol=1e-10, atol=1e-12):
        raise ValueError("covariance_matrices must be symmetric")
    return array


def build_netted_coin_targets(
    pair_targets: pd.DataFrame,
    hedge_matrices: HedgeMatrices,
    *,
    coin_symbols: Sequence[str] | None = None,
    availability: pd.DataFrame | None = None,
    constraints: PortfolioConstraints = PortfolioConstraints(),
    btc_beta_loadings: pd.DataFrame | pd.Series | np.ndarray | None = None,
    covariance_matrices: CovarianceMatrices | None = None,
    covariance_annualization: float = 365.0 * 24.0,
) -> TargetProjectionResult:
    """Map pair targets to coins, disable broken pairs, then project radially.

    ``hedge_matrices`` may be a dense ``(time, coin, pair)`` tensor or a
    mapping from causal effective timestamps to coin-by-pair DataFrames.  A
    mapping is expanded with an as-of join; future hedges are never selected.

    If any nonzero leg of a pair is unavailable at a timestamp, that complete
    pair target is set to zero.  This closes both sides of a delisted pair and
    avoids leaving an unintended orphan leg.

    Projection is a single, deterministic scale from the origin.  It preserves
    all cross-coin ratios and uses the smallest of the gross, position,
    dollar-net, optional BTC-beta and optional volatility scales.  Covariances
    and beta loadings are accepted only as precomputed inputs; fitting them is
    intentionally outside this execution core so the runner can enforce
    train-only provenance.
    """

    q_frame = _validate_named_frame(pair_targets, name="pair_targets").astype(float)
    if not np.isfinite(q_frame.to_numpy(dtype=float)).all():
        raise ValueError("pair_targets must be finite")
    clock = q_frame.index
    pair_names = tuple(str(name) for name in q_frame.columns)
    h, coins = _coerce_hedges(
        hedge_matrices,
        clock,
        pair_names,
        coin_symbols=coin_symbols,
    )
    available = _coerce_availability(availability, clock, coins)
    beta = _coerce_beta_loadings(btc_beta_loadings, clock, coins)
    covariance = _coerce_covariances(covariance_matrices, clock, coins)
    if constraints.maximum_absolute_btc_beta is not None and beta is None:
        raise ValueError("maximum_absolute_btc_beta requires train-only btc_beta_loadings")
    if constraints.target_annualized_volatility is not None and covariance is None:
        raise ValueError("target_annualized_volatility requires train-only covariance_matrices")
    if not np.isfinite(covariance_annualization) or covariance_annualization <= 0:
        raise ValueError("covariance_annualization must be finite and positive")

    q = q_frame.to_numpy(dtype=float)
    leg_present = np.abs(h) > 0.0
    broken_pairs = np.any(leg_present & ~available[:, :, None], axis=1)
    effective_q = np.where(broken_pairs, 0.0, q)
    raw = np.einsum("tcp,tp->tc", h, effective_q)
    raw = np.where(available, raw, 0.0)

    projected = np.zeros_like(raw)
    rows: list[dict[str, float]] = []
    for row in range(len(clock)):
        vector = raw[row]
        gross = float(np.abs(vector).sum())
        maximum_coin = float(np.abs(vector).max(initial=0.0))
        dollar_net = float(vector.sum())
        scales = [1.0]
        if constraints.target_annualized_volatility is not None:
            variance = float(vector @ covariance[row] @ vector)
            if variance < -1e-12:
                raise ValueError("covariance input implies a materially negative portfolio variance")
            raw_volatility = float(np.sqrt(max(variance, 0.0) * covariance_annualization))
            volatility_scale = (
                constraints.target_annualized_volatility / raw_volatility
                if raw_volatility > 0.0
                else 1.0
            )
            scales[0] = volatility_scale
        else:
            raw_volatility = float("nan")
            volatility_scale = float("nan")
        if gross > 0.0:
            scales.append(constraints.maximum_total_gross / gross)
        if maximum_coin > 0.0:
            scales.append(constraints.maximum_absolute_coin_weight / maximum_coin)
        if abs(dollar_net) > 0.0:
            scales.append(constraints.maximum_absolute_dollar_net / abs(dollar_net))
        raw_beta = float(vector @ beta[row]) if beta is not None else float("nan")
        if constraints.maximum_absolute_btc_beta is not None and abs(raw_beta) > 0.0:
            scales.append(constraints.maximum_absolute_btc_beta / abs(raw_beta))
        scale = float(max(0.0, min(scales)))
        constrained = vector * scale
        projected[row] = constrained
        constrained_variance = (
            float(constrained @ covariance[row] @ constrained)
            if covariance is not None
            else float("nan")
        )
        rows.append(
            {
                "disabled_pair_count": float(broken_pairs[row].sum()),
                "raw_gross": gross,
                "raw_maximum_coin": maximum_coin,
                "raw_dollar_net": dollar_net,
                "raw_btc_beta": raw_beta,
                "raw_annualized_volatility": raw_volatility,
                "volatility_scale": volatility_scale,
                "projection_scale": scale,
                "gross": float(np.abs(constrained).sum()),
                "maximum_coin": float(np.abs(constrained).max(initial=0.0)),
                "dollar_net": float(constrained.sum()),
                "btc_beta": (
                    float(constrained @ beta[row]) if beta is not None else float("nan")
                ),
                "annualized_volatility": (
                    float(np.sqrt(max(constrained_variance, 0.0) * covariance_annualization))
                    if covariance is not None
                    else float("nan")
                ),
            }
        )

    tolerance = 1e-12
    if (np.abs(projected).sum(axis=1) > constraints.maximum_total_gross + tolerance).any():
        raise AssertionError("gross projection constraint failed")
    if (np.abs(projected).max(axis=1) > constraints.maximum_absolute_coin_weight + tolerance).any():
        raise AssertionError("per-coin projection constraint failed")
    if (np.abs(projected.sum(axis=1)) > constraints.maximum_absolute_dollar_net + tolerance).any():
        raise AssertionError("dollar-net projection constraint failed")
    if constraints.maximum_absolute_btc_beta is not None:
        exposures = np.abs(np.einsum("tc,tc->t", projected, beta))
        if (exposures > constraints.maximum_absolute_btc_beta + tolerance).any():
            raise AssertionError("BTC-beta projection constraint failed")

    return TargetProjectionResult(
        targets=pd.DataFrame(projected, index=clock, columns=coins),
        raw_targets=pd.DataFrame(raw, index=clock, columns=coins),
        effective_pair_targets=pd.DataFrame(effective_q, index=clock, columns=pair_names),
        diagnostics=pd.DataFrame(rows, index=clock),
    )


def _align_coin_frame(
    frame: pd.DataFrame,
    clock: pd.DatetimeIndex,
    coins: Sequence[str],
    *,
    name: str,
) -> pd.DataFrame:
    result = frame.copy()
    result.index = _validate_clock(result.index, name=name)
    if not result.index.equals(clock) or set(result.columns) != set(coins):
        raise ValueError(f"{name} must exactly match the target clock and coins")
    return result.loc[:, list(coins)].astype(float)


def _align_sparse_funding(
    funding: pd.DataFrame,
    clock: pd.DatetimeIndex,
    coins: Sequence[str],
) -> pd.DataFrame:
    result = funding.copy()
    result.index = _validate_clock(result.index, name="funding_rates_at_boundary")
    if set(result.columns) != set(coins):
        raise ValueError("funding_rates_at_boundary must contain the exact coin columns")
    if not result.index.isin(clock).all():
        raise ValueError("funding timestamps must be members of the execution clock")
    result = result.loc[:, list(coins)].astype(float)
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError("funding rates cannot contain infinities")
    # Only a timestamp absent from the sparse event table is a genuine
    # no-event row.  A NaN on a supplied union-event row means that this
    # symbol's funding archive is unknown and must remain distinguishable
    # from an observed zero rate.
    event_rows = clock.isin(result.index)
    aligned = result.reindex(clock)
    aligned.loc[~event_rows, :] = 0.0
    return aligned


def _daily_attribution(bars: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    dates = bars.index.normalize()
    prior_day_growth = np.empty(len(bars), dtype=float)
    running = 1.0
    previous_date: pd.Timestamp | None = None
    for row, (date, net_return) in enumerate(zip(dates, bars["net_return"], strict=True)):
        if previous_date is None or date != previous_date:
            running = 1.0
        prior_day_growth[row] = running
        running *= 1.0 + float(net_return)
        previous_date = date
    attribution = pd.DataFrame(
        {
            "price_contribution": bars["price_return"].to_numpy() * prior_day_growth,
            "funding_contribution": bars["funding_return"].to_numpy() * prior_day_growth,
            "transaction_cost_contribution": -bars["transaction_cost"].to_numpy()
            * prior_day_growth,
        },
        index=bars.index,
    )
    daily_attribution = attribution.groupby(dates).sum()
    daily_attribution["net_return"] = daily_attribution.sum(axis=1)
    daily = (1.0 + bars["net_return"]).groupby(dates).prod() - 1.0
    daily.index.name = "date"
    daily_attribution.index.name = "date"
    if not np.allclose(
        daily_attribution["net_return"], daily, rtol=1e-12, atol=1e-14
    ):
        raise AssertionError("daily return attribution does not reconcile")
    return daily, daily_attribution, prior_day_growth


def _performance_metrics(bars: pd.DataFrame, daily: pd.Series) -> dict[str, float]:
    mean = float(daily.mean())
    volatility = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")
    sharpe = (
        float(np.sqrt(365.0) * mean / volatility)
        if np.isfinite(volatility) and volatility > 0.0
        else float("nan")
    )
    years = len(daily) / 365.0
    compounded = float((1.0 + daily).prod())
    annual_return = (
        float(compounded ** (1.0 / years) - 1.0)
        if years > 0.0 and compounded > 0.0
        else float("nan")
    )
    nav = (1.0 + daily).cumprod()
    drawdown = nav.div(nav.cummax()).sub(1.0)
    return {
        "daily_observations": float(len(daily)),
        "annual_return": annual_return,
        "annual_volatility": float(volatility * np.sqrt(365.0)),
        "sharpe": sharpe,
        "maximum_drawdown": float(drawdown.min()),
        "total_return": compounded - 1.0,
        "total_turnover": float(bars["turnover"].sum()),
        "annualized_turnover": float(bars["turnover"].sum() / years),
        "total_transaction_cost": float(bars["transaction_cost"].sum()),
        "total_funding_return": float(bars["funding_return"].sum()),
        "mean_gross_exposure": float(bars["gross_exposure"].mean()),
        "maximum_gross_exposure": float(bars["gross_exposure"].max()),
        "target_change_count": float(bars["target_changed"].sum()),
        "forced_closure_count": float(bars["forced_closure"].sum()),
    }


def backtest_netted_coin_ledger(
    target_weights: pd.DataFrame,
    coin_open_to_open_returns: pd.DataFrame,
    funding_rates_at_boundary: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    availability: pd.DataFrame | None = None,
    liquidate_at_end: bool = True,
) -> MultiCoinLedgerResult:
    """Execute an already-netted coin target with fixed contract quantities.

    Timeline at each timestamp ``t``:

    1. apply ``funding_rates_at_boundary[t]`` to quantities carried into ``t``;
    2. if and only if the *entire net target vector* changed, trade at open t;
    3. earn ``coin_open_to_open_returns[t]`` from t to t+1;
    4. after the final return, optionally liquidate every coin.

    Positive funding is paid by a long and received by a short.  Funding input
    may be sparse on the execution clock; missing non-event rows are filled
    with zero.  Return input must be dense except for an unavailable coin,
    which is forced flat before its missing forward return.
    """

    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    targets_frame = _validate_named_frame(target_weights, name="target_weights").astype(float)
    if not np.isfinite(targets_frame.to_numpy(dtype=float)).all():
        raise ValueError("target_weights must be finite")
    clock = targets_frame.index
    coins = tuple(str(name) for name in targets_frame.columns)
    returns_frame = _align_coin_frame(
        coin_open_to_open_returns,
        clock,
        coins,
        name="coin_open_to_open_returns",
    )
    funding_frame = _align_sparse_funding(funding_rates_at_boundary, clock, coins)
    available = _coerce_availability(availability, clock, coins)
    returns = returns_frame.to_numpy(dtype=float)
    funding = funding_frame.to_numpy(dtype=float)
    invalid_available_returns = available & ~np.isfinite(returns)
    if invalid_available_returns.any():
        raise ValueError("available coins require finite open-to-open returns")
    if ((returns <= -1.0) & available).any():
        raise ValueError("available coin returns must be greater than -100%")
    returns = np.where(available, returns, 0.0)
    targets = np.where(available, targets_frame.to_numpy(dtype=float), 0.0)
    cost_rate = float(transaction_cost_bps) / 10_000.0

    nav = 1.0
    carried = np.zeros(len(coins), dtype=float)
    previous_target = np.zeros(len(coins), dtype=float)
    bar_rows: list[dict[str, float | bool]] = []
    coin_rows: list[dict[str, float | str | pd.Timestamp | bool]] = []
    for row, timestamp in enumerate(clock):
        nav_before = nav
        if nav_before <= 0.0 or not np.isfinite(nav_before):
            raise ValueError("portfolio NAV is invalid")
        carried_before = carried.copy()
        unknown_funding = ~np.isfinite(funding[row])
        if np.any(unknown_funding & (np.abs(carried_before) > 1e-15)):
            unknown = [
                coins[position]
                for position in np.flatnonzero(
                    unknown_funding & (np.abs(carried_before) > 1e-15)
                )
            ]
            raise ValueError(
                f"unknown boundary funding for carried positions at {timestamp}: {unknown}"
            )
        effective_funding = np.where(unknown_funding, 0.0, funding[row])
        funding_pnl_coin = -(carried_before * effective_funding)
        funding_pnl = float(funding_pnl_coin.sum())
        nav_after_funding = nav_before + funding_pnl
        if nav_after_funding <= 0.0 or not np.isfinite(nav_after_funding):
            raise ValueError("funding exhausts portfolio NAV")

        desired = targets[row]
        changed = not np.array_equal(desired, previous_target)
        forced_closure = bool(np.any((~available[row]) & (np.abs(carried_before) > 0.0)))
        start = carried_before.copy()
        trade = np.zeros_like(start)
        entry_cost_dollars = 0.0
        if changed:
            provisional_trade = float(np.abs(desired * nav_after_funding - carried_before).sum())
            provisional_post_cost_nav = nav_after_funding - cost_rate * provisional_trade
            if provisional_post_cost_nav <= 0.0:
                raise ValueError("entry transaction cost exhausts portfolio NAV")
            start = desired * provisional_post_cost_nav
            trade = start - carried_before
            entry_cost_dollars = cost_rate * float(np.abs(trade).sum())
        nav_after_entry = nav_after_funding - entry_cost_dollars
        if nav_after_entry <= 0.0 or not np.isfinite(nav_after_entry):
            raise ValueError("entry transaction cost exhausts portfolio NAV")

        price_pnl_coin = start * returns[row]
        end = start + price_pnl_coin
        price_pnl = float(price_pnl_coin.sum())
        nav_before_liquidation = nav_after_entry + price_pnl
        if nav_before_liquidation <= 0.0 or not np.isfinite(nav_before_liquidation):
            raise ValueError("price PnL exhausts portfolio NAV")

        liquidation_trade = np.zeros_like(end)
        liquidation_cost_dollars = 0.0
        close = end.copy()
        if liquidate_at_end and row == len(clock) - 1:
            liquidation_trade = -end
            liquidation_cost_dollars = cost_rate * float(np.abs(liquidation_trade).sum())
            close[:] = 0.0
        nav = nav_before_liquidation - liquidation_cost_dollars
        if nav <= 0.0 or not np.isfinite(nav):
            raise ValueError("terminal liquidation exhausts portfolio NAV")

        rebalance_turnover = float(np.abs(trade).sum()) / nav_before
        liquidation_turnover = float(np.abs(liquidation_trade).sum()) / nav_before
        turnover = rebalance_turnover + liquidation_turnover
        transaction_cost = (entry_cost_dollars + liquidation_cost_dollars) / nav_before
        price_return = price_pnl / nav_before
        funding_return = funding_pnl / nav_before
        gross_return = price_return + funding_return
        net_return = nav / nav_before - 1.0
        bar_rows.append(
            {
                "nav_before": nav_before,
                "nav_after_funding": nav_after_funding,
                "nav_after_entry_cost": nav_after_entry,
                "nav": nav,
                "target_changed": changed,
                "forced_closure": forced_closure,
                "target_gross": float(np.abs(desired).sum()),
                "target_net": float(desired.sum()),
                "gross_exposure": float(np.abs(start).sum()) / nav_before,
                "net_exposure": float(start.sum()) / nav_before,
                "rebalance_turnover": rebalance_turnover,
                "liquidation_turnover": liquidation_turnover,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "transaction_cost_dollars": entry_cost_dollars + liquidation_cost_dollars,
                "price_return": price_return,
                "funding_return": funding_return,
                "gross_return": gross_return,
                "net_return": net_return,
            }
        )
        for coin, symbol in enumerate(coins):
            coin_cost_dollars = cost_rate * (
                abs(trade[coin]) + abs(liquidation_trade[coin])
            )
            coin_rows.append(
                {
                    "timestamp": timestamp,
                    "coin": symbol,
                    "available": bool(available[row, coin]),
                    "target_weight": desired[coin],
                    "carried_notional": carried_before[coin],
                    "carried_weight": carried_before[coin] / nav_before,
                    "funding_rate_at_boundary": funding[row, coin],
                    "funding_pnl_dollars": funding_pnl_coin[coin],
                    "funding_return": funding_pnl_coin[coin] / nav_before,
                    "trade_notional": trade[coin],
                    "start_notional": start[coin],
                    "open_to_open_return": returns[row, coin],
                    "price_pnl_dollars": price_pnl_coin[coin],
                    "price_return": price_pnl_coin[coin] / nav_before,
                    "end_notional": end[coin],
                    "liquidation_trade_notional": liquidation_trade[coin],
                    "close_notional": close[coin],
                    "turnover": (abs(trade[coin]) + abs(liquidation_trade[coin]))
                    / nav_before,
                    "transaction_cost": coin_cost_dollars / nav_before,
                    "net_contribution": (
                        funding_pnl_coin[coin]
                        + price_pnl_coin[coin]
                        - coin_cost_dollars
                    )
                    / nav_before,
                }
            )
        carried = close
        previous_target = desired.copy()

    bars = pd.DataFrame(bar_rows, index=clock)
    bars.index.name = "timestamp"
    coin_ledger = pd.DataFrame(coin_rows).set_index(["timestamp", "coin"])
    if not np.allclose(
        bars["transaction_cost"],
        cost_rate * bars["turnover"],
        rtol=1e-12,
        atol=1e-14,
    ):
        raise AssertionError("transaction costs do not equal full-L1 cost rate times turnover")
    if not np.allclose(
        bars["net_return"],
        bars["gross_return"] - bars["transaction_cost"],
        rtol=1e-12,
        atol=1e-14,
    ):
        raise AssertionError("bar return attribution does not reconcile")
    coin_sums = coin_ledger.groupby(level="timestamp")[
        ["price_return", "funding_return", "transaction_cost", "net_contribution"]
    ].sum()
    if not np.allclose(coin_sums["price_return"], bars["price_return"], rtol=1e-12, atol=1e-14):
        raise AssertionError("coin price attribution does not reconcile")
    if not np.allclose(coin_sums["funding_return"], bars["funding_return"], rtol=1e-12, atol=1e-14):
        raise AssertionError("coin funding attribution does not reconcile")
    if not np.allclose(coin_sums["transaction_cost"], bars["transaction_cost"], rtol=1e-12, atol=1e-14):
        raise AssertionError("coin cost attribution does not reconcile")

    daily, daily_attribution, prior_day_growth = _daily_attribution(bars)
    coin_reset = coin_ledger.reset_index()
    timestamp_locations = clock.get_indexer(pd.DatetimeIndex(coin_reset["timestamp"]))
    coin_reset["prior_day_growth"] = prior_day_growth[timestamp_locations]
    for component in ("price_return", "funding_return"):
        coin_reset[f"{component}_contribution"] = (
            coin_reset[component] * coin_reset["prior_day_growth"]
        )
    coin_reset["transaction_cost_contribution"] = (
        -coin_reset["transaction_cost"] * coin_reset["prior_day_growth"]
    )
    coin_reset["date"] = pd.DatetimeIndex(coin_reset["timestamp"]).normalize()
    daily_coin = coin_reset.groupby(["date", "coin"])[
        [
            "price_return_contribution",
            "funding_return_contribution",
            "transaction_cost_contribution",
        ]
    ].sum()
    daily_coin["net_return_contribution"] = daily_coin.sum(axis=1)
    if not np.allclose(
        daily_coin.groupby(level="date")["net_return_contribution"].sum(),
        daily,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise AssertionError("daily coin attribution does not reconcile")

    return MultiCoinLedgerResult(
        bars=bars,
        coin_ledger=coin_ledger,
        daily_returns=daily,
        daily_attribution=daily_attribution,
        daily_coin_attribution=daily_coin,
        metrics=_performance_metrics(bars, daily),
    )


def backtest_multicoin_pair_ledger(
    pair_targets: pd.DataFrame,
    hedge_matrices: HedgeMatrices,
    coin_open_to_open_returns: pd.DataFrame,
    funding_rates_at_boundary: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    coin_symbols: Sequence[str] | None = None,
    availability: pd.DataFrame | None = None,
    constraints: PortfolioConstraints = PortfolioConstraints(),
    btc_beta_loadings: pd.DataFrame | pd.Series | np.ndarray | None = None,
    covariance_matrices: CovarianceMatrices | None = None,
    covariance_annualization: float = 365.0 * 24.0,
    liquidate_at_end: bool = True,
) -> MultiCoinLedgerResult:
    """Aggregate pair targets, project constraints and run the common ledger."""

    projection = build_netted_coin_targets(
        pair_targets,
        hedge_matrices,
        coin_symbols=coin_symbols,
        availability=availability,
        constraints=constraints,
        btc_beta_loadings=btc_beta_loadings,
        covariance_matrices=covariance_matrices,
        covariance_annualization=covariance_annualization,
    )
    result = backtest_netted_coin_ledger(
        projection.targets,
        coin_open_to_open_returns,
        funding_rates_at_boundary,
        transaction_cost_bps=transaction_cost_bps,
        availability=availability,
        liquidate_at_end=liquidate_at_end,
    )
    return replace(result, projection=projection)
