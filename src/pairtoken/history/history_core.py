"""Pure contract, calendar, performance, and robustness helpers.

The GPU runner is deliberately thin around these functions.  Keeping the
calendar and result analysis independent of Torch and the audited execution
snapshot makes the retrospective label, continuous-clock rule, and statistical
summaries cheap to test before any long fit is launched.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contract.json"
HARD_CUTOFF_EXCLUSIVE = pd.Timestamp("2025-07-01T00:00:00Z")
EXPECTED_FOLDS = tuple(
    f"{year}Q{quarter}"
    for year, quarters in ((2022, (2, 3, 4)), (2023, (1, 2, 3, 4)),
                           (2024, (1, 2, 3, 4)), (2025, (1, 2)))
    for quarter in quarters
)
EXPECTED_SERIES = ("B00", "N02", "C02")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("contract_name") != "crypto_pairtoken_1m_historical_robustness":
        raise ValueError("history contract name differs")
    if contract.get("contract_version") != 1:
        raise ValueError("history contract version differs")
    claim = contract.get("claim_boundary", {})
    if claim.get("sample_role") != "retrospective_development_robustness" or claim.get("not_confirmatory") is not True:
        raise ValueError("history claim boundary is not retrospective and non-confirmatory")
    data = contract.get("data", {})
    if utc(data.get("hard_cutoff_exclusive_utc")) != HARD_CUTOFF_EXCLUSIVE:
        raise ValueError("history hard cutoff differs")
    if data.get("confirmation_rows_permitted") is not False:
        raise ValueError("history contract permits confirmation rows")
    folds = list(contract.get("folds", []))
    names = tuple(str(row.get("name")) for row in folds)
    if names != EXPECTED_FOLDS:
        raise ValueError("history folds are not exact 2022Q2--2025Q2 family")
    starts = [utc(row["test_start"]) for row in folds]
    ends = [utc(row["test_end_exclusive"]) for row in folds]
    if starts[0] != utc("2022-04-08T02:00:00Z") or ends[-1] != utc("2025-06-30T23:00:00Z"):
        raise ValueError("history boundary hours differ")
    if any(left != right for left, right in zip(ends[:-1], starts[1:], strict=True)):
        raise ValueError("history folds are not contiguous")
    if any(start >= end for start, end in zip(starts, ends, strict=True)):
        raise ValueError("history contains an empty fold")
    if any(end > HARD_CUTOFF_EXCLUSIVE for end in ends):
        raise PermissionError("history fold reaches confirmation")
    walk = contract.get("walk_forward", {})
    if (walk.get("training_days"), walk.get("validation_days"), walk.get("target_purge_hours")) != (365, 90, 1):
        raise ValueError("history walk-forward window differs")
    if walk.get("positions_forced_flat_at_refit") is not False:
        raise ValueError("history contract liquidates at refits")
    model = contract.get("model", {})
    if tuple(model.get("fixed_seeds", [])) != (1701, 1702, 1703):
        raise ValueError("history fixed seeds differ")
    if model.get("declared_fit_count") != len(folds) * 7:
        raise ValueError("history declared fit count differs")
    portfolio = contract.get("portfolio", {})
    expected_portfolio = {
        "long_pairs": 8,
        "short_pairs": 8,
        "absolute_pair_target": 0.05,
        "maximum_total_gross": 1.0,
        "maximum_absolute_coin_weight": 0.15,
        "maximum_absolute_dollar_net": 0.02,
        "maximum_absolute_btc_beta": 0.05,
        "target_annualized_volatility": 0.1,
        "cost_bps": [0.0, 0.5, 1.5, 5.0],
    }
    for key, expected in expected_portfolio.items():
        if portfolio.get(key) != expected:
            raise ValueError(f"history portfolio setting differs: {key}")
    if contract.get("outputs", {}).get("confirmation_parsed") is not False:
        raise ValueError("history output declaration parses confirmation")
    amendments = contract.get("amendments")
    if not isinstance(amendments, list) or len(amendments) != 1:
        raise ValueError("history pre-fit amendment record differs")
    amendment = amendments[0]
    required_amendment = {
        "amendment_id": "A01_prefit_endpoint_history_floor",
        "old_first_test_start": "2022-04-01T02:00:00Z",
        "new_first_test_start": "2022-04-08T02:00:00Z",
        "scientific_choices_changed": False,
        "fit_or_result_observed_before_amendment": False,
        "forecasts_or_performance_artifacts_created_before_amendment": False,
        "result_driven": False,
    }
    if any(amendment.get(field) != value for field, value in required_amendment.items()):
        raise ValueError("history pre-fit amendment is not outcome-free")


def expected_decision_clock(contract: Mapping[str, Any]) -> pd.DatetimeIndex:
    validate_contract(contract)
    clocks = [
        pd.date_range(utc(row["test_start"]), utc(row["test_end_exclusive"]) - pd.Timedelta(hours=1), freq="1h")
        for row in contract["folds"]
    ]
    clock = clocks[0]
    for addition in clocks[1:]:
        clock = clock.append(addition)
    expected = pd.date_range(clock[0], clock[-1], freq="1h")
    if not clock.equals(expected) or clock.has_duplicates:
        raise ValueError("history decision clock is not one contiguous hourly path")
    if clock[-1] + pd.Timedelta(hours=1) != utc(contract["data"]["boundary_policy"]["terminal_zero_target"]):
        raise ValueError("history decision clock does not precede the terminal row")
    if (clock >= HARD_CUTOFF_EXCLUSIVE).any():
        raise PermissionError("history decision clock reaches confirmation")
    return clock


def expected_fold_clocks(contract: Mapping[str, Any]) -> dict[str, pd.DatetimeIndex]:
    return {
        str(row["name"]): pd.date_range(
            utc(row["test_start"]), utc(row["test_end_exclusive"]) - pd.Timedelta(hours=1), freq="1h"
        )
        for row in contract["folds"]
    }


def fold_for_dates(dates: Sequence[object], contract: Mapping[str, Any]) -> np.ndarray:
    normalized = pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).normalize()
    result = np.full(len(normalized), "", dtype=object)
    for row in contract["folds"]:
        start = utc(row["test_start"]).normalize()
        last = (utc(row["test_end_exclusive"]) - pd.Timedelta(hours=1)).normalize()
        chosen = (normalized >= start) & (normalized <= last)
        result[chosen] = str(row["name"])
    if np.any(result == ""):
        raise ValueError("a daily observation lies outside the declared folds")
    return result.astype(str)


def performance_metrics(
    returns: Iterable[float], *, turnover: float, gross_sum: float, annualization_days: int = 365
) -> dict[str, Any]:
    values = np.asarray(list(returns), dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("performance metrics require a finite non-empty return vector")
    if np.any(values <= -1.0):
        raise ValueError("performance metrics cannot compound a return at or below -100%")
    volatility = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    sharpe = float(np.sqrt(annualization_days) * values.mean() / volatility) if np.isfinite(volatility) and volatility > 0 else float("nan")
    growth = np.cumprod(1.0 + values)
    compounded = float(growth[-1])
    years = len(values) / float(annualization_days)
    peak = np.maximum.accumulate(np.concatenate(([1.0], growth)))[1:]
    return {
        "daily_observations": int(len(values)),
        "mean_daily_return": float(values.mean()),
        "annual_return": float(compounded ** (1.0 / years) - 1.0) if compounded > 0 and years > 0 else float("nan"),
        "annual_volatility": float(volatility * np.sqrt(annualization_days)),
        "sharpe": sharpe,
        "maximum_drawdown": float(np.min(growth / peak - 1.0)),
        "total_return": compounded - 1.0,
        "total_turnover": float(turnover),
        "turnover_per_day": float(turnover / len(values)),
        "linearized_break_even_cost_bps": float(10_000.0 * gross_sum / turnover) if turnover > 0 else float("nan"),
    }


def moving_block_bootstrap(
    values: pd.DataFrame,
    *,
    replications: int = 10_000,
    block_days: int = 7,
    seed: int = 0x13527D8,
) -> dict[str, Any]:
    required = {"date", "difference"}
    if not required.issubset(values.columns):
        raise ValueError("paired bootstrap columns are incomplete")
    data = values[list(required)].copy().sort_values("date")
    data["date"] = pd.to_datetime(data["date"], utc=True).dt.normalize()
    data["difference"] = pd.to_numeric(data["difference"], errors="raise")
    if data.empty or data["date"].duplicated().any() or not np.isfinite(data["difference"]).all():
        raise ValueError("paired bootstrap needs unique finite daily differences")
    expected = pd.date_range(data["date"].iloc[0], data["date"].iloc[-1], freq="1D")
    if not pd.DatetimeIndex(data["date"]).equals(expected):
        raise ValueError("paired bootstrap requires the continuous daily path")
    array = data["difference"].to_numpy(dtype=float)
    block = int(block_days)
    draws = int(replications)
    if block < 1 or block > len(array) or draws < 2:
        raise ValueError("paired bootstrap block or replication count is invalid")
    rng = np.random.default_rng(int(seed))
    offsets = np.arange(block, dtype=np.int64)
    blocks_per_draw = math.ceil(len(array) / block)
    sampled = np.empty(draws, dtype=float)
    for draw in range(draws):
        starts = rng.integers(0, len(array) - block + 1, size=blocks_per_draw)
        locations = (starts[:, None] + offsets[None, :]).ravel()[: len(array)]
        sampled[draw] = float(array[locations].mean())
    observed = float(array.mean())
    return {
        "method": "continuous_non_circular_moving_block_bootstrap",
        "observations": int(len(array)),
        "block_days": block,
        "replications": draws,
        "seed": int(seed),
        "observed_mean_daily_difference": observed,
        "annualized_mean_difference": float(365.0 * observed),
        "bootstrap_mean_daily_difference": float(sampled.mean()),
        "bootstrap_standard_error": float(sampled.std(ddof=1)),
        "one_sided_90pct_lower_bound_daily": float(np.quantile(sampled, 0.10)),
        "two_sided_95pct_lower_bound_daily": float(np.quantile(sampled, 0.025)),
        "two_sided_95pct_upper_bound_daily": float(np.quantile(sampled, 0.975)),
        "interpretation": "descriptive_after_model_selection_not_confirmatory",
    }


def leave_one_fold_out_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"series_id", "cost_bps", "fold", "date", "net_return", "turnover", "gross_return"}
    if not required.issubset(daily.columns):
        raise ValueError("daily table lacks leave-one-fold-out inputs")
    rows: list[dict[str, Any]] = []
    folds = tuple(dict.fromkeys(daily.sort_values("date")["fold"].astype(str)))
    for (series_id, cost), group in daily.groupby(["series_id", "cost_bps"], sort=True):
        for omitted in folds:
            chosen = group.loc[~group["fold"].eq(omitted)].sort_values("date")
            rows.append({
                "series_id": str(series_id),
                "cost_bps": float(cost),
                "omitted_fold": omitted,
                **performance_metrics(chosen["net_return"], turnover=float(chosen["turnover"].sum()), gross_sum=float(chosen["gross_return"].sum())),
            })
    return pd.DataFrame(rows).sort_values(["series_id", "cost_bps", "omitted_fold"]).reset_index(drop=True)


def stability_summary(quarterly: pd.DataFrame, leave_one_out: pd.DataFrame) -> pd.DataFrame:
    required = {"series_id", "cost_bps", "fold", "sharpe", "total_return", "maximum_drawdown"}
    if not required.issubset(quarterly.columns):
        raise ValueError("quarterly metrics lack stability inputs")
    rows: list[dict[str, Any]] = []
    for (series_id, cost), group in quarterly.groupby(["series_id", "cost_bps"], sort=True):
        group = group.sort_values("fold")
        loo = leave_one_out.loc[
            leave_one_out["series_id"].eq(series_id)
            & np.isclose(leave_one_out["cost_bps"], float(cost), rtol=0.0, atol=1e-12)
        ]
        sharpes = group["sharpe"].to_numpy(dtype=float)
        finite = sharpes[np.isfinite(sharpes)]
        worst_position = int(np.nanargmin(sharpes))
        best_position = int(np.nanargmax(sharpes))
        absolute_returns = group["total_return"].abs()
        denominator = float(absolute_returns.sum())
        rows.append({
            "series_id": str(series_id),
            "cost_bps": float(cost),
            "quarters": int(len(group)),
            "positive_return_quarters": int((group["total_return"] > 0).sum()),
            "positive_sharpe_quarters": int((group["sharpe"] > 0).sum()),
            "median_quarter_sharpe": float(np.median(finite)),
            "minimum_quarter_sharpe": float(np.min(finite)),
            "maximum_quarter_sharpe": float(np.max(finite)),
            "quarter_sharpe_iqr": float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)),
            "worst_fold": str(group.iloc[worst_position]["fold"]),
            "best_fold": str(group.iloc[best_position]["fold"]),
            "worst_quarter_drawdown": float(group["maximum_drawdown"].min()),
            "maximum_single_quarter_absolute_return_fraction": float(absolute_returns.max() / denominator) if denominator > 0 else float("nan"),
            "minimum_leave_one_quarter_out_sharpe": float(loo["sharpe"].min()),
            "maximum_leave_one_quarter_out_sharpe": float(loo["sharpe"].max()),
        })
    return pd.DataFrame(rows).sort_values(["series_id", "cost_bps"]).reset_index(drop=True)


def contribution_concentration(coin: pd.DataFrame) -> pd.DataFrame:
    required = {"series_id", "cost_bps", "coin", "net_return_contribution"}
    if not required.issubset(coin.columns):
        raise ValueError("coin table lacks contribution-concentration inputs")
    rows: list[dict[str, Any]] = []
    totals = coin.groupby(["series_id", "cost_bps", "coin"], as_index=False)["net_return_contribution"].sum()
    for (series_id, cost), group in totals.groupby(["series_id", "cost_bps"], sort=True):
        absolute = group["net_return_contribution"].abs().to_numpy(dtype=float)
        denominator = float(absolute.sum())
        shares = absolute / denominator if denominator > 0 else np.zeros_like(absolute)
        leader = int(np.argmax(absolute)) if len(absolute) else 0
        rows.append({
            "series_id": str(series_id),
            "cost_bps": float(cost),
            "coin_count": int(len(group)),
            "largest_absolute_contribution_coin": str(group.iloc[leader]["coin"]),
            "largest_absolute_contribution_fraction": float(shares.max()) if len(shares) else float("nan"),
            "absolute_contribution_hhi": float(np.square(shares).sum()) if denominator > 0 else float("nan"),
            "net_contribution_sum": float(group["net_return_contribution"].sum()),
            "absolute_contribution_sum": denominator,
        })
    return pd.DataFrame(rows).sort_values(["series_id", "cost_bps"]).reset_index(drop=True)


__all__ = [
    "CONTRACT_PATH",
    "EXPECTED_FOLDS",
    "EXPECTED_SERIES",
    "HARD_CUTOFF_EXCLUSIVE",
    "canonical_json",
    "contribution_concentration",
    "expected_decision_clock",
    "expected_fold_clocks",
    "fold_for_dates",
    "leave_one_fold_out_metrics",
    "load_contract",
    "moving_block_bootstrap",
    "performance_metrics",
    "sha256_file",
    "stability_summary",
    "utc",
    "validate_contract",
]
