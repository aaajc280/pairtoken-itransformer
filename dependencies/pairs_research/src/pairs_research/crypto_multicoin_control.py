"""Stable causal shuffles for the multi-coin path-order negative control.

A global reversal of every residual prefix is not a strong negative control
when a full tensor-signature model is refitted: time reversal induces one
fixed invertible linear map on each complete tensor level.  A sufficiently
flexible refit can therefore undo the control by relabelling coordinates.

The control below instead assigns every ``(pair, 8h episode, decision)`` its
own deterministic within-prefix permutation.  The permutation is generated
from a stable BLAKE2 hash of identifiers known at the decision, followed by a
vectorized SplitMix64 key schedule.  It uses only the already observed prefix,
preserves its total residual increment exactly and is reproducible across
processes without relying on Python's randomized ``hash`` implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import blake2b

import numpy as np
import pandas as pd


PREFIX_SHUFFLE_CONTROL_VERSION = "pair_episode_decision_splitmix64_v1"
_UINT64_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def stable_prefix_control_seed(pair_id: str, anchor: str | pd.Timestamp) -> int:
    """Return a process-stable 64-bit seed for one pair/episode identity."""

    name = str(pair_id).strip()
    if not name:
        raise ValueError("pair_id must not be empty")
    timestamp = _utc(anchor)
    if pd.isna(timestamp):
        raise ValueError("anchor is not a valid timestamp")
    payload = (
        PREFIX_SHUFFLE_CONTROL_VERSION.encode("ascii")
        + b"\x00"
        + name.encode("utf-8")
        + b"\x00"
        + int(timestamp.value).to_bytes(8, byteorder="little", signed=True)
    )
    digest = blake2b(
        payload,
        digest_size=8,
        person=b"pair-prefix-v1",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def stable_prefix_control_seeds(
    pair_ids: Sequence[str], anchors: Sequence[str | pd.Timestamp]
) -> np.ndarray:
    """Vector form of :func:`stable_prefix_control_seed`."""

    names = tuple(str(value) for value in pair_ids)
    dates = tuple(anchors)
    if len(names) != len(dates):
        raise ValueError("pair_ids and anchors must have equal length")
    result = np.fromiter(
        (
            stable_prefix_control_seed(pair_id, anchor)
            for pair_id, anchor in zip(names, dates, strict=True)
        ),
        dtype=np.uint64,
        count=len(names),
    )
    result.setflags(write=False)
    return result


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized fixed-width SplitMix64 finalizer with intentional overflow."""

    with np.errstate(over="ignore"):
        mixed = np.asarray(values, dtype=np.uint64) + np.uint64(_GOLDEN)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(_MIX_A)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(_MIX_B)
        return mixed ^ (mixed >> np.uint64(31))


def deterministic_prefix_shuffle_indices(
    episode_seeds: Sequence[int] | np.ndarray,
    observed_length: int,
) -> np.ndarray:
    """Return one causal permutation of ``0..L-1`` per episode seed.

    ``observed_length`` is part of the key schedule and therefore makes the
    control decision-specific.  No unobserved increment or terminal episode
    value enters the permutation.  Identity is excluded for ``L >= 2`` and
    pure reversal is additionally excluded for ``L >= 3``; this guarantees an
    actual order perturbation on every production prefix (which starts at four
    observed 15-minute increments).
    """

    if (
        isinstance(observed_length, (bool, np.bool_))
        or not isinstance(observed_length, (int, np.integer))
        or observed_length < 0
    ):
        raise ValueError("observed_length must be a non-negative integer")
    seeds = np.asarray(episode_seeds, dtype=np.uint64)
    if seeds.ndim != 1:
        raise ValueError("episode_seeds must be a one-dimensional vector")
    length = int(observed_length)
    if length == 0:
        result = np.empty((len(seeds), 0), dtype=np.int64)
        result.setflags(write=False)
        return result
    if length == 1:
        result = np.zeros((len(seeds), 1), dtype=np.int64)
        result.setflags(write=False)
        return result

    decision_key = _splitmix64(
        np.asarray([(length * _GOLDEN) & _UINT64_MASK], dtype=np.uint64)
    )[0]
    positions = np.arange(1, length + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        raw = seeds[:, None] ^ decision_key ^ (positions[None, :] * np.uint64(_GOLDEN))
    keys = _splitmix64(raw)
    permutation = np.argsort(keys, axis=1, kind="stable").astype(np.int64)

    identity = np.arange(length, dtype=np.int64)
    reverse = identity[::-1]
    degenerate = np.all(permutation == identity, axis=1)
    if length >= 3:
        degenerate |= np.all(permutation == reverse, axis=1)
    for row in np.flatnonzero(degenerate):
        # A nonzero cyclic shift is neither identity nor reversal for L>=3.
        shift = 1 + int(int(seeds[row]) % (length - 1))
        permutation[row] = np.roll(identity, shift)
    permutation.setflags(write=False)
    return permutation


def shuffle_observed_prefixes(
    increments: np.ndarray,
    episode_seeds: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Shuffle each finite observed prefix while preserving every row sum."""

    values = np.asarray(increments, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("increments must be a finite two-dimensional array")
    seeds = np.asarray(episode_seeds, dtype=np.uint64)
    if seeds.shape != (len(values),):
        raise ValueError("episode_seeds must align with increment rows")
    permutation = deterministic_prefix_shuffle_indices(seeds, values.shape[1])
    shuffled = np.take_along_axis(values, permutation, axis=1)
    if not np.allclose(
        shuffled.sum(axis=1), values.sum(axis=1), atol=2e-15, rtol=2e-14
    ):
        raise AssertionError("prefix shuffle did not preserve residual endpoints")
    return shuffled
