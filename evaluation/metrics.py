"""Metric utilities for MindVote evaluation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr, wasserstein_distance

from .data_loader import PollRecord

EPS = 1e-12


def _normalize_distribution(values: Sequence[float]) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = arr.sum()
    if not math.isfinite(total) or total <= 0:
        arr = np.ones_like(arr)
        total = arr.sum()
    return arr / total


def _coerce_percentage(value: object) -> Optional[float]:
    """Best-effort conversion for percentage-style numeric fields."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "guess here":
            return None
        value = stripped.replace("%", "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prediction_from_response(
    parsed_response: Mapping[str, object],
    num_options: int,
) -> Optional[np.ndarray]:
    if not isinstance(parsed_response, Mapping):
        return None

    options = parsed_response.get("options")
    if isinstance(options, list) and options:
        probs = np.zeros(num_options, dtype=float)
        filled = 0
        for idx, item in enumerate(options):
            if not isinstance(item, Mapping):
                continue
            option_idx = item.get("option_index", idx)
            if option_idx is None:
                continue
            option_idx = int(option_idx)
            if option_idx < 0 or option_idx >= num_options:
                continue
            pct_value = _coerce_percentage(item.get("percentage"))
            if pct_value is None:
                continue
            probs[option_idx] = pct_value
            filled += 1
        if filled == num_options:
            return _normalize_distribution(probs)

    if "predicted_distribution" in parsed_response:
        entries = parsed_response["predicted_distribution"]
        if isinstance(entries, list) and entries:
            probs = np.zeros(num_options, dtype=float)
            for idx, item in enumerate(entries):
                if isinstance(item, Mapping):
                    option_idx = item.get("option_index", idx)
                    prob = item.get("probability")
                else:
                    option_idx = idx
                    prob = item
                if option_idx is None or option_idx >= num_options:
                    continue
                if isinstance(prob, str):
                    try:
                        prob = float(prob)
                    except ValueError:
                        prob = 0.0
                probs[int(option_idx)] = float(prob or 0.0)
            return _normalize_distribution(probs)

    if "probabilities" in parsed_response:
        probs = parsed_response["probabilities"]
        if isinstance(probs, list) and len(probs) == num_options:
            return _normalize_distribution(probs)

    if "distribution" in parsed_response:
        probs = parsed_response["distribution"]
        if isinstance(probs, list) and len(probs) == num_options:
            return _normalize_distribution(probs)

    return None


def compute_metrics(ground_truth: Sequence[float], prediction: Sequence[float]) -> Dict[str, float]:
    gt = _normalize_distribution(ground_truth)
    pred = _normalize_distribution(prediction)

    support = np.arange(len(gt))
    wd = wasserstein_distance(support, support, gt, pred)
    kl = entropy(gt + EPS, pred + EPS)
    rho = spearmanr(gt, pred).correlation
    if rho is None or math.isnan(rho):
        rho = 0.0
    acc = float(int(np.argmax(gt)) == int(np.argmax(pred)))

    return {
        "1-wasserstein": 1.0 - wd,
        "1-kl": 1.0 - kl,
        "spearman": float(rho),
        "one_hot_accuracy": acc,
    }


def evaluate_predictions(
    records: Iterable[PollRecord],
    cache_entries: Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    """Compute per-poll metrics from cached model responses."""
    record_map = {rec.poll_id: rec for rec in records}
    metrics_rows: List[Dict[str, object]] = []

    for entry in cache_entries:
        poll_id = entry["poll_id"]
        record = record_map.get(poll_id)
        if record is None:
            continue
        parsed = entry.get("parsed_response")
        if parsed is None:
            continue

        prediction = _prediction_from_response(parsed, num_options=len(record.options))
        if prediction is None:
            continue

        ground_truth = _normalize_distribution(record.vote_counts)
        metric_values = compute_metrics(ground_truth, prediction)
        metrics_rows.append(
            {
                "poll_id": poll_id,
                **metric_values,
            }
        )

    return pd.DataFrame(metrics_rows)


def load_cache_file(cache_path: Path) -> List[Mapping[str, object]]:
    entries: List[Mapping[str, object]] = []
    if not cache_path.exists():
        return entries
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(json.loads(line))
    return entries


def summarize_metrics(metrics_df: pd.DataFrame) -> Dict[str, float]:
    if metrics_df.empty:
        return {key: float("nan") for key in ["1-wasserstein", "1-kl", "spearman", "one_hot_accuracy"]}
    return metrics_df[["1-wasserstein", "1-kl", "spearman", "one_hot_accuracy"]].mean().to_dict()
