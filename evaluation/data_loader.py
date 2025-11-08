"""Utilities for loading and normalizing MindVote poll data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math

import numpy as np
import pandas as pd


def _infer_platform_from_path(path: Path) -> Optional[str]:
    name = path.stem.lower()
    if "reddit" in name:
        return "reddit"
    if "weibo" in name:
        return "weibo"
    return None


def _safe_percentage_to_votes(percentage: Optional[float], total_votes: Optional[int]) -> Optional[int]:
    if percentage is None or total_votes in (None, 0):
        return None
    votes = round((percentage / 100.0) * total_votes)
    return max(votes, 0)


def _safe_votes_to_percentage(votes: Optional[int], total_votes: Optional[int]) -> Optional[float]:
    if votes is None or total_votes in (None, 0):
        return None
    return (votes / total_votes) * 100.0


@dataclass(frozen=True)
class PollRecord:
    """Normalized view of a single poll."""

    poll_id: str
    platform: Optional[str]
    question: str
    options: Tuple[str, ...]
    vote_counts: Tuple[int, ...]
    percentages: Tuple[float, ...]
    total_votes: int
    topic: Optional[str]
    subtopic: Optional[str]
    platform_context: Optional[str]
    topical_context: Optional[str]
    temporal_context: Optional[str]
    source_path: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "poll_id": self.poll_id,
            "platform": self.platform,
            "question": self.question,
            "options": list(self.options),
            "vote_counts": list(self.vote_counts),
            "percentages": list(self.percentages),
            "total_votes": self.total_votes,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "platform_context": self.platform_context,
            "topical_context": self.topical_context,
            "temporal_context": self.temporal_context,
            "source_path": self.source_path,
        }


@dataclass
class DatasetSplit:
    """Container for deterministic dataset splits."""

    train: pd.DataFrame
    eval: pd.DataFrame
    records: List[PollRecord]


@dataclass
class DataLoaderConfig:
    data_paths: Sequence[Path]
    holdout_fraction: float = 0.2
    random_seed: int = 42


class DatasetLoader:
    """Loads poll JSON files, normalizes schema, and creates deterministic splits."""

    def __init__(self, config: DataLoaderConfig):
        if not config.data_paths:
            raise ValueError("At least one data path must be provided.")
        if not 0.0 <= config.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in [0, 1).")
        self.config = config

    def load(self) -> DatasetSplit:
        records = []
        for path in self.config.data_paths:
            records.extend(self._load_file(path))

        df = pd.DataFrame([rec.to_dict() for rec in records])
        df["poll_index"] = np.arange(len(df))

        rng = np.random.default_rng(self.config.random_seed)
        shuffled = rng.permutation(len(df))
        split_idx = math.floor(len(df) * (1 - self.config.holdout_fraction))
        train_idx = shuffled[:split_idx]
        eval_idx = shuffled[split_idx:]

        df["split"] = "train"
        if len(eval_idx) > 0:
            df.loc[eval_idx, "split"] = "eval"

        train_df = df[df["split"] == "train"].reset_index(drop=True)
        eval_df = df[df["split"] == "eval"].reset_index(drop=True)
        return DatasetSplit(train=train_df, eval=eval_df, records=records)

    def _load_file(self, path: Path) -> List[PollRecord]:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        records: List[PollRecord] = []
        platform = _infer_platform_from_path(path)

        for idx, poll in enumerate(payload):
            poll_id = str(poll.get("poll_id") or poll.get("id") or f"{path.stem}-{idx}")
            question = str(poll.get("question") or "").strip()
            if not question:
                continue

            total_votes = poll.get("totalVotes") or poll.get("total_votes")

            options_in = poll.get("options") or []
            norm_texts: List[str] = []
            norm_votes: List[int] = []
            norm_percentages: List[float] = []

            for opt_idx, raw_option in enumerate(options_in):
                option_text = str(raw_option.get("optionText") or raw_option.get("text") or "").strip()
                if not option_text:
                    option_text = f"Option {opt_idx + 1}"

                votes = raw_option.get("votes")
                percentage = raw_option.get("percentage")

                if votes is None and percentage is not None and total_votes:
                    votes = _safe_percentage_to_votes(percentage, total_votes)
                if percentage is None and votes is not None and total_votes:
                    percentage = _safe_votes_to_percentage(votes, total_votes)

                norm_texts.append(option_text)
                norm_votes.append(int(votes or 0))
                norm_percentages.append(float(percentage or 0.0))

            if total_votes in (None, 0):
                vote_sum = int(np.sum(norm_votes))
                total_votes = vote_sum if vote_sum > 0 else None

            if total_votes is None:
                continue

            norm_percentages = [
                (votes / total_votes) * 100.0 if total_votes else 0.0 for votes in norm_votes
            ]

            records.append(
                PollRecord(
                    poll_id=poll_id,
                    platform=poll.get("platform") or platform,
                    question=question,
                    options=tuple(norm_texts),
                    vote_counts=tuple(norm_votes),
                    percentages=tuple(norm_percentages),
                    total_votes=total_votes,
                    topic=poll.get("topic"),
                    subtopic=poll.get("subtopic"),
                    platform_context=poll.get("platform_context") or poll.get("platformContext"),
                    topical_context=poll.get("topical_context") or poll.get("topicalContext"),
                    temporal_context=poll.get("temporal_context") or poll.get("temporalContext"),
                    source_path=str(path),
                )
            )

        return records


def load_default_dataset(
    data_dir: Path = Path("data"),
    holdout_fraction: float = 0.2,
    random_seed: int = 42,
) -> DatasetSplit:
    """Helper for quickly loading the default MindVote dataset."""
    json_paths = sorted(data_dir.glob("*.json"))
    config = DataLoaderConfig(
        data_paths=json_paths,
        holdout_fraction=holdout_fraction,
        random_seed=random_seed,
    )
    loader = DatasetLoader(config)
    return loader.load()
