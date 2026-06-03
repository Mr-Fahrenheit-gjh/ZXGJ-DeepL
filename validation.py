from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_walk_forward_splits(
    index,
    train_bars: int,
    valid_bars: int,
    step_bars: int,
    expanding: bool = False,
) -> list[dict]:
    """Create chronological walk-forward train/validation slices.

    The split is index-position based, so it works for irregular intraday calendars
    without accidentally assuming weekends, holidays, or lunch breaks are present.
    """
    idx = pd.Index(index)
    n = len(idx)
    if train_bars <= 0 or valid_bars <= 0 or step_bars <= 0:
        raise ValueError("train_bars, valid_bars, and step_bars must be positive")
    splits = []
    start = 0
    fold = 0
    while True:
        train_start = 0 if expanding else start
        train_end = start + train_bars
        valid_start = train_end
        valid_end = valid_start + valid_bars
        if valid_end > n:
            break
        splits.append(
            {
                "fold": fold,
                "train_start_pos": int(train_start),
                "train_end_pos": int(train_end),
                "valid_start_pos": int(valid_start),
                "valid_end_pos": int(valid_end),
                "train_start": str(idx[train_start]),
                "train_end": str(idx[train_end - 1]),
                "valid_start": str(idx[valid_start]),
                "valid_end": str(idx[valid_end - 1]),
                "train_bars": int(train_end - train_start),
                "valid_bars": int(valid_end - valid_start),
                "expanding": bool(expanding),
            }
        )
        fold += 1
        start += step_bars
    return splits


def export_walk_forward_splits(splits: list[dict], output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_df = pd.DataFrame(splits)
    split_df.to_csv(output_dir / "walk_forward_splits.csv", index=False)
    with open(output_dir / "walk_forward_splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    return split_df


def slice_by_walk_forward_split(data: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.iloc[split["train_start_pos"] : split["train_end_pos"]].copy()
    valid = data.iloc[split["valid_start_pos"] : split["valid_end_pos"]].copy()
    return train, valid
