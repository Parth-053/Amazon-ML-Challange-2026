"""
Data loading utilities for Business Entity Resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PipelineConfig


logger = logging.getLogger(__name__)

SOURCE_COLUMNS = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)

GROUND_TRUTH_COLUMNS = (
    "source1_entity_id",
    "matched_entity_ids",
)


def _validate_file(path: Path) -> None:
    """Validate that an input path exists and is a file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")


def _validate_source_columns(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Validate required source columns."""
    missing = set(SOURCE_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing required columns: "
            f"{sorted(missing)}. Available columns: {list(df.columns)}"
        )


def _validate_ground_truth_columns(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Validate required ground-truth columns."""
    missing = set(GROUND_TRUTH_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing required columns: "
            f"{sorted(missing)}. Available columns: {list(df.columns)}"
        )


def _validate_entity_ids(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Ensure source entity IDs are unique."""
    duplicated = df["entity_id"].duplicated(keep=False)

    if duplicated.any():
        duplicate_count = int(duplicated.sum())
        sample_ids = (
            df.loc[duplicated, "entity_id"]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{path.name}: found {duplicate_count:,} rows with duplicate "
            f"entity_id values. Example IDs: {sample_ids}"
        )


def load_source(
    path: Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a source TSV file.

    Returns a DataFrame containing the required source columns with
    entity_id retained as both a column and index.
    """
    path = Path(path)
    _validate_file(path)

    if nrows is not None and nrows <= 0:
        raise ValueError(
            f"nrows must be positive or None, got {nrows}"
        )

    logger.info(
        "Loading %s%s ...",
        path.name,
        f" (first {nrows:,} rows)" if nrows is not None else "",
    )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )

    _validate_source_columns(df, path)
    _validate_entity_ids(df, path)

    df = df.loc[:, SOURCE_COLUMNS].copy()
    df = df.set_index("entity_id", drop=False)

    logger.info(
        "%s: %,d rows loaded.",
        path.name,
        len(df),
    )

    return df


def load_ground_truth(
    path: Path,
) -> dict[str, set[str]]:
    """
    Load ground truth as:

        {s1_entity_id: {matched_entity_ids}}
    """
    path = Path(path)
    _validate_file(path)

    logger.info("Loading ground truth from %s ...", path.name)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    _validate_ground_truth_columns(df, path)

    duplicate_s1 = df["source1_entity_id"].duplicated(keep=False)

    if duplicate_s1.any():
        duplicate_count = int(duplicate_s1.sum())
        sample_ids = (
            df.loc[duplicate_s1, "source1_entity_id"]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{path.name}: found {duplicate_count:,} rows with duplicate "
            f"source1_entity_id values. Example IDs: {sample_ids}"
        )

    gt: dict[str, set[str]] = {}

    for s1_id, matched_ids in zip(
        df["source1_entity_id"],
        df["matched_entity_ids"],
        strict=True,
    ):
        if matched_ids.strip():
            matches = {
                entity_id.strip()
                for entity_id in matched_ids.split(",")
                if entity_id.strip()
            }
        else:
            matches = set()

        gt[s1_id] = matches

    n_with_matches = sum(
        1 for matches in gt.values() if matches
    )
    n_without_matches = len(gt) - n_with_matches

    logger.info(
        "Ground truth: %,d S1 entities.",
        len(gt),
    )
    logger.info(
        "With matches: %,d; without matches: %,d.",
        n_with_matches,
        n_without_matches,
    )

    return gt


def _collect_required_match_ids(
    ground_truth: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    """Extract required S2 and S3 IDs from ground truth."""
    required_s2: set[str] = set()
    required_s3: set[str] = set()

    for matched_ids in ground_truth.values():
        for entity_id in matched_ids:
            if entity_id.startswith("S2-"):
                required_s2.add(entity_id)
            elif entity_id.startswith("S3-"):
                required_s3.add(entity_id)

    return required_s2, required_s3


def _load_required_rows(
    path: Path,
    required_ids: set[str],
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """Load specific entity rows from a large TSV using chunks."""
    if not required_ids:
        return pd.DataFrame(columns=SOURCE_COLUMNS)

    path = Path(path)
    _validate_file(path)

    remaining = set(required_ids)
    recovered_chunks: list[pd.DataFrame] = []

    logger.info(
        "Recovering %,d required matches from %s ...",
        len(remaining),
        path.name,
    )

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=list(SOURCE_COLUMNS),
        chunksize=chunk_size,
    ):
        mask = chunk["entity_id"].isin(remaining)

        if not mask.any():
            continue

        recovered = chunk.loc[mask].copy()
        recovered_chunks.append(recovered)

        remaining.difference_update(
            recovered["entity_id"]
        )

        if not remaining:
            break

    if not recovered_chunks:
        logger.warning(
            "Could not recover any required rows from %s.",
            path.name,
        )
        return pd.DataFrame(columns=SOURCE_COLUMNS)

    recovered_df = pd.concat(
        recovered_chunks,
        ignore_index=True,
    )

    if remaining:
        logger.warning(
            "Could not recover %,d of %,d requested IDs from %s.",
            len(remaining),
            len(required_ids),
            path.name,
        )

    return recovered_df


def _append_recovered_rows(
    original: pd.DataFrame,
    recovered: pd.DataFrame,
) -> pd.DataFrame:
    """Append recovered rows without duplicating existing entity IDs."""
    if recovered.empty:
        return original

    recovered = recovered.loc[
        ~recovered["entity_id"].isin(original.index)
    ]

    if recovered.empty:
        return original

    recovered = recovered.set_index(
        "entity_id",
        drop=False,
    )

    return pd.concat(
        [original, recovered],
        axis=0,
    )


def load_all_train(
    config: PipelineConfig,
    nrows: Optional[int] = None,
    s2_nrows: Optional[int] = None,
    s3_nrows: Optional[int] = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, set[str]],
]:
    """
    Load training S1, S2, S3 and ground truth.

    When source tables are sampled, required true matches are recovered
    from the full S2/S3 files.
    """
    paths = config.paths

    s1 = load_source(
        paths.train_s1,
        nrows=nrows,
    )
    s2 = load_source(
        paths.train_s2,
        nrows=s2_nrows,
    )
    s3 = load_source(
        paths.train_s3,
        nrows=s3_nrows,
    )

    ground_truth = load_ground_truth(
        paths.train_gt
    )

    if nrows is not None:
        s1_ids = set(s1.index)

        ground_truth = {
            s1_id: matches
            for s1_id, matches in ground_truth.items()
            if s1_id in s1_ids
        }

    if nrows is not None and (
        s2_nrows is not None or s3_nrows is not None
    ):
        required_s2, required_s3 = _collect_required_match_ids(
            ground_truth
        )

        missing_s2 = required_s2 - set(s2.index)

        if missing_s2:
            recovered_s2 = _load_required_rows(
                paths.train_s2,
                missing_s2,
            )
            s2 = _append_recovered_rows(
                s2,
                recovered_s2,
            )

        missing_s3 = required_s3 - set(s3.index)

        if missing_s3:
            recovered_s3 = _load_required_rows(
                paths.train_s3,
                missing_s3,
            )
            s3 = _append_recovered_rows(
                s3,
                recovered_s3,
            )

        final_s2_ids = set(s2.index)
        final_s3_ids = set(s3.index)

        unresolved_matches = sum(
            1
            for matched_ids in ground_truth.values()
            for entity_id in matched_ids
            if (
                entity_id.startswith("S2-")
                and entity_id not in final_s2_ids
            )
            or (
                entity_id.startswith("S3-")
                and entity_id not in final_s3_ids
            )
        )

        if unresolved_matches:
            logger.warning(
                "%,d ground-truth match IDs are still absent "
                "from the sampled source tables.",
                unresolved_matches,
            )

    logger.info(
        "Training data loaded: S1=%,d, S2=%,d, S3=%,d, GT=%,d.",
        len(s1),
        len(s2),
        len(s3),
        len(ground_truth),
    )

    return s1, s2, s3, ground_truth


def load_all_test(
    config: PipelineConfig,
    s1_nrows: Optional[int] = None,
    s2_nrows: Optional[int] = None,
    s3_nrows: Optional[int] = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load test S1, S2 and S3 source files."""
    paths = config.paths

    s1 = load_source(
        paths.test_s1,
        nrows=s1_nrows,
    )
    s2 = load_source(
        paths.test_s2,
        nrows=s2_nrows,
    )
    s3 = load_source(
        paths.test_s3,
        nrows=s3_nrows,
    )

    logger.info(
        "Test data loaded: S1=%,d, S2=%,d, S3=%,d.",
        len(s1),
        len(s2),
        len(s3),
    )

    return s1, s2, s3


def entity_level_split(
    ground_truth: dict[str, set[str]],
    val_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
]:
    """
    Split ground truth by Source-1 entity.

    No S1 entity appears in both train and validation.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            "val_fraction must be between 0 and 1."
        )

    if not ground_truth:
        return {}, {}

    import numpy as np

    rng = np.random.default_rng(random_seed)

    all_s1_ids = list(ground_truth.keys())
    rng.shuffle(all_s1_ids)

    n_val = int(len(all_s1_ids) * val_fraction)

    if len(all_s1_ids) > 1:
        n_val = max(1, n_val)
        n_val = min(n_val, len(all_s1_ids) - 1)

    train_gt = {
        s1_id: ground_truth[s1_id]
        for s1_id in all_s1_ids[n_val:]
    }

    val_gt = {
        s1_id: ground_truth[s1_id]
        for s1_id in all_s1_ids[:n_val]
    }

    logger.info(
        "Entity-level split: train=%,d, val=%,d "
        "(val_fraction=%.2f, seed=%d)",
        len(train_gt),
        len(val_gt),
        val_fraction,
        random_seed,
    )

    return train_gt, val_gt


def get_positive_pairs(
    ground_truth: dict[str, set[str]],
) -> pd.DataFrame:
    """Convert ground truth into positive training pairs."""
    rows = [
        (s1_id, candidate_id)
        for s1_id, matched_ids in ground_truth.items()
        for candidate_id in matched_ids
    ]

    if not rows:
        return pd.DataFrame(
            {
                "s1_id": pd.Series(dtype="string"),
                "candidate_id": pd.Series(dtype="string"),
                "label": pd.Series(dtype="int8"),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["s1_id", "candidate_id"],
    )
    df["label"] = 1

    logger.info(
        "Positive pairs: %,d",
        len(df),
    )

    return df


def _country_counts(
    df: pd.DataFrame,
) -> dict[str, int]:
    """Return country frequencies."""
    return {
        str(country): int(count)
        for country, count in df["country"].value_counts().items()
    }


def _missing_string_count(
    df: pd.DataFrame,
    column: str,
) -> int:
    """Count empty or whitespace-only values."""
    return int(
        df[column]
        .astype("string")
        .str.strip()
        .eq("")
        .sum()
    )


def data_summary(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: Optional[dict[str, set[str]]] = None,
) -> dict:
    """Compute summary statistics for a data split."""
    summary = {
        "s1_count": len(s1),
        "s2_count": len(s2),
        "s3_count": len(s3),
        "s1_countries": _country_counts(s1),
        "s2_countries": _country_counts(s2),
        "s3_countries": _country_counts(s3),
        "s2_missing_addr": _missing_string_count(
            s2,
            "business_address",
        ),
        "s3_missing_addr": _missing_string_count(
            s3,
            "business_address",
        ),
    }

    if gt is not None:
        match_counts = [
            len(matches)
            for matches in gt.values()
        ]

        summary["gt_entities"] = len(gt)
        summary["gt_singletons"] = sum(
            count == 0
            for count in match_counts
        )
        summary["gt_with_matches"] = sum(
            count > 0
            for count in match_counts
        )
        summary["gt_avg_matches"] = (
            sum(match_counts) / len(match_counts)
            if match_counts
            else 0.0
        )
        summary["gt_max_matches"] = (
            max(match_counts)
            if match_counts
            else 0
        )
        summary["gt_total_pairs"] = sum(match_counts)

    return summary