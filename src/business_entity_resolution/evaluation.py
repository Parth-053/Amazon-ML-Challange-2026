from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BETA = 0.5
BETA_SQ = BETA**2


def f05_single_entity(
    predicted: Set[str],
    truth: Set[str],
) -> float:
    """Compute F0.5 for one Source-1 entity."""

    if not truth and not predicted:
        return 1.0

    if not truth and predicted:
        return 0.0

    if truth and not predicted:
        return 0.0

    tp = len(predicted & truth)

    if tp == 0:
        return 0.0

    precision = tp / len(predicted)
    recall = tp / len(truth)

    denominator = (
        BETA_SQ * precision + recall
    )

    if denominator == 0:
        return 0.0

    return float(
        (1 + BETA_SQ)
        * precision
        * recall
        / denominator
    )


def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, Dict[str, float]]:
    """Compute macro F0.5 across all Source-1 entities."""

    per_entity: Dict[str, float] = {}

    for s1_id, truth in ground_truth.items():
        predicted = predictions.get(
            s1_id,
            set(),
        )

        per_entity[s1_id] = f05_single_entity(
            predicted,
            truth,
        )

    if not per_entity:
        return 0.0, per_entity

    macro_f05 = float(
        np.mean(
            list(per_entity.values())
        )
    )

    return macro_f05, per_entity


def apply_threshold(
    probas: pd.DataFrame,
    threshold: float,
) -> Dict[str, Set[str]]:
    """Convert pair probabilities into entity-level predictions."""

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    required_columns = {
        "s1_id",
        "candidate_id",
        "proba",
    }

    missing = required_columns - set(
        probas.columns
    )

    if missing:
        raise ValueError(
            f"probas is missing required columns: "
            f"{sorted(missing)}"
        )

    positive = probas[
        probas["proba"] >= threshold
    ]

    predictions: Dict[str, Set[str]] = {}

    for s1_id, group in positive.groupby(
        "s1_id"
    ):
        predictions[s1_id] = set(
            group["candidate_id"]
        )

    return predictions


def optimize_threshold(
    probas: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
) -> Tuple[
    float,
    float,
    List[Tuple[float, float]],
]:
    """Find the threshold maximizing macro F0.5."""

    if not 0 <= threshold_min <= 1:
        raise ValueError(
            "threshold_min must be between 0 and 1"
        )

    if not 0 <= threshold_max <= 1:
        raise ValueError(
            "threshold_max must be between 0 and 1"
        )

    if threshold_min > threshold_max:
        raise ValueError(
            "threshold_min cannot exceed threshold_max"
        )

    if threshold_step <= 0:
        raise ValueError(
            "threshold_step must be greater than 0"
        )

    required_columns = {
        "s1_id",
        "candidate_id",
        "proba",
    }

    missing = required_columns - set(
        probas.columns
    )

    if missing:
        raise ValueError(
            f"probas is missing required columns: "
            f"{sorted(missing)}"
        )

    all_s1_ids = set(
        ground_truth.keys()
    )

    thresholds = np.arange(
        threshold_min,
        threshold_max
        + threshold_step / 2,
        threshold_step,
    )

    best_threshold = float(
        threshold_min
    )
    best_f05 = -1.0

    results: List[
        Tuple[float, float]
    ] = []

    logger.info(
        "Sweeping %s thresholds in [%.2f, %.2f] ...",
        len(thresholds),
        threshold_min,
        threshold_max,
    )

    for threshold in thresholds:
        threshold = float(threshold)

        predictions = apply_threshold(
            probas,
            threshold,
        )

        full_predictions = {
            s1_id: predictions.get(
                s1_id,
                set(),
            )
            for s1_id in all_s1_ids
        }

        macro_f05, _ = compute_macro_f05(
            full_predictions,
            ground_truth,
        )

        results.append(
            (
                threshold,
                macro_f05,
            )
        )

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_threshold = threshold

    logger.info(
        "Best threshold = %.3f -> macro F0.5 = %.5f",
        best_threshold,
        best_f05,
    )

    return (
        best_threshold,
        best_f05,
        results,
    )


def measure_blocking_recall(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, int, int, int]:
    """Measure true-match recall inside blocking candidates."""

    true_total = 0
    true_found = 0
    total_candidates = 0

    for s1_id, true_ids in ground_truth.items():
        if not true_ids:
            continue

        candidate_set = candidates.get(
            s1_id,
            set(),
        )

        true_total += len(true_ids)
        true_found += len(
            true_ids & candidate_set
        )
        total_candidates += len(
            candidate_set
        )

    if true_total > 0:
        recall = true_found / true_total
    else:
        recall = 1.0

    return (
        float(recall),
        true_found,
        true_total,
        total_candidates,
    )


def blocking_recall_by_source(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> dict:
    """Measure blocking recall separately for S2 and S3."""

    s2_total = 0
    s2_found = 0

    s3_total = 0
    s3_found = 0

    for s1_id, true_ids in ground_truth.items():
        candidate_set = candidates.get(
            s1_id,
            set(),
        )

        for true_id in true_ids:
            if true_id.startswith("S2-"):
                s2_total += 1

                if true_id in candidate_set:
                    s2_found += 1

            elif true_id.startswith("S3-"):
                s3_total += 1

                if true_id in candidate_set:
                    s3_found += 1

    return {
        "s2_recall": (
            s2_found / s2_total
            if s2_total > 0
            else 1.0
        ),
        "s2_found": s2_found,
        "s2_total": s2_total,
        "s3_recall": (
            s3_found / s3_total
            if s3_total > 0
            else 1.0
        ),
        "s3_found": s3_found,
        "s3_total": s3_total,
    }


def classification_report(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> dict:
    """Compute aggregate classification statistics."""

    tp_total = 0
    fp_total = 0
    fn_total = 0

    true_empty = 0
    correct_empty = 0
    wrong_empty = 0
    predicted_empty = 0

    for s1_id, truth in ground_truth.items():
        predicted = predictions.get(
            s1_id,
            set(),
        )

        if not truth:
            true_empty += 1

            if not predicted:
                correct_empty += 1
                predicted_empty += 1
            else:
                wrong_empty += 1
                fp_total += len(predicted)

            continue

        if not predicted:
            predicted_empty += 1
            fn_total += len(truth)
            continue

        true_positive = len(
            predicted & truth
        )
        false_positive = len(
            predicted - truth
        )
        false_negative = len(
            truth - predicted
        )

        tp_total += true_positive
        fp_total += false_positive
        fn_total += false_negative

    precision_denominator = (
        tp_total + fp_total
    )

    recall_denominator = (
        tp_total + fn_total
    )

    precision = (
        tp_total / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        tp_total / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f05_denominator = (
        BETA_SQ * precision + recall
    )

    f05 = (
        (1 + BETA_SQ)
        * precision
        * recall
        / f05_denominator
        if f05_denominator > 0
        else 0.0
    )

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f05": f05,
        "true_positives": tp_total,
        "false_positives": fp_total,
        "false_negatives": fn_total,
        "true_singletons": true_empty,
        "correct_singletons": correct_empty,
        "false_merge_on_singletons": wrong_empty,
        "predicted_empty": predicted_empty,
    }


def error_analysis(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    top_k: int = 20,
) -> Tuple[
    List[Tuple[str, float, Set[str], Set[str]]],
    List[Tuple[str, float, Set[str], Set[str]]],
]:
    """Return entities with the worst FP and FN behavior."""

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1"
        )

    entities = []

    for s1_id, truth in ground_truth.items():
        predicted = predictions.get(
            s1_id,
            set(),
        )

        score = f05_single_entity(
            predicted,
            truth,
        )

        fp_count = len(
            predicted - truth
        )

        fn_count = (
            len(truth - predicted)
            if truth
            else 0
        )

        entities.append(
            (
                s1_id,
                score,
                predicted,
                truth,
                fp_count,
                fn_count,
            )
        )

    worst_fp = sorted(
        entities,
        key=lambda item: (
            -item[4],
            item[1],
        ),
    )[:top_k]

    worst_fp_output = [
        (
            item[0],
            item[1],
            item[2],
            item[3],
        )
        for item in worst_fp
        if item[4] > 0
    ]

    worst_fn = sorted(
        entities,
        key=lambda item: (
            -item[5],
            item[1],
        ),
    )[:top_k]

    worst_fn_output = [
        (
            item[0],
            item[1],
            item[2],
            item[3],
        )
        for item in worst_fn
        if item[5] > 0
    ]

    return (
        worst_fp_output,
        worst_fn_output,
    )