from __future__ import annotations

import logging

import pandas as pd

from .blocking import (
    BlockingEngine,
    candidates_to_dataframe,
    write_candidate_pairs_tsv,
)
from .config import PipelineConfig
from .data_loader import (
    entity_level_split,
    get_positive_pairs,
    load_all_test,
    load_all_train,
)
from .evaluation import (
    apply_threshold,
    measure_blocking_recall,
)
from .features import (
    FEATURE_NAMES,
    batch_compute_features,
)
from .model import (
    EntityMatcherModel,
    compare_models,
    generate_hard_negatives,
    load_threshold,
    save_threshold,
)
from .preprocessing import preprocess_dataframe_chunked

logger = logging.getLogger(__name__)


def _get_source_country_map(
    dataframe: pd.DataFrame,
) -> dict[str, str]:
    """Build entity-ID to country mapping."""

    if dataframe.empty:
        return {}

    if "country" not in dataframe.columns:
        return {}

    if "entity_id" in dataframe.columns:
        entity_ids = dataframe["entity_id"]
    else:
        entity_ids = dataframe.index

    result: dict[str, str] = {}

    for entity_id, country in zip(
        entity_ids,
        dataframe["country"],
        strict=True,
    ):
        if pd.notna(country):
            result[str(entity_id)] = str(country)

    return result


def _label_candidate_pairs(
    pairs: pd.DataFrame,
    ground_truth: dict[str, set[str]],
) -> pd.Series:
    """Assign binary labels to candidate pairs."""

    if pairs.empty:
        return pd.Series(
            dtype="int8",
            index=pairs.index,
        )

    required_columns = {
        "s1_id",
        "candidate_id",
    }

    missing = required_columns - set(pairs.columns)

    if missing:
        raise ValueError(
            "Candidate pairs missing required columns: "
            f"{sorted(missing)}"
        )

    positive_pairs = get_positive_pairs(
        ground_truth
    )

    positive_set = set(
        zip(
            positive_pairs["s1_id"],
            positive_pairs["candidate_id"],
            strict=True,
        )
    )

    pair_tuples = zip(
        pairs["s1_id"],
        pairs["candidate_id"],
        strict=True,
    )

    return pd.Series(
        (
            int(pair in positive_set)
            for pair in pair_tuples
        ),
        index=pairs.index,
        dtype="int8",
    )


def _prepare_dataframes(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    config: PipelineConfig,
    s1_name: str,
    s2_name: str,
    s3_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Apply preprocessing to all source dataframes."""

    s1 = preprocess_dataframe_chunked(
        s1,
        s1_name,
        chunk_size=config.chunk_size,
    )

    s2 = preprocess_dataframe_chunked(
        s2,
        s2_name,
        chunk_size=config.chunk_size,
    )

    s3 = preprocess_dataframe_chunked(
        s3,
        s3_name,
        chunk_size=config.chunk_size,
    )

    return s1, s2, s3


def _get_source_row_limit(
    sample_size: int | None,
) -> int | None:
    """Calculate source-2/source-3 row limit for sample runs."""

    if sample_size is None:
        return None

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be greater than 0"
        )

    return min(
        sample_size * 20,
        100_000,
    )


def _validate_feature_matrix(
    X: object,
    expected_features: int,
    name: str,
) -> None:
    """Validate feature matrix shape."""

    if not hasattr(X, "ndim"):
        raise ValueError(
            f"{name} is not a valid feature matrix"
        )

    if X.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D matrix"
        )

    if X.shape[1] != expected_features:
        raise ValueError(
            f"{name} has {X.shape[1]} features, "
            f"but {expected_features} are expected"
        )


def _combine_candidate_sources(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> pd.DataFrame:
    """Combine candidate sources for pair-feature lookup."""

    if s2.empty and s3.empty:
        return pd.DataFrame()

    if s2.empty:
        return s3.copy()

    if s3.empty:
        return s2.copy()

    return pd.concat(
        [s2, s3],
        axis=0,
        copy=False,
    )


def run_training_pipeline(
    config: PipelineConfig,
    sample_size: int | None = None,
) -> EntityMatcherModel:
    """Run the complete training pipeline."""

    logger.info(
        "Starting training pipeline..."
    )

    s2_s3_limit = _get_source_row_limit(
        sample_size
    )

    s1, s2, s3, ground_truth = load_all_train(
        config,
        nrows=sample_size,
        s2_nrows=s2_s3_limit,
        s3_nrows=s2_s3_limit,
    )

    logger.info(
        "Loaded training data: "
        "S1=%s, S2=%s, S3=%s",
        f"{len(s1):,}",
        f"{len(s2):,}",
        f"{len(s3):,}",
    )

    s1, s2, s3 = _prepare_dataframes(
        s1,
        s2,
        s3,
        config,
        "S1_Train",
        "S2_Train",
        "S3_Train",
    )

    train_gt, val_gt = entity_level_split(
        ground_truth,
        val_fraction=config.val_fraction,
        random_seed=config.random_seed,
    )

    train_s1_ids = set(train_gt.keys())
    val_s1_ids = set(val_gt.keys())

    if "entity_id" in s1.columns:
        s1_train = s1[
            s1["entity_id"].astype(str).isin(
                train_s1_ids
            )
        ].copy()

        s1_val = s1[
            s1["entity_id"].astype(str).isin(
                val_s1_ids
            )
        ].copy()
    else:
        s1_train = s1.loc[
            s1.index.astype(str).isin(
                train_s1_ids
            )
        ].copy()

        s1_val = s1.loc[
            s1.index.astype(str).isin(
                val_s1_ids
            )
        ].copy()

    logger.info(
        "Train S1: %s, Val S1: %s",
        f"{len(s1_train):,}",
        f"{len(s1_val):,}",
    )

    if s1_train.empty:
        raise ValueError(
            "Training S1 dataframe is empty"
        )

    if s1_val.empty:
        raise ValueError(
            "Validation S1 dataframe is empty"
        )

    blocker = BlockingEngine(
        config.blocking
    )

    logger.info(
        "Generating candidates for training set..."
    )

    train_candidates = blocker.generate_candidates(
        s1_train,
        s2,
        s3,
    )

    logger.info(
        "Generating candidates for validation set..."
    )

    val_candidates = blocker.generate_candidates(
        s1_val,
        s2,
        s3,
    )

    (
        recall,
        found,
        total,
        candidate_count,
    ) = measure_blocking_recall(
        val_candidates,
        val_gt,
    )

    logger.info(
        "Validation blocking recall: %.4f "
        "(%s/%s true pairs found, %s candidates)",
        recall,
        found,
        total,
        f"{candidate_count:,}",
    )

    if total > 0 and recall <= 0:
        raise ValueError(
            "Blocking recall is zero. "
            "Training cannot proceed safely."
        )

    train_positive = get_positive_pairs(
        train_gt
    )

    s1_country_map = (
        _get_source_country_map(
            s1_train
        )
    )

    s2_country_map = (
        _get_source_country_map(s2)
    )

    s3_country_map = (
        _get_source_country_map(s3)
    )

    candidate_country_map = {
        **s2_country_map,
        **s3_country_map,
    }

    train_negative = generate_hard_negatives(
        train_positive,
        train_candidates,
        neg_pos_ratio=config.model.neg_pos_ratio,
        hard_fraction=config.model.hard_neg_fraction,
        random_seed=config.random_seed,
        s1_countries=s1_country_map,
        candidate_countries=candidate_country_map,
    )

    train_pairs = pd.concat(
        [
            train_positive,
            train_negative,
        ],
        ignore_index=True,
    )

    val_pairs = candidates_to_dataframe(
        val_candidates
    )

    val_pairs["label"] = _label_candidate_pairs(
        val_pairs,
        val_gt,
    )

    logger.info(
        "Train pairs: %s "
        "(Pos: %s, Neg: %s)",
        f"{len(train_pairs):,}",
        f"{len(train_positive):,}",
        f"{len(train_negative):,}",
    )

    logger.info(
        "Validation pairs: %s",
        f"{len(val_pairs):,}",
    )

    if train_pairs.empty:
        raise ValueError(
            "No training pairs were generated"
        )

    if val_pairs.empty:
        raise ValueError(
            "No validation candidates were generated"
        )

    train_positive_count = int(
        train_pairs["label"].sum()
    )

    if train_positive_count == 0:
        raise ValueError(
            "Training pairs contain no positive examples"
        )

    val_positive_count = int(
        val_pairs["label"].sum()
    )

    if val_positive_count == 0:
        logger.warning(
            "Validation candidates contain no "
            "positive examples."
        )

    s23_combined = _combine_candidate_sources(
        s2,
        s3,
    )

    if s23_combined.empty:
        raise ValueError(
            "S2 and S3 are both empty"
        )

    logger.info(
        "Computing training features..."
    )

    X_train = batch_compute_features(
        train_pairs,
        s1_train,
        s23_combined,
        config.chunk_size,
    )

    _validate_feature_matrix(
        X_train,
        len(FEATURE_NAMES),
        "X_train",
    )

    y_train = train_pairs[
        "label"
    ].to_numpy()

    logger.info(
        "Computing validation features..."
    )

    X_val = batch_compute_features(
        val_pairs,
        s1_val,
        s23_combined,
        config.chunk_size,
    )

    _validate_feature_matrix(
        X_val,
        len(FEATURE_NAMES),
        "X_val",
    )

    y_val = val_pairs[
        "label"
    ].to_numpy()

    logger.info(
        "Training model comparison..."
    )

    best_model, model_results = compare_models(
        X_train,
        y_train,
        X_val,
        y_val,
        val_pairs=val_pairs,
        val_gt=val_gt,
        lgbm_params=config.model.lgbm_params,
        feature_names=FEATURE_NAMES,
    )

    if best_model.model_type not in model_results:
        raise RuntimeError(
            "Selected model type is missing "
            "from model comparison results"
        )

    best_threshold = model_results[
        best_model.model_type
    ]["best_threshold"]

    if not 0 <= best_threshold <= 1:
        raise ValueError(
            f"Invalid optimized threshold: "
            f"{best_threshold}"
        )

    best_model.save(
        config.paths.model_path
    )

    save_threshold(
        best_threshold,
        config.paths.threshold_path,
    )

    logger.info(
        "Selected model: %s",
        best_model.model_type,
    )

    logger.info(
        "Selected threshold: %.4f",
        best_threshold,
    )

    logger.info(
        "Training pipeline complete."
    )

    return best_model


def run_inference_pipeline(
    config: PipelineConfig,
    sample_size: int | None = None,
) -> None:
    """Run the complete inference pipeline on the test set."""

    logger.info(
        "Starting inference pipeline..."
    )

    s2_s3_limit = _get_source_row_limit(
        sample_size
    )

    s1, s2, s3 = load_all_test(
        config,
        s1_nrows=sample_size,
        s2_nrows=s2_s3_limit,
        s3_nrows=s2_s3_limit,
    )

    logger.info(
        "Loaded test data: "
        "S1=%s, S2=%s, S3=%s",
        f"{len(s1):,}",
        f"{len(s2):,}",
        f"{len(s3):,}",
    )

    s1, s2, s3 = _prepare_dataframes(
        s1,
        s2,
        s3,
        config,
        "S1_Test",
        "S2_Test",
        "S3_Test",
    )

    if s1.empty:
        raise ValueError(
            "Test S1 dataframe is empty"
        )

    blocker = BlockingEngine(
        config.blocking
    )

    logger.info(
        "Generating candidates for test set..."
    )

    candidates = blocker.generate_candidates(
        s1,
        s2,
        s3,
    )

    candidate_output_path = (
        config.paths.candidate_output
    )

    candidate_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_s1_ids = (
        s1["entity_id"].astype(str).tolist()
        if "entity_id" in s1.columns
        else s1.index.astype(str).tolist()
    )

    write_candidate_pairs_tsv(
        candidates,
        candidate_output_path,
        all_s1_ids=all_s1_ids,
    )

    logger.info(
        "Candidate pairs written to: %s",
        candidate_output_path,
    )

    test_pairs = candidates_to_dataframe(
        candidates
    )

    model = EntityMatcherModel()

    model.load(
        config.paths.model_path
    )

    threshold = load_threshold(
        config.paths.threshold_path
    )

    logger.info(
        "Loaded model: %s",
        model.model_type,
    )

    logger.info(
        "Using threshold: %.4f",
        threshold,
    )

    if test_pairs.empty:
        logger.warning(
            "No candidates generated. "
            "Writing empty predictions."
        )

        predictions = {}

    else:
        s23_combined = _combine_candidate_sources(
            s2,
            s3,
        )

        if s23_combined.empty:
            raise ValueError(
                "S2 and S3 are empty while "
                "candidate pairs exist"
            )

        logger.info(
            "Computing test features..."
        )

        X_test = batch_compute_features(
            test_pairs,
            s1,
            s23_combined,
            config.chunk_size,
        )

        _validate_feature_matrix(
            X_test,
            len(FEATURE_NAMES),
            "X_test",
        )

        logger.info(
            "Running model inference..."
        )

        test_pairs["proba"] = (
            model.predict_proba(
                X_test
            )
        )

        predictions = apply_threshold(
            test_pairs,
            threshold,
        )

    matching_output_path = (
        config.paths.matching_output
    )

    matching_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        matching_output_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "source1_entity_id\t"
            "matched_entity_ids\n"
        )

        if "entity_id" in s1.columns:
            output_s1_ids = (
                s1["entity_id"]
                .astype(str)
                .tolist()
            )
        else:
            output_s1_ids = (
                s1.index.astype(str).tolist()
            )

        for s1_id in output_s1_ids:
            matched_ids = predictions.get(
                s1_id,
                set(),
            )

            if matched_ids:
                matched_string = ",".join(
                    sorted(
                        str(candidate_id)
                        for candidate_id in matched_ids
                    )
                )
            else:
                matched_string = ""

            file.write(
                f"{s1_id}\t"
                f"{matched_string}\n"
            )

    logger.info(
        "Matching results written to: %s",
        matching_output_path,
    )

    logger.info(
        "Inference pipeline complete."
    )