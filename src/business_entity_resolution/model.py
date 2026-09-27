from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_hard_negatives(
    positive_pairs: pd.DataFrame,
    candidates: dict[str, set[str]],
    neg_pos_ratio: int = 5,
    hard_fraction: float = 0.7,
    random_seed: int = 42,
    s1_countries: dict[str, str] | None = None,
    candidate_countries: dict[str, str] | None = None,
) -> pd.DataFrame:
    if neg_pos_ratio < 1:
        raise ValueError(
            "neg_pos_ratio must be at least 1"
        )

    if not 0 <= hard_fraction <= 1:
        raise ValueError(
            "hard_fraction must be between 0 and 1"
        )

    required_columns = {
        "s1_id",
        "candidate_id",
    }

    missing = (
        required_columns
        - set(positive_pairs.columns)
    )

    if missing:
        raise ValueError(
            "positive_pairs is missing required columns: "
            f"{sorted(missing)}"
        )

    rng = np.random.RandomState(random_seed)

    positive_pairs = (
        positive_pairs[
            ["s1_id", "candidate_id"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    n_pos = len(positive_pairs)

    if n_pos == 0:
        return pd.DataFrame(
            columns=[
                "s1_id",
                "candidate_id",
                "label",
            ]
        )

    n_neg_target = n_pos * neg_pos_ratio

    n_hard_target = int(
        n_neg_target * hard_fraction
    )

    n_random_target = (
        n_neg_target - n_hard_target
    )

    logger.info(
        "Generating negatives: %s total "
        "(%s hard + %s random) for %s positives",
        f"{n_neg_target:,}",
        f"{n_hard_target:,}",
        f"{n_random_target:,}",
        f"{n_pos:,}",
    )

    positive_set = set(
        zip(
            positive_pairs["s1_id"],
            positive_pairs["candidate_id"],
            strict=True,
        )
    )

    hard_rows: list[tuple[str, str]] = []

    s1_ids = list(candidates.keys())
    rng.shuffle(s1_ids)

    for s1_id in s1_ids:
        if len(hard_rows) >= n_hard_target:
            break

        candidate_set = candidates.get(
            s1_id,
            set(),
        )

        false_candidates = [
            candidate_id
            for candidate_id in candidate_set
            if (
                s1_id,
                candidate_id,
            )
            not in positive_set
        ]

        if not false_candidates:
            continue

        max_per_entity = max(
            1,
            neg_pos_ratio * 2,
        )

        sample_size = min(
            len(false_candidates),
            max_per_entity,
            n_hard_target - len(hard_rows),
        )

        sampled = rng.choice(
            false_candidates,
            size=sample_size,
            replace=False,
        )

        for candidate_id in sampled:
            hard_rows.append(
                (
                    str(s1_id),
                    str(candidate_id),
                )
            )

            if len(hard_rows) >= n_hard_target:
                break

    logger.info(
        "Hard negatives generated: %s",
        f"{len(hard_rows):,}",
    )

    random_rows: list[tuple[str, str]] = []

    if n_random_target > 0 and s1_ids:
        candidate_ids = list(
            {
                candidate_id
                for candidate_set in candidates.values()
                for candidate_id in candidate_set
            }
        )

        if candidate_ids:
            rng.shuffle(candidate_ids)

            if (
                s1_countries
                and candidate_countries
            ):
                country_to_candidates: dict[
                    str,
                    list[str],
                ] = {}

                for candidate_id in candidate_ids:
                    country = candidate_countries.get(
                        candidate_id
                    )

                    if country is None:
                        continue

                    country_to_candidates.setdefault(
                        country,
                        [],
                    ).append(candidate_id)

                eligible_s1 = [
                    s1_id
                    for s1_id in s1_ids
                    if s1_countries.get(s1_id)
                    in country_to_candidates
                ]

                attempts = 0
                max_attempts = max(
                    n_random_target * 10,
                    100,
                )

                while (
                    len(random_rows)
                    < n_random_target
                    and attempts < max_attempts
                    and eligible_s1
                ):
                    s1_id = rng.choice(
                        eligible_s1
                    )

                    country = s1_countries.get(
                        s1_id
                    )

                    country_candidates = (
                        country_to_candidates.get(
                            country,
                            [],
                        )
                    )

                    if not country_candidates:
                        attempts += 1
                        continue

                    candidate_id = rng.choice(
                        country_candidates
                    )

                    pair = (
                        str(s1_id),
                        str(candidate_id),
                    )

                    if pair not in positive_set:
                        random_rows.append(pair)

                    attempts += 1

            else:
                attempts = 0
                max_attempts = max(
                    n_random_target * 10,
                    100,
                )

                while (
                    len(random_rows)
                    < n_random_target
                    and attempts < max_attempts
                ):
                    s1_id = rng.choice(
                        s1_ids
                    )

                    candidate_id = rng.choice(
                        candidate_ids
                    )

                    pair = (
                        str(s1_id),
                        str(candidate_id),
                    )

                    if pair not in positive_set:
                        random_rows.append(pair)

                    attempts += 1

    logger.info(
        "Random negatives generated: %s",
        f"{len(random_rows):,}",
    )

    negative_rows = (
        hard_rows + random_rows
    )

    if not negative_rows:
        return pd.DataFrame(
            columns=[
                "s1_id",
                "candidate_id",
                "label",
            ]
        )

    negatives = pd.DataFrame(
        negative_rows,
        columns=[
            "s1_id",
            "candidate_id",
        ],
    ).drop_duplicates(
        subset=[
            "s1_id",
            "candidate_id",
        ]
    )

    negatives["label"] = 0

    logger.info(
        "Total negatives after dedup: %s",
        f"{len(negatives):,}",
    )

    return negatives.reset_index(
        drop=True
    )


class EntityMatcherModel:
    SUPPORTED_MODELS: ClassVar[set[str]] = {
        "lightgbm",
        "logreg",
    }

    def __init__(
        self,
        model_type: str = "lightgbm",
        params: dict | None = None,
    ) -> None:
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Expected one of "
                f"{sorted(self.SUPPORTED_MODELS)}."
            )

        self.model_type = model_type
        self.params = (
            params.copy()
            if params
            else {}
        )

        self.model = None
        self._booster = None
        self._scaler = None
        self.feature_names: list[str] | None = None
        self.n_features_: int | None = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        X_train = np.asarray(
            X_train,
            dtype=np.float32,
        )

        y_train = np.asarray(
            y_train,
            dtype=np.int8,
        )

        if X_train.ndim != 2:
            raise ValueError(
                "X_train must be a 2D array"
            )

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train must have "
                "the same number of rows"
            )

        if len(X_train) == 0:
            raise ValueError(
                "X_train is empty"
            )

        if X_val is not None:
            X_val = np.asarray(
                X_val,
                dtype=np.float32,
            )

        if y_val is not None:
            y_val = np.asarray(
                y_val,
                dtype=np.int8,
            )

        if (
            X_val is not None
            and y_val is not None
        ):
            if X_val.ndim != 2:
                raise ValueError(
                    "X_val must be a 2D array"
                )

            if (
                X_val.shape[1]
                != X_train.shape[1]
            ):
                raise ValueError(
                    "X_train and X_val must have "
                    "the same number of features"
                )

            if len(X_val) != len(y_val):
                raise ValueError(
                    "X_val and y_val must have "
                    "the same number of rows"
                )

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else None
        )

        if (
            self.feature_names is not None
            and len(self.feature_names)
            != X_train.shape[1]
        ):
            raise ValueError(
                "Number of feature names must match "
                "the number of training features"
            )

        self.n_features_ = X_train.shape[1]

        logger.info(
            "Training %s on %s samples, %s features",
            self.model_type,
            f"{X_train.shape[0]:,}",
            X_train.shape[1],
        )

        if self.model_type == "lightgbm":
            self._train_lgbm(
                X_train,
                y_train,
                X_val,
                y_val,
            )
        else:
            self._train_logreg(
                X_train,
                y_train,
            )

    def _train_lgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        import lightgbm as lgb

        params = self.params.copy()

        n_estimators = int(
            params.pop(
                "n_estimators",
                800,
            )
        )

        n_pos = int(
            np.sum(y_train == 1)
        )

        n_neg = int(
            np.sum(y_train == 0)
        )

        if n_pos == 0:
            raise ValueError(
                "Training data contains no positive samples"
            )

        if n_neg == 0:
            raise ValueError(
                "Training data contains no negative samples"
            )

        params.setdefault(
            "objective",
            "binary",
        )

        params.setdefault(
            "random_state",
            42,
        )

        params.setdefault(
            "verbosity",
            -1,
        )

        params.setdefault(
            "n_jobs",
            -1,
        )

        params.setdefault(
            "scale_pos_weight",
            n_neg / n_pos,
        )

        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            **params,
        )

        eval_set = None

        if (
            X_val is not None
            and y_val is not None
        ):
            eval_set = [
                (
                    X_val,
                    y_val,
                )
            ]

        callbacks = []

        if eval_set is not None:
            callbacks.append(
                lgb.early_stopping(
                    50,
                    verbose=True,
                )
            )

            callbacks.append(
                lgb.log_evaluation(100)
            )

        fit_kwargs = {}

        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set

        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        self.model.fit(
            X_train,
            y_train,
            **fit_kwargs,
        )

        self._booster = self.model.booster_

        self.n_features_ = (
            self._booster.num_feature()
        )

        logger.info(
            "LightGBM training complete. "
            "Features=%s, Trees=%s",
            self.n_features_,
            self._booster.current_iteration(),
        )

    def _train_logreg(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        from sklearn.linear_model import (
            LogisticRegression,
        )
        from sklearn.preprocessing import (
            StandardScaler,
        )

        self._scaler = StandardScaler()

        X_scaled = (
            self._scaler.fit_transform(
                X_train
            )
        )

        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )

        self.model.fit(
            X_scaled,
            y_train,
        )

        self.n_features_ = X_train.shape[1]

        logger.info(
            "Logistic regression training complete."
        )

    def _validate_features(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array"
            )

        if (
            self.n_features_ is not None
            and X.shape[1] != self.n_features_
        ):
            raise ValueError(
                "Feature count mismatch: "
                f"model expects {self.n_features_}, "
                f"received {X.shape[1]}"
            )

        return X

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        X = self._validate_features(X)

        if self.model_type == "lightgbm":
            if self._booster is None:
                if self.model is not None:
                    self._booster = (
                        self.model.booster_
                    )
                else:
                    raise RuntimeError(
                        "LightGBM model is not trained "
                        "or loaded"
                    )

            probabilities = self._booster.predict(
                X
            )

            probabilities = np.asarray(
                probabilities,
                dtype=np.float64,
            )

            if probabilities.ndim != 1:
                probabilities = probabilities[:, 1]

            return np.clip(
                probabilities,
                0.0,
                1.0,
            )

        if self.model is None:
            raise RuntimeError(
                "Logistic regression model is not "
                "trained or loaded"
            )

        if self._scaler is None:
            raise RuntimeError(
                "Logistic regression scaler "
                "is not available"
            )

        X_scaled = self._scaler.transform(X)

        probabilities = self.model.predict_proba(
            X_scaled
        )[:, 1]

        return np.clip(
            probabilities,
            0.0,
            1.0,
        )

    def feature_importance(
        self,
    ) -> pd.DataFrame | None:
        if self.model_type != "lightgbm":
            return None

        if self._booster is None:
            if self.model is None:
                return None

            self._booster = self.model.booster_

        importances = (
            self._booster.feature_importance(
                importance_type="gain"
            )
        )

        if self.feature_names is None:
            names = [
                f"f_{index}"
                for index in range(
                    len(importances)
                )
            ]
        else:
            names = self.feature_names

        if len(names) != len(importances):
            raise ValueError(
                "Number of feature names does not "
                "match number of model features"
            )

        return (
            pd.DataFrame(
                {
                    "feature": names,
                    "importance": importances,
                }
            )
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    def save(
        self,
        path: Path,
    ) -> None:
        path = Path(path)

        if self.model_type == "lightgbm":
            if self._booster is None:
                if self.model is not None:
                    self._booster = (
                        self.model.booster_
                    )
                else:
                    raise RuntimeError(
                        "Cannot save an untrained "
                        "LightGBM model"
                    )
        elif self.model is None:
            raise RuntimeError(
                "Cannot save an untrained model"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path = path.with_suffix(
            path.suffix + ".meta.json"
        )

        metadata = {
            "model_type": self.model_type,
            "params": self.params,
            "feature_names": self.feature_names,
            "n_features": self.n_features_,
        }

        if self.model_type == "lightgbm":
            self._booster.save_model(
                str(path)
            )
        else:
            payload = {
                "model": self.model,
                "scaler": self._scaler,
            }

            joblib.dump(
                payload,
                path,
            )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )

        logger.info(
            "Model saved to %s",
            path,
        )

    def load(
        self,
        path: Path,
    ) -> None:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        metadata_path = path.with_suffix(
            path.suffix + ".meta.json"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                "Model metadata file not found: "
                f"{metadata_path}"
            )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        saved_model_type = metadata.get(
            "model_type",
            self.model_type,
        )

        if (
            saved_model_type
            not in self.SUPPORTED_MODELS
        ):
            raise ValueError(
                "Unsupported saved model type: "
                f"{saved_model_type}"
            )

        self.model_type = saved_model_type

        self.params = metadata.get(
            "params",
            self.params,
        )

        self.feature_names = metadata.get(
            "feature_names"
        )

        saved_n_features = metadata.get(
            "n_features"
        )

        self.n_features_ = (
            int(saved_n_features)
            if saved_n_features is not None
            else None
        )

        self.model = None
        self._booster = None
        self._scaler = None

        if self.model_type == "lightgbm":
            import lightgbm as lgb

            self._booster = lgb.Booster(
                model_file=str(path)
            )

            booster_features = (
                self._booster.num_feature()
            )

            if (
                self.n_features_ is not None
                and booster_features
                != self.n_features_
            ):
                raise ValueError(
                    "Saved metadata feature count "
                    f"({self.n_features_}) does not "
                    "match LightGBM model feature "
                    f"count ({booster_features})"
                )

            self.n_features_ = booster_features

        else:
            payload = joblib.load(path)

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(
                    "Invalid logistic-regression "
                    "model artifact"
                )

            self.model = payload.get(
                "model"
            )

            self._scaler = payload.get(
                "scaler"
            )

            if self.model is None:
                raise ValueError(
                    "Logistic-regression model "
                    "artifact is missing the model"
                )

            if self._scaler is None:
                raise ValueError(
                    "Logistic-regression model "
                    "artifact is missing the scaler"
                )

            if self.n_features_ is None:
                self.n_features_ = int(
                    self.model.n_features_in_
                )

        logger.info(
            "Model loaded from %s",
            path,
        )


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_pairs: pd.DataFrame,
    val_gt: dict[str, set[str]],
    lgbm_params: dict | None = None,
    feature_names: list[str] | None = None,
) -> tuple[EntityMatcherModel, dict]:
    from .evaluation import optimize_threshold

    results = {}

    for model_type in (
        "lightgbm",
        "logreg",
    ):
        logger.info(
            "%s",
            "=" * 50,
        )

        logger.info(
            "Training %s ...",
            model_type,
        )

        params = (
            lgbm_params.copy()
            if (
                model_type == "lightgbm"
                and lgbm_params
            )
            else {}
        )

        model = EntityMatcherModel(
            model_type=model_type,
            params=params,
        )

        model.train(
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names,
        )

        val_proba = model.predict_proba(
            X_val
        )

        if len(val_pairs) != len(val_proba):
            raise ValueError(
                "Validation pair count does not "
                "match probability count"
            )

        proba_df = val_pairs[
            ["s1_id", "candidate_id"]
        ].copy()

        proba_df["proba"] = val_proba

        (
            best_threshold,
            best_f05,
            sweep,
        ) = optimize_threshold(
            proba_df,
            val_gt,
        )

        results[model_type] = {
            "model": model,
            "best_threshold": best_threshold,
            "best_f05": best_f05,
            "sweep": sweep,
        }

        logger.info(
            "%s: threshold=%.4f, "
            "macro_F0.5=%.5f",
            model_type,
            best_threshold,
            best_f05,
        )

    if not results:
        raise RuntimeError(
            "No models were trained"
        )

    best_type = max(
        results,
        key=lambda model_name: results[
            model_name
        ]["best_f05"],
    )

    best_model = results[
        best_type
    ]["model"]

    logger.info(
        "Selected model type: %s",
        best_type,
    )

    return best_model, results


def save_threshold(
    threshold: float,
    path: Path,
) -> None:
    threshold = float(threshold)

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "threshold": threshold,
            },
            file,
            indent=2,
        )

    logger.info(
        "Threshold saved: %.4f -> %s",
        threshold,
        path,
    )


def load_threshold(
    path: Path,
) -> float:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if "threshold" not in data:
        raise ValueError(
            f"Threshold key missing from {path}"
        )

    threshold = float(
        data["threshold"]
    )

    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Invalid threshold: {threshold}"
        )

    return threshold