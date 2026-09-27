"""
Configuration for the Business Entity Resolution pipeline.

All paths, hyperparameters, and runtime settings are centralized here.

Important:
- No country list is hard-coded.
- The pipeline is designed for open-set countries.
- Paths support both local execution and Google Colab/Google Drive.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
VAL_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def detect_project_root() -> Path:
    """
    Detect the project root across local and Google Colab environments.

    Resolution order:
    1. Known Google Drive project path used in Colab.
    2. Repository root inferred from this config.py location.
    3. Current working directory as a final fallback.
    """
    colab_path = Path(
        "/content/drive/MyDrive/Amazon ML Challenge 2026"
    )
    if colab_path.is_dir():
        return colab_path

    here = Path(__file__).resolve()
    project_root = here.parents[2]

    if project_root.is_dir():
        return project_root

    # Final fallback
    return Path.cwd().resolve()


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    """Data, output, model, cache, and utility paths."""

    project_root: Path = field(default_factory=detect_project_root)

    @staticmethod
    def _ensure_directory(path: Path) -> Path:
        """Create a directory if it does not already exist."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_dir(self) -> Path:
        """
        Return the dataset directory.

        The project may use either:
            project_root/data/
        or:
            project_root/dataset/

        ``data`` is preferred when both exist.
        """
        data_path = self.project_root / "data"
        dataset_path = self.project_root / "dataset"

        if data_path.is_dir():
            return data_path

        if dataset_path.is_dir():
            return dataset_path

        # Default location for a fresh project.
        return data_path

    @property
    def train_dir(self) -> Path:
        """Training dataset directory."""
        return self.data_dir / "train"

    @property
    def test_dir(self) -> Path:
        """Test dataset directory."""
        return self.data_dir / "test"

    # -----------------------------------------------------------------------
    # Runtime/output directories
    # -----------------------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        """Directory containing official submission outputs."""
        return self._ensure_directory(self.project_root / "output")

    @property
    def models_dir(self) -> Path:
        """Directory containing trained models and model metadata."""
        return self._ensure_directory(self.project_root / "models")

    @property
    def results_dir(self) -> Path:
        """Directory containing experiment results."""
        return self._ensure_directory(self.project_root / "results")

    @property
    def logs_dir(self) -> Path:
        """Directory containing pipeline logs."""
        return self._ensure_directory(self.project_root / "logs")

    @property
    def cache_dir(self) -> Path:
        """Directory containing reusable intermediate/cache data."""
        return self._ensure_directory(self.project_root / "cache")

    # -----------------------------------------------------------------------
    # Training files
    # -----------------------------------------------------------------------

    @property
    def train_s1(self) -> Path:
        """Training source-1 file."""
        return self.train_dir / "train_source1.tsv"

    @property
    def train_s2(self) -> Path:
        """Training source-2 file."""
        return self.train_dir / "train_source2.tsv"

    @property
    def train_s3(self) -> Path:
        """Training source-3 file."""
        return self.train_dir / "train_source3.tsv"

    @property
    def train_gt(self) -> Path:
        """Training ground-truth file."""
        return self.train_dir / "train_ground_truth.tsv"

    # -----------------------------------------------------------------------
    # Test files
    # -----------------------------------------------------------------------

    @property
    def test_s1(self) -> Path:
        """Test source-1 file."""
        return self.test_dir / "test_source1.tsv"

    @property
    def test_s2(self) -> Path:
        """Test source-2 file."""
        return self.test_dir / "test_source2.tsv"

    @property
    def test_s3(self) -> Path:
        """Test source-3 file."""
        return self.test_dir / "test_source3.tsv"

    # -----------------------------------------------------------------------
    # Official output files
    # -----------------------------------------------------------------------

    @property
    def matching_output(self) -> Path:
        """Official matching-results submission file."""
        return self.output_dir / "matching_results.tsv"

    @property
    def candidate_output(self) -> Path:
        """Official candidate-pairs submission file."""
        return self.output_dir / "candidate_pairs.tsv"

    # -----------------------------------------------------------------------
    # Model files
    # -----------------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        """Primary LightGBM model file."""
        return self.models_dir / "lgbm_model.txt"

    @property
    def threshold_path(self) -> Path:
        """Saved decision-threshold metadata."""
        return self.models_dir / "threshold.json"

    # -----------------------------------------------------------------------
    # Utility directory
    # -----------------------------------------------------------------------

    @property
    def utils_dir(self) -> Path:
        """
        Locate the challenge utility directory.

        Possible locations:
        1. project_root/utils
        2. project_root.parent/student_resource/utils
        3. project_root.parent/utils

        If none exists, project_root/utils is created as a fallback.
        """
        candidates = (
            self.project_root / "utils",
            self.project_root.parent / "student_resource" / "utils",
            self.project_root.parent / "utils",
        )

        for candidate in candidates:
            if candidate.is_dir():
                return candidate

        return self._ensure_directory(self.project_root / "utils")


# ---------------------------------------------------------------------------
# Blocking configuration
# ---------------------------------------------------------------------------

@dataclass
class BlockingConfig:
    """Candidate-generation and blocking settings."""

    # Prefix-based name blocking.
    name_prefix_lengths: list[int] = field(
        default_factory=lambda: [3, 4, 5]
    )

    # Token-based blocking.
    use_token_blocking: bool = True
    use_sorted_token_blocking: bool = True

    # Address/numeric blocking.
    use_numeric_blocking: bool = True

    # Character n-gram blocking.
    use_ngram_blocking: bool = True
    ngram_size: int = 3
    ngram_signature_count: int = 3

    # Rare-token blocking.
    min_rare_token_length: int = 4
    max_token_frequency: int = 5000

    # Safety limit. This is NOT a guarantee that true matches are retained.
    # Candidate recall must be measured empirically.
    max_candidates_per_entity: int = 500

    # Target used for diagnostics/experiments.
    blocking_recall_target: float = 0.99


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Feature-engineering settings."""

    name_ngram_size: int = 3
    addr_ngram_size: int = 3


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model training, hard-negative, and threshold settings."""

    # Primary supervised model.
    lgbm_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 800,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": RANDOM_SEED,
            "verbose": -1,
            "n_jobs": -1,
        }
    )

    # -----------------------------------------------------------------------
    # Hard-negative generation
    # -----------------------------------------------------------------------

    neg_pos_ratio: int = 5
    hard_neg_fraction: float = 0.7

    # -----------------------------------------------------------------------
    # Threshold search
    # -----------------------------------------------------------------------

    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.01


# ---------------------------------------------------------------------------
# Master pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Master configuration grouping all pipeline components."""

    paths: Paths = field(default_factory=Paths)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    random_seed: int = RANDOM_SEED
    val_fraction: float = VAL_FRACTION

    # Set to an integer for debugging/sample experiments.
    sample_size: int | None = None

    # Number of candidate pairs processed per feature-engineering batch.
    chunk_size: int = 50_000

    def __post_init__(self) -> None:
        """Validate configuration values early."""

        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(
                "val_fraction must be between 0 and 1."
            )

        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError(
                "sample_size must be a positive integer or None."
            )

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be a positive integer."
            )

        if not self.blocking.name_prefix_lengths:
            raise ValueError(
                "name_prefix_lengths must contain at least one length."
            )

        if any(
            length <= 0
            for length in self.blocking.name_prefix_lengths
        ):
            raise ValueError(
                "All name prefix lengths must be positive."
            )

        if self.blocking.ngram_size <= 0:
            raise ValueError(
                "ngram_size must be positive."
            )

        if self.blocking.ngram_signature_count <= 0:
            raise ValueError(
                "ngram_signature_count must be positive."
            )

        if self.blocking.min_rare_token_length <= 0:
            raise ValueError(
                "min_rare_token_length must be positive."
            )

        if self.blocking.max_token_frequency <= 0:
            raise ValueError(
                "max_token_frequency must be positive."
            )

        if self.blocking.max_candidates_per_entity <= 0:
            raise ValueError(
                "max_candidates_per_entity must be positive."
            )

        if not 0.0 < self.blocking.blocking_recall_target <= 1.0:
            raise ValueError(
                "blocking_recall_target must be in the range (0, 1]."
            )

        if self.model.neg_pos_ratio <= 0:
            raise ValueError(
                "neg_pos_ratio must be positive."
            )

        if not 0.0 <= self.model.hard_neg_fraction <= 1.0:
            raise ValueError(
                "hard_neg_fraction must be between 0 and 1."
            )

        if not 0.0 <= self.model.threshold_min < 1.0:
            raise ValueError(
                "threshold_min must be in the range [0, 1)."
            )

        if not 0.0 < self.model.threshold_max <= 1.0:
            raise ValueError(
                "threshold_max must be in the range (0, 1]."
            )

        if self.model.threshold_min >= self.model.threshold_max:
            raise ValueError(
                "threshold_min must be smaller than threshold_max."
            )

        if self.model.threshold_step <= 0:
            raise ValueError(
                "threshold_step must be positive."
            )