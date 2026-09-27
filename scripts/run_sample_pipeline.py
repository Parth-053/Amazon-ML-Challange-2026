import logging
import runpy
import sys
from pathlib import Path

SAMPLE_SIZE = 5_000


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def _run_validator(
    validator_path: Path,
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
) -> int:
    """Run the official validator without spawning a subprocess."""

    original_argv = sys.argv.copy()

    sys.argv = [
        str(validator_path),
        "--matching",
        str(matching_path),
        "--candidate",
        str(candidate_path),
        "--test-dir",
        str(test_dir),
    ]

    try:
        runpy.run_path(
            str(validator_path),
            run_name="__main__",
        )
    except SystemExit as exc:
        if exc.code is None:
            return 0

        if isinstance(exc.code, int):
            return exc.code

        return 1
    except Exception:
        logger.exception(
            "Official validator failed"
        )
        return 1
    finally:
        sys.argv = original_argv

    return 0


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from business_entity_resolution.config import PipelineConfig
    from business_entity_resolution.pipeline import (
        run_inference_pipeline,
        run_training_pipeline,
    )

    logger.info(
        "Initializing configuration for sample run..."
    )

    config = PipelineConfig()

    config.model.lgbm_params["n_estimators"] = 50
    config.chunk_size = 50_000

    logger.info(
        "Running sample pipeline with %s S1 entities",
        f"{SAMPLE_SIZE:,}",
    )

    try:
        run_training_pipeline(
            config,
            sample_size=SAMPLE_SIZE,
        )
    except Exception:
        logger.exception(
            "Training pipeline failed"
        )
        return 1

    logger.info(
        "--- Sample Training Completed Successfully ---"
    )

    try:
        run_inference_pipeline(
            config,
            sample_size=SAMPLE_SIZE,
        )
    except Exception:
        logger.exception(
            "Inference pipeline failed"
        )
        return 1

    logger.info(
        "--- Sample Inference Completed Successfully ---"
    )

    validator_path = (
        config.paths.utils_dir
        / "validate_submission.py"
    )

    if not validator_path.exists():
        logger.warning(
            "Validator script not found at %s",
            validator_path,
        )
        return 0

    logger.info(
        "Running official validator..."
    )

    validator_status = _run_validator(
        validator_path,
        config.paths.matching_output,
        config.paths.candidate_output,
        config.paths.test_dir,
    )

    if validator_status != 0:
        logger.error(
            "--- Validator FAILED ---"
        )
        return validator_status

    logger.info(
        "--- Validator PASSED ---"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())