from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rapidfuzz import fuzz as rfuzz

logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_wratio",
    "name_nosuffix_ratio",
    "name_nosuffix_token_sort",
    "name_compact_ratio",
    "name_transliterated_ratio",
    "name_token_jaccard",
    "name_char3gram_jaccard",
    "name_common_token_ratio",
    "name_length_ratio",
    "name_exact_match",
    "name_first_token_match",
    "name_suffix_match",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_transliterated_ratio",
    "addr_token_jaccard",
    "addr_char3gram_jaccard",
    "addr_number_overlap",
    "addr_number_jaccard",
    "addr_first_number_match",
    "addr_postal_match",
    "addr_length_ratio",
    "addr_has_both",
    "addr_has_neither",
    "addr_has_one",
    "country_match",
    "source_is_s3",
    "name_is_domain",
    "combined_name_addr",
    "name_addr_product",
]


NUM_FEATURES = len(FEATURE_NAMES)


def _char_ngrams(
    text: str,
    n: int = 3,
) -> set[str]:
    if not text:
        return set()

    if len(text) < n:
        return {text}

    return {
        text[index : index + n]
        for index in range(len(text) - n + 1)
    }


def _jaccard(
    set_a: set[str],
    set_b: set[str],
) -> float:
    if not set_a and not set_b:
        return 1.0

    if not set_a or not set_b:
        return 0.0

    union = set_a | set_b

    if not union:
        return 0.0

    return len(set_a & set_b) / len(union)


def _safe_ratio(
    a: float,
    b: float,
) -> float:
    if a == 0 and b == 0:
        return 1.0

    if a == 0 or b == 0:
        return 0.0

    return min(a, b) / max(a, b)


def _common_token_ratio(
    tokens_a: list[str],
    tokens_b: list[str],
) -> float:
    if not tokens_a or not tokens_b:
        return 0.0

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    common = len(set_a & set_b)

    return common / min(
        len(set_a),
        len(set_b),
    )


def _number_overlap(
    nums_a: list[str],
    nums_b: list[str],
) -> float:
    if not nums_a:
        return 1.0 if not nums_b else 0.0

    if not nums_b:
        return 0.0

    set_a = set(nums_a)
    set_b = set(nums_b)

    return len(set_a & set_b) / len(set_a)


def _split_numbers(
    value: str,
) -> list[str]:
    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def _tokens(
    value: str,
) -> set[str]:
    if not value:
        return set()

    return set(value.split())


def _similarity(
    a: str,
    b: str,
) -> tuple[float, float, float, float, float]:
    return (
        rfuzz.ratio(a, b) / 100.0,
        rfuzz.partial_ratio(a, b) / 100.0,
        rfuzz.token_sort_ratio(a, b) / 100.0,
        rfuzz.token_set_ratio(a, b) / 100.0,
        rfuzz.WRatio(a, b) / 100.0,
    )


def compute_pair_features(
    s1_name_norm: str,
    s1_name_compact: str,
    s1_name_tokens_sorted: str,
    s1_name_no_suffix: str,
    s1_name_suffix: str,
    s1_name_transliterated: str,
    s1_name_is_domain: bool,
    s1_addr_norm: str,
    s1_addr_nums_str: str,
    s1_addr_postal: str,
    s1_addr_transliterated: str,
    s1_country: str,
    c_name_norm: str,
    c_name_compact: str,
    c_name_tokens_sorted: str,
    c_name_no_suffix: str,
    c_name_suffix: str,
    c_name_transliterated: str,
    c_name_is_domain: bool,
    c_addr_norm: str,
    c_addr_nums_str: str,
    c_addr_postal: str,
    c_addr_transliterated: str,
    c_country: str,
    c_entity_id: str,
) -> np.ndarray:
    features = np.zeros(
        NUM_FEATURES,
        dtype=np.float32,
    )

    s1_name_norm = str(s1_name_norm or "")
    c_name_norm = str(c_name_norm or "")

    s1_name_compact = str(s1_name_compact or "")
    c_name_compact = str(c_name_compact or "")

    s1_name_tokens_sorted = str(
        s1_name_tokens_sorted or ""
    )
    c_name_tokens_sorted = str(
        c_name_tokens_sorted or ""
    )

    s1_name_no_suffix = str(
        s1_name_no_suffix or ""
    )
    c_name_no_suffix = str(
        c_name_no_suffix or ""
    )

    s1_name_suffix = str(
        s1_name_suffix or ""
    )
    c_name_suffix = str(
        c_name_suffix or ""
    )

    s1_name_transliterated = str(
        s1_name_transliterated or ""
    )
    c_name_transliterated = str(
        c_name_transliterated or ""
    )

    s1_addr_norm = str(s1_addr_norm or "")
    c_addr_norm = str(c_addr_norm or "")

    s1_addr_nums_str = str(
        s1_addr_nums_str or ""
    )
    c_addr_nums_str = str(
        c_addr_nums_str or ""
    )

    s1_addr_postal = str(
        s1_addr_postal or ""
    )
    c_addr_postal = str(
        c_addr_postal or ""
    )

    s1_addr_transliterated = str(
        s1_addr_transliterated or ""
    )
    c_addr_transliterated = str(
        c_addr_transliterated or ""
    )

    s1_country = str(s1_country or "")
    c_country = str(c_country or "")

    index = 0

    (
        name_ratio,
        name_partial,
        name_sort,
        name_set,
        name_wratio,
    ) = _similarity(
        s1_name_norm,
        c_name_norm,
    )

    features[index] = name_ratio
    index += 1

    features[index] = name_partial
    index += 1

    features[index] = name_sort
    index += 1

    features[index] = name_set
    index += 1

    features[index] = name_wratio
    index += 1

    features[index] = (
        rfuzz.ratio(
            s1_name_no_suffix,
            c_name_no_suffix,
        )
        / 100.0
    )
    index += 1

    features[index] = (
        rfuzz.token_sort_ratio(
            s1_name_no_suffix,
            c_name_no_suffix,
        )
        / 100.0
    )
    index += 1

    features[index] = (
        rfuzz.ratio(
            s1_name_compact,
            c_name_compact,
        )
        / 100.0
    )
    index += 1

    features[index] = (
        rfuzz.ratio(
            s1_name_transliterated,
            c_name_transliterated,
        )
        / 100.0
    )
    index += 1

    s1_name_tokens = _tokens(s1_name_norm)
    c_name_tokens = _tokens(c_name_norm)

    features[index] = _jaccard(
        s1_name_tokens,
        c_name_tokens,
    )
    index += 1

    s1_name_ngrams = _char_ngrams(
        s1_name_norm,
        3,
    )
    c_name_ngrams = _char_ngrams(
        c_name_norm,
        3,
    )

    features[index] = _jaccard(
        s1_name_ngrams,
        c_name_ngrams,
    )
    index += 1

    features[index] = _common_token_ratio(
        s1_name_norm.split(),
        c_name_norm.split(),
    )
    index += 1

    features[index] = _safe_ratio(
        len(s1_name_norm),
        len(c_name_norm),
    )
    index += 1

    features[index] = float(
        bool(s1_name_norm)
        and s1_name_norm == c_name_norm
    )
    index += 1

    s1_first = (
        s1_name_norm.split()[0]
        if s1_name_norm.split()
        else ""
    )

    c_first = (
        c_name_norm.split()[0]
        if c_name_norm.split()
        else ""
    )

    features[index] = float(
        bool(s1_first)
        and s1_first == c_first
    )
    index += 1

    features[index] = float(
        bool(s1_name_suffix)
        and s1_name_suffix == c_name_suffix
    )
    index += 1

    (
        addr_ratio,
        _,
        addr_sort,
        addr_set,
        _,
    ) = _similarity(
        s1_addr_norm,
        c_addr_norm,
    )

    features[index] = addr_ratio
    index += 1

    features[index] = addr_sort
    index += 1

    features[index] = addr_set
    index += 1

    features[index] = (
        rfuzz.ratio(
            s1_addr_transliterated,
            c_addr_transliterated,
        )
        / 100.0
    )
    index += 1

    s1_addr_tokens = _tokens(s1_addr_norm)
    c_addr_tokens = _tokens(c_addr_norm)

    features[index] = _jaccard(
        s1_addr_tokens,
        c_addr_tokens,
    )
    index += 1

    s1_addr_ngrams = _char_ngrams(
        s1_addr_norm,
        3,
    )
    c_addr_ngrams = _char_ngrams(
        c_addr_norm,
        3,
    )

    features[index] = _jaccard(
        s1_addr_ngrams,
        c_addr_ngrams,
    )
    index += 1

    s1_nums = _split_numbers(
        s1_addr_nums_str
    )
    c_nums = _split_numbers(
        c_addr_nums_str
    )

    features[index] = _number_overlap(
        s1_nums,
        c_nums,
    )
    index += 1

    features[index] = _jaccard(
        set(s1_nums),
        set(c_nums),
    )
    index += 1

    s1_first_number = (
        s1_nums[0]
        if s1_nums
        else ""
    )

    c_first_number = (
        c_nums[0]
        if c_nums
        else ""
    )

    features[index] = float(
        bool(s1_first_number)
        and s1_first_number == c_first_number
    )
    index += 1

    features[index] = float(
        bool(s1_addr_postal)
        and s1_addr_postal == c_addr_postal
    )
    index += 1

    features[index] = _safe_ratio(
        len(s1_addr_norm),
        len(c_addr_norm),
    )
    index += 1

    s1_has_address = bool(
        s1_addr_norm.strip()
    )
    c_has_address = bool(
        c_addr_norm.strip()
    )

    features[index] = float(
        s1_has_address
        and c_has_address
    )
    index += 1

    features[index] = float(
        not s1_has_address
        and not c_has_address
    )
    index += 1

    features[index] = float(
        s1_has_address != c_has_address
    )
    index += 1

    features[index] = float(
        bool(s1_country)
        and s1_country == c_country
    )
    index += 1

    features[index] = float(
        c_entity_id.startswith("S3-")
    )
    index += 1

    features[index] = float(
        s1_name_is_domain
        or c_name_is_domain
    )
    index += 1

    best_name = max(
        name_ratio,
        name_sort,
        name_set,
        name_wratio,
    )

    best_addr = max(
        addr_ratio,
        addr_sort,
        addr_set,
    )

    features[index] = (
        best_name + best_addr
    ) / 2.0
    index += 1

    features[index] = (
        best_name * best_addr
    )
    index += 1

    if index != NUM_FEATURES:
        raise RuntimeError(
            f"Feature count mismatch: "
            f"{index} != {NUM_FEATURES}"
        )

    return features


_S1_COLS = [
    "name_normalized",
    "name_compact",
    "name_tokens_sorted",
    "name_no_suffix",
    "name_suffix",
    "name_transliterated",
    "name_is_domain",
    "addr_normalized",
    "addr_numbers_str",
    "addr_postal_code",
    "addr_transliterated",
    "country_clean",
]

_CAND_COLS = _S1_COLS


def _row_value(
    row: pd.Series,
    column: str,
    default: str = "",
) -> str:
    value = row.get(
        column,
        default,
    )

    if pd.isna(value):
        return default

    return str(value)


def _row_bool(
    row: pd.Series,
    column: str,
) -> bool:
    value = row.get(
        column,
        False,
    )

    if pd.isna(value):
        return False

    return bool(value)


def _build_row_lookup(
    df: pd.DataFrame,
) -> dict[object, pd.Series]:
    return {
        entity_id: row
        for entity_id, row in df.iterrows()
    }


def _compute_features_from_rows(
    s1_row: pd.Series,
    candidate_row: pd.Series,
    candidate_id: str,
) -> np.ndarray:
    return compute_pair_features(
        s1_name_norm=_row_value(
            s1_row,
            "name_normalized",
        ),
        s1_name_compact=_row_value(
            s1_row,
            "name_compact",
        ),
        s1_name_tokens_sorted=_row_value(
            s1_row,
            "name_tokens_sorted",
        ),
        s1_name_no_suffix=_row_value(
            s1_row,
            "name_no_suffix",
        ),
        s1_name_suffix=_row_value(
            s1_row,
            "name_suffix",
        ),
        s1_name_transliterated=_row_value(
            s1_row,
            "name_transliterated",
        ),
        s1_name_is_domain=_row_bool(
            s1_row,
            "name_is_domain",
        ),
        s1_addr_norm=_row_value(
            s1_row,
            "addr_normalized",
        ),
        s1_addr_nums_str=_row_value(
            s1_row,
            "addr_numbers_str",
        ),
        s1_addr_postal=_row_value(
            s1_row,
            "addr_postal_code",
        ),
        s1_addr_transliterated=_row_value(
            s1_row,
            "addr_transliterated",
        ),
        s1_country=_row_value(
            s1_row,
            "country_clean",
        ),
        c_name_norm=_row_value(
            candidate_row,
            "name_normalized",
        ),
        c_name_compact=_row_value(
            candidate_row,
            "name_compact",
        ),
        c_name_tokens_sorted=_row_value(
            candidate_row,
            "name_tokens_sorted",
        ),
        c_name_no_suffix=_row_value(
            candidate_row,
            "name_no_suffix",
        ),
        c_name_suffix=_row_value(
            candidate_row,
            "name_suffix",
        ),
        c_name_transliterated=_row_value(
            candidate_row,
            "name_transliterated",
        ),
        c_name_is_domain=_row_bool(
            candidate_row,
            "name_is_domain",
        ),
        c_addr_norm=_row_value(
            candidate_row,
            "addr_normalized",
        ),
        c_addr_nums_str=_row_value(
            candidate_row,
            "addr_numbers_str",
        ),
        c_addr_postal=_row_value(
            candidate_row,
            "addr_postal_code",
        ),
        c_addr_transliterated=_row_value(
            candidate_row,
            "addr_transliterated",
        ),
        c_country=_row_value(
            candidate_row,
            "country_clean",
        ),
        c_entity_id=str(candidate_id),
    )


def batch_compute_features(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    chunk_size: int = 50_000,
) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    required_columns = {
        "s1_id",
        "candidate_id",
    }

    missing_columns = (
        required_columns
        - set(pairs_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "pairs_df missing columns: "
            f"{sorted(missing_columns)}"
        )

    n = len(pairs_df)

    logger.info(
        "Computing features for %s pairs...",
        f"{n:,}",
    )

    all_features = np.zeros(
        (n, NUM_FEATURES),
        dtype=np.float32,
    )

    if n == 0:
        return all_features

    s1_lookup = _build_row_lookup(
        s1_df
    )

    s23_lookup = _build_row_lookup(
        s23_df
    )

    s1_ids = pairs_df[
        "s1_id"
    ].to_numpy(
        copy=False
    )

    candidate_ids = pairs_df[
        "candidate_id"
    ].to_numpy(
        copy=False
    )

    for start in range(
        0,
        n,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            n,
        )

        logger.info(
            "Features: %s/%s...",
            f"{start:,}",
            f"{n:,}",
        )

        for index in range(
            start,
            end,
        ):
            s1_id = s1_ids[index]
            candidate_id = candidate_ids[index]

            s1_row = s1_lookup.get(
                s1_id
            )

            candidate_row = s23_lookup.get(
                candidate_id
            )

            if s1_row is None:
                continue

            if candidate_row is None:
                continue

            all_features[index] = (
                _compute_features_from_rows(
                    s1_row,
                    candidate_row,
                    str(candidate_id),
                )
            )

    logger.info(
        "Features computed: %s pairs × %s features.",
        f"{n:,}",
        NUM_FEATURES,
    )

    return all_features