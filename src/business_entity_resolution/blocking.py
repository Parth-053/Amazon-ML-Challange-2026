from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from .config import BlockingConfig


logger = logging.getLogger(__name__)


# ============================================================================
# Blocking-key helpers
# ============================================================================


def _name_prefix_keys(
    name_norm: str,
    lengths: List[int],
) -> List[str]:
    if not name_norm:
        return []

    return [
        f"npfx_{name_norm[:length]}"
        for length in lengths
        if len(name_norm) >= length
    ]


def _name_token_keys(
    name_norm: str,
    token_freq: Optional[Dict[str, int]] = None,
    max_freq: int = 5000,
    min_len: int = 4,
) -> List[str]:
    if not name_norm:
        return []

    keys: List[str] = []

    for token in set(name_norm.split()):
        if len(token) < min_len:
            continue

        if token_freq and token_freq.get(token, 0) > max_freq:  # nosec B105
            continue

        keys.append(f"ntok_{token}")

    return keys


def _name_sorted_bigram_keys(
    name_norm: str,
) -> List[str]:
    tokens = sorted(
        {
            token
            for token in name_norm.split()
            if len(token) >= 3
        }
    )

    if len(tokens) < 2:
        return []

    keys: List[str] = []

    for i in range(len(tokens) - 1):
        for j in range(
            i + 1,
            min(i + 3, len(tokens)),
        ):
            keys.append(
                f"nbg_{tokens[i]}_{tokens[j]}"
            )

    return keys


def _numeric_addr_keys(
    addr_numbers: List[str],
    first_name_token: str,
) -> List[str]:
    if not addr_numbers:
        return []

    prefix = (
        first_name_token[:4]
        if len(first_name_token) >= 3
        else ""
    )

    keys: List[str] = []

    for number in addr_numbers[:4]:
        if len(number) >= 2:
            keys.append(
                f"anum_{number}_{prefix}"
            )

    return keys


def _addr_token_keys(
    addr_norm: str,
    token_freq: Optional[Dict[str, int]] = None,
    max_freq: int = 5000,
    min_len: int = 4,
) -> List[str]:
    if not addr_norm:
        return []

    keys: List[str] = []

    for token in set(addr_norm.split()):
        if len(token) < min_len:
            continue

        if token.isdigit():
            continue

        if token_freq and token_freq.get(token, 0) > max_freq:  # nosec B105
            continue

        keys.append(f"atok_{token}")

    return keys


def _compact_name_key(
    name_compact: str,
) -> List[str]:
    if len(name_compact) < 6:
        return []

    return [
        f"ncmp_{name_compact[:8]}",
    ]


def _transliterated_prefix_keys(
    name_transliterated: str,
    lengths: List[int],
) -> List[str]:
    if not name_transliterated:
        return []

    return [
        f"tpfx_{name_transliterated[:length]}"
        for length in lengths
        if len(name_transliterated) >= length
    ]


def _postal_code_key(
    postal_code: str,
    first_name_token: str,
) -> List[str]:
    if not postal_code or len(postal_code) < 4:
        return []

    prefix = (
        first_name_token[:3]
        if first_name_token
        else ""
    )

    return [
        f"pst_{postal_code}_{prefix}",
    ]


def _char_ngram_keys(
    text: str,
    n: int = 3,
    top_k: int = 3,
) -> List[str]:
    if len(text) < n:
        return []

    counts: Dict[str, int] = defaultdict(int)

    for i in range(len(text) - n + 1):
        ngram = text[i : i + n]

        if ngram.strip():
            counts[ngram] += 1

    if not counts:
        return []

    top = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]

    signature = "|".join(
        ngram
        for ngram, _ in top
    )

    return [f"cng_{signature}"]


def _safe_string(value: object) -> str:
    if isinstance(value, str):
        return value

    return ""


def _first_token(name: str) -> str:
    tokens = name.split()
    return tokens[0] if tokens else ""


# ============================================================================
# Token frequencies
# ============================================================================


def compute_token_frequencies(
    dfs: Iterable[pd.DataFrame],
    col: str = "name_normalized",
) -> Dict[str, int]:
    """
    Compute token document frequencies.

    Each token is counted at most once per row.
    DataFrames are processed one at a time.
    """

    frequency: Dict[str, int] = defaultdict(int)

    for df in dfs:
        if col not in df.columns:
            continue

        for value in df[col].values:
            if not isinstance(value, str):
                continue

            for token in set(value.split()):
                if len(token) >= 3:
                    frequency[token] += 1

    return dict(frequency)


# ============================================================================
# Blocking engine
# ============================================================================


class BlockingEngine:
    """
    Multi-pass blocking engine.

    The original key-generation and ranking logic is preserved.

    For full-scale datasets, prefer:
        generate_candidates_to_tsv()

    because retaining candidates for every S1 entity in Python memory is
    substantially more expensive than writing them incrementally to disk.
    """

    def __init__(
        self,
        config: Optional[BlockingConfig] = None,
    ) -> None:
        self.config = config or BlockingConfig()

        self.name_token_freq: Optional[Dict[str, int]] = None
        self.addr_token_freq: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------
    # Token frequencies
    # ------------------------------------------------------------------

    def build_token_frequencies(
        self,
        s2: pd.DataFrame,
        s3: pd.DataFrame,
    ) -> None:
        logger.info(
            "Building token frequencies..."
        )

        self.name_token_freq = compute_token_frequencies(
            [s2, s3],
            col="name_normalized",
        )

        self.addr_token_freq = compute_token_frequencies(
            [s2, s3],
            col="addr_normalized",
        )

        logger.info(
            "Name vocab: %s, Addr vocab: %s",
            f"{len(self.name_token_freq):,}",
            f"{len(self.addr_token_freq):,}",
        )

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def _generate_keys(
        self,
        row: pd.Series,
    ) -> List[str]:
        name_norm = _safe_string(
            row.get("name_normalized", "")
        )

        name_compact = _safe_string(
            row.get("name_compact", "")
        )

        name_transliterated = _safe_string(
            row.get("name_transliterated", "")
        )

        addr_norm = _safe_string(
            row.get("addr_normalized", "")
        )

        addr_numbers = _safe_string(
            row.get("addr_numbers_str", "")
        )

        postal_code = _safe_string(
            row.get("addr_postal_code", "")
        )

        name_no_suffix = _safe_string(
            row.get("name_no_suffix", "")
        )

        numbers = [
            number.strip()
            for number in addr_numbers.split(",")
            if number.strip()
        ]

        first_token = _first_token(name_norm)

        keys: List[str] = []

        # 1. Name prefixes
        keys.extend(
            _name_prefix_keys(
                name_norm,
                self.config.name_prefix_lengths,
            )
        )

        # Name without legal suffix
        if (
            name_no_suffix
            and name_no_suffix != name_norm
        ):
            keys.extend(
                _name_prefix_keys(
                    name_no_suffix,
                    self.config.name_prefix_lengths,
                )
            )

        # 2. Rare name tokens
        if self.config.use_token_blocking:
            keys.extend(
                _name_token_keys(
                    name_norm,
                    self.name_token_freq,
                    self.config.max_token_frequency,
                    self.config.min_rare_token_length,
                )
            )

        # 3. Sorted token bigrams
        if self.config.use_sorted_token_blocking:
            keys.extend(
                _name_sorted_bigram_keys(
                    name_norm
                )
            )

        # 4. Address number + first name token
        if (
            self.config.use_numeric_blocking
            and numbers
        ):
            keys.extend(
                _numeric_addr_keys(
                    numbers,
                    first_token,
                )
            )

        # 5. Compact name
        keys.extend(
            _compact_name_key(
                name_compact
            )
        )

        # 6. Transliteration prefix
        keys.extend(
            _transliterated_prefix_keys(
                name_transliterated,
                [4, 5, 6],
            )
        )

        # 7. Postal + first token
        keys.extend(
            _postal_code_key(
                postal_code,
                first_token,
            )
        )

        # 8. Rare address tokens
        if self.config.use_token_blocking:
            keys.extend(
                _addr_token_keys(
                    addr_norm,
                    self.addr_token_freq,
                    self.config.max_token_frequency,
                    self.config.min_rare_token_length,
                )
            )

        # 9. Character n-gram signature
        if (
            self.config.use_ngram_blocking
            and len(name_norm) >= 5
        ):
            keys.extend(
                _char_ngram_keys(
                    name_norm,
                    self.config.ngram_size,
                    self.config.ngram_signature_count,
                )
            )

        return list(dict.fromkeys(keys))

    # ------------------------------------------------------------------
    # Key scoring
    # ------------------------------------------------------------------

    def _key_weight(
        self,
        key: str,
    ) -> float:
        if key.startswith("ncmp_"):
            return 8.0

        if key.startswith("npfx_"):
            length = len(key) - 5

            if length >= 8:
                return 7.0

            if length >= 6:
                return 5.0

            return 3.0

        if key.startswith("tpfx_"):
            return 4.0

        if key.startswith("nbg_"):
            return 6.0

        if key.startswith("anum_"):
            return 6.0

        if key.startswith("pst_"):
            return 6.0

        if key.startswith("ntok_"):
            return 2.5

        if key.startswith("atok_"):
            return 1.5

        if key.startswith("cng_"):
            return 1.0

        return 1.0

    def _candidate_score(
        self,
        candidate_id: str,
        key_scores: Dict[str, float],
    ) -> tuple:
        return (
            key_scores.get(
                candidate_id,
                0.0,
            ),
            candidate_id,
        )

    # ------------------------------------------------------------------
    # Candidate ranking
    # ------------------------------------------------------------------

    def _rank_candidates(
        self,
        candidate_ids: Set[str],
        candidate_key_scores: Dict[str, float],
        exact_name_candidates: Set[str],
        exact_compact_candidates: Set[str],
        cap: int,
    ) -> Set[str]:
        if len(candidate_ids) <= cap:
            return candidate_ids

        selected: List[str] = []

        priority_groups = (
            exact_name_candidates,
            exact_compact_candidates,
        )

        for group in priority_groups:
            if not group:
                continue

            ranked_group = sorted(
                group & candidate_ids,
                key=lambda candidate_id: (
                    -candidate_key_scores.get(
                        candidate_id,
                        0.0,
                    ),
                    candidate_id,
                ),
            )

            for candidate_id in ranked_group:
                if candidate_id not in selected:
                    selected.append(candidate_id)

                if len(selected) >= cap:
                    return set(selected)

        ranked = sorted(
            candidate_ids,
            key=lambda candidate_id: (
                -candidate_key_scores.get(
                    candidate_id,
                    0.0,
                ),
                candidate_id,
            ),
        )

        for candidate_id in ranked:
            if candidate_id not in selected:
                selected.append(candidate_id)

            if len(selected) >= cap:
                break

        return set(selected)

    # ------------------------------------------------------------------
    # Internal index construction
    # ------------------------------------------------------------------

    def _build_source_index(
        self,
        source_df: pd.DataFrame,
    ) -> Dict[str, Set[str]]:
        """
        Build an inverted key -> entity-ID index for one source.

        S2 and S3 are intentionally indexed separately.
        There is no S2+S3 concatenated DataFrame.
        """

        inverted: Dict[str, Set[str]] = defaultdict(set)

        for entity_id, row in source_df.iterrows():
            keys = self._generate_keys(row)

            for key in set(keys):
                inverted[key].add(str(entity_id))

        return dict(inverted)

    def _merge_source_indexes(
        self,
        first: Dict[str, Set[str]],
        second: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """
        Merge two source indexes key-by-key.

        This avoids constructing a concatenated S2/S3 DataFrame.
        """

        merged: Dict[str, Set[str]] = defaultdict(set)

        for key, entity_ids in first.items():
            merged[key].update(entity_ids)

        for key, entity_ids in second.items():
            merged[key].update(entity_ids)

        return dict(merged)

    # ------------------------------------------------------------------
    # Country candidate generation
    # ------------------------------------------------------------------

    def generate_candidates_for_country(
        self,
        s1_country: pd.DataFrame,
        s23_country: pd.DataFrame,
        country: str,
    ) -> Dict[str, Set[str]]:
        """
        Backward-compatible in-memory country blocking.

        This method remains compatible with the original pipeline.
        Full-scale processing should use generate_candidates_to_tsv().
        """

        logger.info(
            "Blocking country='%s': S1=%s, S2+S3=%s",
            country,
            f"{len(s1_country):,}",
            f"{len(s23_country):,}",
        )

        inverted = self._build_source_index(
            s23_country
        )

        logger.info(
            "Inverted index: %s unique keys",
            f"{len(inverted):,}",
        )

        cap = max(
            int(
                self.config.max_candidates_per_entity
            ),
            1,
        )

        candidates: Dict[str, Set[str]] = {}

        total_pairs = 0
        capped = 0

        for entity_id, row in s1_country.iterrows():
            keys = list(
                dict.fromkeys(
                    self._generate_keys(row)
                )
            )

            if not keys:
                candidates[entity_id] = set()
                continue

            candidate_ids: Set[str] = set()

            candidate_key_scores: Dict[
                str,
                float,
            ] = defaultdict(float)

            exact_name_candidates: Set[str] = set()
            exact_compact_candidates: Set[str] = set()

            name_norm = _safe_string(
                row.get(
                    "name_normalized",
                    "",
                )
            )

            name_compact = _safe_string(
                row.get(
                    "name_compact",
                    "",
                )
            )

            for key in keys:
                matched_ids = inverted.get(key)

                if not matched_ids:
                    continue

                weight = self._key_weight(key)

                for candidate_id in matched_ids:
                    candidate_ids.add(
                        candidate_id
                    )

                    candidate_key_scores[
                        candidate_id
                    ] += weight

                    if (
                        key.startswith("npfx_")
                        and len(name_norm) >= 8
                        and key
                        == f"npfx_{name_norm[:8]}"
                    ):
                        exact_name_candidates.add(
                            candidate_id
                        )

                    if (
                        name_compact
                        and key
                        == f"ncmp_{name_compact[:8]}"
                    ):
                        exact_compact_candidates.add(
                            candidate_id
                        )

            if len(candidate_ids) > cap:
                candidate_ids = (
                    self._rank_candidates(
                        candidate_ids,
                        candidate_key_scores,
                        exact_name_candidates,
                        exact_compact_candidates,
                        cap,
                    )
                )

                capped += 1

            candidates[entity_id] = candidate_ids
            total_pairs += len(candidate_ids)

        avg_candidates = (
            total_pairs
            / max(len(s1_country), 1)
        )

        logger.info(
            "Candidates: %s total pairs, avg %.1f per S1, %s capped at %s",
            f"{total_pairs:,}",
            avg_candidates,
            f"{capped:,}",
            f"{cap:,}",
        )

        return candidates

    # ------------------------------------------------------------------
    # Streaming source candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates_for_s1_chunk(
        self,
        s1_chunk: pd.DataFrame,
        inverted: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """
        Generate candidates for one S1 chunk.

        Only this chunk's candidate sets are retained.
        """

        cap = max(
            int(
                self.config.max_candidates_per_entity
            ),
            1,
        )

        candidates: Dict[str, Set[str]] = {}

        for entity_id, row in s1_chunk.iterrows():
            keys = list(
                dict.fromkeys(
                    self._generate_keys(row)
                )
            )

            if not keys:
                candidates[entity_id] = set()
                continue

            candidate_ids: Set[str] = set()

            candidate_key_scores: Dict[
                str,
                float,
            ] = defaultdict(float)

            exact_name_candidates: Set[str] = set()
            exact_compact_candidates: Set[str] = set()

            name_norm = _safe_string(
                row.get(
                    "name_normalized",
                    "",
                )
            )

            name_compact = _safe_string(
                row.get(
                    "name_compact",
                    "",
                )
            )

            for key in keys:
                matched_ids = inverted.get(key)

                if not matched_ids:
                    continue

                weight = self._key_weight(key)

                for candidate_id in matched_ids:
                    candidate_ids.add(
                        candidate_id
                    )

                    candidate_key_scores[
                        candidate_id
                    ] += weight

                    if (
                        key.startswith("npfx_")
                        and len(name_norm) >= 8
                        and key
                        == f"npfx_{name_norm[:8]}"
                    ):
                        exact_name_candidates.add(
                            candidate_id
                        )

                    if (
                        name_compact
                        and key
                        == f"ncmp_{name_compact[:8]}"
                    ):
                        exact_compact_candidates.add(
                            candidate_id
                        )

            if len(candidate_ids) > cap:
                candidate_ids = (
                    self._rank_candidates(
                        candidate_ids,
                        candidate_key_scores,
                        exact_name_candidates,
                        exact_compact_candidates,
                        cap,
                    )
                )

            candidates[
                str(entity_id)
            ] = candidate_ids

        return candidates

    # ------------------------------------------------------------------
    # Full-scale disk-backed blocking
    # ------------------------------------------------------------------

    def generate_candidates_to_tsv(
        self,
        s1: pd.DataFrame,
        s2: pd.DataFrame,
        s3: pd.DataFrame,
        output_path: Path,
        s1_chunk_size: int = 25_000,
    ) -> Dict[str, int]:
        """
        Full-scale candidate generation.

        Important:
        - No pd.concat([s2, s3]).
        - S2 and S3 are indexed separately.
        - Candidate results are written incrementally.
        - Only one S1 chunk's candidate sets exist at a time.
        - Every S1 entity is written exactly once.
        """

        if s1_chunk_size <= 0:
            raise ValueError(
                "s1_chunk_size must be positive."
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        required_columns = {
            "entity_id",
            "country_clean",
        }

        for label, df in (
            ("S1", s1),
            ("S2", s2),
            ("S3", s3),
        ):
            missing = (
                required_columns
                - set(df.columns)
            )

            if missing:
                raise ValueError(
                    f"{label} missing columns: "
                    f"{sorted(missing)}"
                )

        logger.info(
            "Starting disk-backed blocking: "
            "S1=%s, S2=%s, S3=%s",
            f"{len(s1):,}",
            f"{len(s2):,}",
            f"{len(s3):,}",
        )

        # --------------------------------------------------------------
        # Token frequencies
        # --------------------------------------------------------------

        self.build_token_frequencies(
            s2,
            s3,
        )

        # --------------------------------------------------------------
        # Build S2 and S3 indexes independently.
        # --------------------------------------------------------------

        logger.info(
            "Building S2 blocking index..."
        )

        s2_index = self._build_source_index(
            s2
        )

        logger.info(
            "S2 index: %s keys",
            f"{len(s2_index):,}",
        )

        logger.info(
            "Building S3 blocking index..."
        )

        s3_index = self._build_source_index(
            s3
        )

        logger.info(
            "S3 index: %s keys",
            f"{len(s3_index):,}",
        )

        # --------------------------------------------------------------
        # Combine only the indexes, never the DataFrames.
        # --------------------------------------------------------------

        logger.info(
            "Merging S2/S3 inverted indexes..."
        )

        inverted = self._merge_source_indexes(
            s2_index,
            s3_index,
        )

        del s2_index
        del s3_index

        logger.info(
            "Combined blocking index: %s keys",
            f"{len(inverted):,}",
        )

        # --------------------------------------------------------------
        # Output header
        # --------------------------------------------------------------

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "source1_entity_id\t"
                "candidate_entity_ids\n"
            )

            total_pairs = 0
            entities_with_candidates = 0
            capped_entities = 0

            # ----------------------------------------------------------
            # Process S1 in chunks.
            # ----------------------------------------------------------

            for start in range(
                0,
                len(s1),
                s1_chunk_size,
            ):
                stop = min(
                    start + s1_chunk_size,
                    len(s1),
                )

                s1_chunk = s1.iloc[
                    start:stop
                ]

                logger.info(
                    "Blocking S1 rows %s-%s / %s",
                    f"{start:,}",
                    f"{stop:,}",
                    f"{len(s1):,}",
                )

                chunk_candidates = (
                    self._generate_candidates_for_s1_chunk(
                        s1_chunk,
                        inverted,
                    )
                )

                for s1_id in s1_chunk[
                    "entity_id"
                ].astype(str):

                    candidate_ids = (
                        chunk_candidates.get(
                            s1_id,
                            set(),
                        )
                    )

                    if candidate_ids:
                        entities_with_candidates += 1

                    total_pairs += len(
                        candidate_ids
                    )

                    if (
                        len(candidate_ids)
                        >= self.config.max_candidates_per_entity
                    ):
                        capped_entities += 1

                    candidate_string = (
                        ",".join(
                            sorted(candidate_ids)
                        )
                        if candidate_ids
                        else ""
                    )

                    file.write(
                        f"{s1_id}\t"
                        f"{candidate_string}\n"
                    )

                del chunk_candidates

        logger.info(
            "Disk-backed blocking complete: "
            "%s candidate pairs for %s S1 entities; "
            "%s entities with candidates; "
            "%s at cap",
            f"{total_pairs:,}",
            f"{len(s1):,}",
            f"{entities_with_candidates:,}",
            f"{capped_entities:,}",
        )

        return {
            "s1_entities": len(s1),
            "candidate_pairs": total_pairs,
            "entities_with_candidates": (
                entities_with_candidates
            ),
            "capped_entities": capped_entities,
        }

    # ------------------------------------------------------------------
    # Backward-compatible in-memory API
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        s1: pd.DataFrame,
        s2: pd.DataFrame,
        s3: pd.DataFrame,
    ) -> Dict[str, Set[str]]:
        """
        Backward-compatible in-memory API.

        This should be used for small/controlled datasets only.

        For the complete challenge dataset use:
            generate_candidates_to_tsv()
        """

        self.build_token_frequencies(
            s2,
            s3,
        )

        s2_index = self._build_source_index(
            s2
        )

        s3_index = self._build_source_index(
            s3
        )

        inverted = self._merge_source_indexes(
            s2_index,
            s3_index,
        )

        del s2_index
        del s3_index

        all_candidates: Dict[
            str,
            Set[str],
        ] = {}

        countries = (
            s1["country_clean"]
            .dropna()
            .unique()
        )

        logger.info(
            "Countries in S1: %s",
            sorted(countries),
        )

        for country in countries:
            s1_country = s1[
                s1["country_clean"] == country
            ]

            if s1_country.empty:
                continue

            country_candidates = (
                self._generate_candidates_for_s1_chunk(
                    s1_country,
                    inverted,
                )
            )

            all_candidates.update(
                country_candidates
            )

            del country_candidates

        for entity_id in s1["entity_id"].astype(str):
            all_candidates.setdefault(
                entity_id,
                set(),
            )

        total = sum(
            len(candidate_ids)
            for candidate_ids in all_candidates.values()
        )

        with_candidates = sum(
            bool(candidate_ids)
            for candidate_ids in all_candidates.values()
        )

        logger.info(
            "Total candidates: %s pairs for %s S1 entities (%s with >=1 candidate)",
            f"{total:,}",
            f"{len(all_candidates):,}",
            f"{with_candidates:,}",
        )

        return all_candidates


# ============================================================================
# Candidate output helpers
# ============================================================================


def candidates_to_dataframe(
    candidates: Dict[str, Set[str]],
) -> pd.DataFrame:
    rows = [
        (s1_id, candidate_id)
        for s1_id, candidate_ids in candidates.items()
        for candidate_id in candidate_ids
    ]

    df = pd.DataFrame(
        rows,
        columns=[
            "s1_id",
            "candidate_id",
        ],
    )

    logger.info(
        "Candidate pairs DataFrame: %s rows",
        f"{len(df):,}",
    )

    return df


def write_candidate_pairs_tsv(
    candidates: Dict[str, Set[str]],
    output_path,
    all_s1_ids: Optional[List[str]] = None,
) -> None:
    s1_ids = (
        all_s1_ids
        if all_s1_ids is not None
        else sorted(candidates.keys())
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "source1_entity_id\t"
            "candidate_entity_ids\n"
        )

        for s1_id in s1_ids:
            candidate_ids = candidates.get(
                s1_id,
                set(),
            )

            candidate_string = (
                ",".join(
                    sorted(candidate_ids)
                )
                if candidate_ids
                else ""
            )

            file.write(
                f"{s1_id}\t"
                f"{candidate_string}\n"
            )

    logger.info(
        "Wrote candidate_pairs.tsv: %s",
        output_path,
    )