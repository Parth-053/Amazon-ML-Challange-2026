"""
Text preprocessing for business names and addresses.

Produces multiple representations for blocking and feature engineering.
The preprocessing is country-agnostic.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)


LEGAL_SUFFIXES = {
    "llc",
    "llp",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "pvt",
    "private",
    "plc",
    "lp",
    "pllc",
    "pc",
    "pa",
    "na",
    "associates",
    "enterprises",
    "enterprise",
    "services",
    "solutions",
    "group",
    "holdings",
    "international",
    "sarl",
    "sas",
    "sa",
    "eurl",
    "sci",
    "scp",
    "snc",
    "sasu",
    "selarl",
    "opc",
    "ngo",
    "trust",
    "foundation",
    "society",
}

ADDRESS_ABBREVS = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "expy": "expressway",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "rte": "route",
    "cty": "city",
    "twp": "township",
    "dist": "district",
    "tq": "taluk",
    "distt": "district",
}

ADDRESS_FILLERS = {
    "c/o",
    "co",
    "near",
    "opp",
    "behind",
    "beside",
    "next",
    "opposite",
    "adjacent",
    "above",
    "below",
    "nr",
    "n/a",
    "na",
    "nil",
    "none",
    "unknown",
}

DOMAIN_TLD_PATTERN = (
    r"\.(?:com|org|net|co|in|io|biz|fr)$"
)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode and remove combining marks."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKD", text)

    return "".join(
        char
        for char in normalized
        if unicodedata.category(char) != "Mn"
    )


def transliterate_basic(text: str) -> str:
    """Transliterate text with unidecode when available."""
    if not text:
        return ""

    try:
        from unidecode import unidecode

        return unidecode(text)
    except ImportError:
        return normalize_unicode(text)


def has_non_latin(text: str) -> bool:
    """Return True when alphabetic characters include non-Latin scripts."""
    for char in text:
        if not char.isalpha():
            continue

        category = unicodedata.category(char)
        name = unicodedata.name(char, "")

        if (
            category.startswith("L")
            and "LATIN" not in name
            and "COMMON" not in name
        ):
            return True

    return False


def _clean_base(text: str) -> str:
    """Apply shared low-level text cleaning."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = normalize_unicode(text)
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"[\(\[\{<]", " ", text)
    text = re.sub(r"[\)\]\}>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _is_domain_name(name: str) -> bool:
    """Detect domain-style business names."""
    return bool(
        re.search(DOMAIN_TLD_PATTERN, name)
    )


def _domain_to_tokens(name: str) -> str:
    """Convert a domain-style name into space-separated tokens."""
    name = re.sub(
        DOMAIN_TLD_PATTERN,
        "",
        name,
    )
    name = name.replace(".", " ")
    name = re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        name,
    )

    return name.lower().strip()


def strip_legal_suffix(
    name: str,
) -> tuple[str, str]:
    """Remove trailing legal suffix tokens."""
    tokens = name.split()

    if len(tokens) <= 1:
        return name, ""

    suffix_parts: list[str] = []
    index = len(tokens) - 1

    while index >= 1:
        token = re.sub(
            r"[^a-z]",
            "",
            tokens[index],
        )

        if token in LEGAL_SUFFIXES:
            suffix_parts.append(tokens[index])
            index -= 1
        else:
            break

    if not suffix_parts:
        return name, ""

    core = " ".join(tokens[: index + 1])
    suffix = " ".join(reversed(suffix_parts))

    return core.strip(), suffix.strip()


def normalize_name(raw_name: str) -> dict:
    """Produce multiple representations of a business name."""
    result = {
        "raw": raw_name if isinstance(raw_name, str) else "",
        "normalized": "",
        "compact": "",
        "tokens": [],
        "tokens_sorted": "",
        "no_suffix": "",
        "suffix": "",
        "is_domain": False,
        "transliterated": "",
    }

    if not isinstance(raw_name, str) or not raw_name.strip():
        return result

    base = _clean_base(raw_name)

    is_domain = _is_domain_name(base)

    if is_domain:
        base = _domain_to_tokens(base)

    result["is_domain"] = is_domain

    normalized = re.sub(
        r"[^a-z0-9\s]",
        " ",
        base,
    )
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    result["normalized"] = normalized

    transliterated = transliterate_basic(raw_name).lower()
    transliterated = re.sub(
        r"[^a-z0-9\s]",
        " ",
        transliterated,
    )
    result["transliterated"] = re.sub(
        r"\s+",
        " ",
        transliterated,
    ).strip()

    result["compact"] = re.sub(
        r"\s",
        "",
        normalized,
    )

    tokens = normalized.split()

    result["tokens"] = tokens
    result["tokens_sorted"] = " ".join(
        sorted(tokens)
    )

    no_suffix, suffix = strip_legal_suffix(
        normalized
    )

    result["no_suffix"] = no_suffix
    result["suffix"] = suffix

    return result


def extract_numbers(text: str) -> list[str]:
    """Extract numeric tokens from address text."""
    if not text:
        return []

    return re.findall(
        r"\b\d+(?:[/-]\d+)*\b",
        text,
    )


def extract_postal_code(text: str) -> str:
    """Extract a likely postal/ZIP/PIN code."""
    if not text:
        return ""

    match = re.search(
        r"\b(\d{5}(?:-\d{4})?)\b",
        text,
    )

    if match:
        return match.group(1)[:5]

    match = re.search(
        r"\b(\d{6})\b",
        text,
    )

    if match:
        return match.group(1)

    return ""


def normalize_address(raw_addr: str) -> dict:
    """Produce multiple representations of a business address."""
    result = {
        "raw": raw_addr if isinstance(raw_addr, str) else "",
        "normalized": "",
        "compact": "",
        "tokens": [],
        "tokens_sorted": "",
        "numbers": [],
        "postal_code": "",
        "transliterated": "",
    }

    if not isinstance(raw_addr, str) or not raw_addr.strip():
        return result

    base = _clean_base(raw_addr)

    result["numbers"] = extract_numbers(base)
    result["postal_code"] = extract_postal_code(base)

    transliterated = transliterate_basic(
        raw_addr
    ).lower()

    transliterated = re.sub(
        r"[^a-z0-9\s]",
        " ",
        transliterated,
    )

    result["transliterated"] = re.sub(
        r"\s+",
        " ",
        transliterated,
    ).strip()

    normalized = re.sub(
        r"[^a-z0-9\s]",
        " ",
        base,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    tokens = normalized.split()

    expanded = [
        ADDRESS_ABBREVS.get(
            token,
            token,
        )
        for token in tokens
    ]

    normalized = " ".join(expanded)

    result["normalized"] = normalized

    compact_tokens = [
        token
        for token in expanded
        if token not in ADDRESS_FILLERS
    ]

    result["compact"] = "".join(
        compact_tokens
    )

    result["tokens"] = expanded
    result["tokens_sorted"] = " ".join(
        sorted(set(expanded))
    )

    return result


def preprocess_dataframe(
    df: pd.DataFrame,
    label: str = "",
) -> pd.DataFrame:
    """
    Add normalized name and address columns.

    The DataFrame is modified in place and returned.
    """
    logger.info(
        "Preprocessing %s (%d rows) ...",
        label,
        len(df),
    )

    df["business_name"] = (
        df["business_name"]
        .fillna("")
        .astype(str)
    )

    df["business_address"] = (
        df["business_address"]
        .fillna("")
        .astype(str)
    )

    df["country"] = (
        df["country"]
        .fillna("")
        .astype(str)
    )

    name_data = df["business_name"].apply(
        normalize_name
    )

    df["name_normalized"] = name_data.map(
        lambda value: value["normalized"]
    )
    df["name_compact"] = name_data.map(
        lambda value: value["compact"]
    )
    df["name_tokens_sorted"] = name_data.map(
        lambda value: value["tokens_sorted"]
    )
    df["name_no_suffix"] = name_data.map(
        lambda value: value["no_suffix"]
    )
    df["name_suffix"] = name_data.map(
        lambda value: value["suffix"]
    )
    df["name_transliterated"] = name_data.map(
        lambda value: value["transliterated"]
    )
    df["name_is_domain"] = name_data.map(
        lambda value: value["is_domain"]
    )

    address_data = df["business_address"].apply(
        normalize_address
    )

    df["addr_normalized"] = address_data.map(
        lambda value: value["normalized"]
    )
    df["addr_compact"] = address_data.map(
        lambda value: value["compact"]
    )
    df["addr_tokens_sorted"] = address_data.map(
        lambda value: value["tokens_sorted"]
    )
    df["addr_numbers_str"] = address_data.map(
        lambda value: ",".join(value["numbers"])
    )
    df["addr_postal_code"] = address_data.map(
        lambda value: value["postal_code"]
    )
    df["addr_transliterated"] = address_data.map(
        lambda value: value["transliterated"]
    )

    df["country_clean"] = (
        df["country"]
        .str.strip()
        .str.lower()
    )

    logger.info(
        "Preprocessing %s done.",
        label,
    )

    return df


def preprocess_dataframe_chunked(
    df: pd.DataFrame,
    label: str = "",
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """Preprocess a large DataFrame in chunks."""
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive."
        )

    if len(df) <= chunk_size:
        return preprocess_dataframe(
            df,
            label,
        )

    logger.info(
        "Chunked preprocessing %s: %d rows, chunk=%d.",
        label,
        len(df),
        chunk_size,
    )

    chunks: list[pd.DataFrame] = []

    for start in range(
        0,
        len(df),
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            len(df),
        )

        chunk = df.iloc[
            start:end
        ].copy()

        preprocess_dataframe(
            chunk,
            f"{label}[{start}:{end}]",
        )

        chunks.append(chunk)

    result = pd.concat(
        chunks,
        ignore_index=False,
    )

    logger.info(
        "Chunked preprocessing %s done.",
        label,
    )

    return result