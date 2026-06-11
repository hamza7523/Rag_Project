import re
from typing import Any, Dict, List, Optional

from text_utils import parse_numeric_fields


def _simple_tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower())]


def _metadata_matches(metadata: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
    if not metadata_filter:
        return True
    for k, v in metadata_filter.items():
        mv = metadata.get(k)
        if mv is None:
            return False
        # simple string containment match (case-insensitive)
        try:
            if isinstance(mv, (list, tuple)):
                if not any(str(v).lower() in str(x).lower() for x in mv):
                    return False
            else:
                if str(v).lower() not in str(mv).lower():
                    return False
        except Exception:
            return False
    return True


BRAND_KEYWORDS = {
    "pepsi": "Pepsi",
    "pepsi co": "Pepsi",
    "mountain dew": "Dew",
    "dew": "Dew",
    "marinda": "Marinda",
}


def build_brand_metadata_filter(query: str) -> Optional[Dict[str, str]]:
    """Extract a brand source filter from the natural language query."""
    text = query.lower()
    for keyword, source_value in BRAND_KEYWORDS.items():
        if keyword in text:
            return {"source": source_value}
    return None


def merge_metadata_filters(base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if base is None:
        return override
    if override is None:
        return base
    merged = base.copy()
    merged.update(override)
    return merged


def hybrid_rerank_candidates(
    semantic_results: List[Dict[str, Any]],
    query: str,
    metadata_filter: Dict[str, Any] = None,
    top_k: int = 5,
):
    """Rerank semantic candidates by applying an optional metadata filter
    then boosting exact numeric or token matches from the query.

    Args:
        semantic_results: List of {'content','metadata','score'} produced by a semantic search.
        query: user query string.
        metadata_filter: optional dict to filter candidates by metadata inclusion.
        top_k: number of final candidates to return.

    Returns:
        List of candidates ordered by final score.
    """

    # filter by metadata first
    candidates = [c for c in semantic_results if _metadata_matches(c.get("metadata", {}), metadata_filter)]

    # extract numeric tokens from query for exact-match boosts
    q_fields, _ = parse_numeric_fields(query)
    query_numbers = set()
    for vals in q_fields.values():
        for v in vals:
            query_numbers.add(str(v))

    q_tokens = _simple_tokenize(query)

    scored = []
    for c in candidates:
        content = c.get("content", "")
        metadata = c.get("metadata", {})
        base_score = float(c.get("score", 0.0))

        # token overlap score
        content_tokens = _simple_tokenize(content)
        overlap = sum(1 for t in q_tokens if t in content_tokens)

        # numeric exact matches in metadata or content
        num_matches = 0
        for n in query_numbers:
            if n in content or any(n in str(v) for v in metadata.values()):
                num_matches += 1

        # combine scores: keep base semantic score and add boosts
        final_score = base_score + 0.1 * overlap + 1.0 * num_matches
        scored.append((final_score, {"content": content, "metadata": metadata}))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]
