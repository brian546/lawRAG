import random
import re
from typing import Iterable, List, Sequence

import numpy as np

MULTISPACE_RE = re.compile(r"\s+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return MULTISPACE_RE.sub(" ", str(text)).strip()


def split_citations(raw: str | None) -> List[str]:
    if raw is None:
        return []
    parts = [normalize_text(part) for part in str(raw).split(";")]
    return [part for part in parts if part]


def join_citations(citations: Sequence[str]) -> str:
    seen = set()
    out = []
    for citation in citations:
        item = normalize_text(citation)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return ";".join(out)


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
