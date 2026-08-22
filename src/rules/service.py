# src/rules/service.py
"""Rule-base scoring: turn four predicted traits into a yam group.

The rule base lives in MySQL (see src/seed.py) and is edited from the admin
page, so it is read through a short-lived process cache instead of a DB round
trip per classification.
"""
import logging
import threading
import time

from src.rules import repository

logger = logging.getLogger(__name__)

TRAIT_KEYS = ("shape", "apex", "base", "margin")

# Falls back to a reload this often even without an explicit invalidation,
# so a row edited straight in the database still takes effect.
CACHE_TTL_SECONDS = 30.0

_cache: list[dict] | None = None
_cache_at = 0.0
_cache_lock = threading.Lock()


def invalidate_cache() -> None:
    """Drop the cached rule base. Called by every write in the router."""
    global _cache, _cache_at
    with _cache_lock:
        _cache = None
        _cache_at = 0.0


def get_rules() -> list[dict]:
    """Active rule groups, cached for CACHE_TTL_SECONDS."""
    global _cache, _cache_at
    now = time.monotonic()
    with _cache_lock:
        if _cache is not None and (now - _cache_at) < CACHE_TTL_SECONDS:
            return _cache

    rules = repository.list_groups(active_only=True)

    with _cache_lock:
        _cache = rules
        _cache_at = time.monotonic()
    return rules


def _same(a, b) -> bool:
    if a is None or b is None:
        return False
    return str(a).strip().lower() == str(b).strip().lower()


def match_group(
    traits: dict[str, str],
    probs: dict[str, float],
    rules: list[dict] | None = None,
    conf_th: float = 0.6,
    top_n: int = 3,
) -> list[dict]:
    """Score every rule group against one prediction, best first.

    Each group names exactly one class per trait, so ranking is primarily a
    count of how many of the four traits line up:

    * a trait that matches counts 1.0
    * a trait the model is unsure about (below conf_th) counts 0.5 whether or
      not it matches — an unsure model should not rule a group out
    * a confident mismatch counts 0

    `score`, the confidence-weighted sum, only breaks ties between groups that
    match equally well. Ranking on confidence alone would let a group matching
    three traits confidently beat a group matching all four less confidently.

    `probs` are 0-1 probabilities, NOT the 0-100 percentages
    classify.service.predict_class returns — callers convert (see
    classify/router.py).
    """
    rules = get_rules() if rules is None else rules
    uncertain = {k for k, p in probs.items() if p is not None and p < conf_th}

    results = []
    for rule in rules:
        matched = 0.0
        score = 0.0
        for key in TRAIT_KEYS:
            if key in uncertain:
                matched += 0.5
                score += 0.5
            elif _same(traits.get(key), rule.get(key)):
                matched += 1.0
                score += float(probs.get(key, 0.0))

        results.append({
            "code": rule["code"],
            "name": rule["name"],
            "matched": round(matched, 2),
            "score": round(score, 3),
        })

    results.sort(key=lambda r: (-r["matched"], -r["score"], r["code"]))
    return results[:top_n]
