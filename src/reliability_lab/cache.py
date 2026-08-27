from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Shared utilities — use these in both ResponseCache and SharedRedisCache
# ---------------------------------------------------------------------------

PRIVACY_PATTERNS = re.compile(
    r"\b(balance|password|credit.card|ssn|social.security|user.\d+|account.\d+)\b",
    re.IGNORECASE,
)


def _is_uncacheable(query: str) -> bool:
    """Return True if query contains privacy-sensitive keywords."""
    return bool(PRIVACY_PATTERNS.search(query))


def _looks_like_false_hit(query: str, cached_key: str) -> bool:
    """Return True if query and cached key contain different 4-digit numbers (years, IDs)."""
    nums_q = set(re.findall(r"\b\d{4}\b", query))
    nums_c = set(re.findall(r"\b\d{4}\b", cached_key))
    return bool(nums_q and nums_c and nums_q != nums_c)


# ---------------------------------------------------------------------------
# In-memory cache (existing)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CacheEntry:
    key: str
    value: str
    created_at: float
    metadata: dict[str, str]


class ResponseCache:
    """In-memory semantic cache with privacy and false-hit guardrails.

    Lookup matches on cosine similarity over word tokens + character 3-grams, so
    paraphrases hit while topic drift does not.  Two guards sit on top: queries
    matching PRIVACY_PATTERNS are never stored or served, and a match whose 4-digit
    numbers disagree with the query's is rejected and recorded in false_hit_log.

    Single-process only — see SharedRedisCache for multi-instance deployments.
    """

    def __init__(self, ttl_seconds: int, similarity_threshold: float):
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self._entries: list[CacheEntry] = []
        self.false_hit_log: list[dict[str, object]] = []

    def get(self, query: str) -> tuple[str | None, float]:
        """Look up a cached response by semantic similarity."""
        if _is_uncacheable(query):
            return None, 0.0

        now = time.time()
        self._entries = [e for e in self._entries if now - e.created_at <= self.ttl_seconds]

        best_entry: CacheEntry | None = None
        best_score = 0.0
        for entry in self._entries:
            score = self.similarity(query, entry.key)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < self.similarity_threshold:
            return None, best_score

        if _looks_like_false_hit(query, best_entry.key):
            self.false_hit_log.append(
                {
                    "query": query,
                    "cached_key": best_entry.key,
                    "score": best_score,
                    "reason": "date_or_number_mismatch",
                }
            )
            return None, best_score

        return best_entry.value, best_score

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        """Store a response in cache."""
        if _is_uncacheable(query):
            return
        self._entries.append(
            CacheEntry(key=query, value=value, created_at=time.time(), metadata=metadata or {})
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into word tokens plus character 3-grams within each word."""
        tokens: list[str] = []
        for word in re.findall(r"\w+", text.lower()):
            tokens.append(word)
            tokens.extend(word[i : i + 3] for i in range(len(word) - 2))
        return tokens

    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Cosine similarity over word tokens + character 3-grams."""
        if a == b:
            return 1.0
        vec_a = Counter(ResponseCache._tokenize(a))
        vec_b = Counter(ResponseCache._tokenize(b))
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(count * vec_b[token] for token, count in vec_a.items())
        norm_a = math.sqrt(sum(count * count for count in vec_a.values()))
        norm_b = math.sqrt(sum(count * count for count in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Redis shared cache (new)
# ---------------------------------------------------------------------------


class SharedRedisCache:
    """Redis-backed shared cache, so cache state survives restarts and is shared
    across every gateway instance instead of being rebuilt per process.

    Data model:
        Key    = "{prefix}{query_hash}"   — deterministic, so any instance computes
                 the same key for the same question (enables the O(1) exact path)
        Value  = Redis Hash with fields "query" and "response".  The original query
                 is stored, not just the response, because the similarity scan has
                 to read the old question back to compare it against the new one.
        TTL    = Redis EXPIRE — automatic cleanup, no manual eviction pass

    Lookup is two-tier: an exact hash hit returns immediately; on a miss it SCANs the
    prefix and scores each stored query with ResponseCache.similarity().  Both the
    privacy and false-hit guardrails apply exactly as in ResponseCache — dropping
    either here would leak sensitive queries into shared, persistent storage.
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        similarity_threshold: float,
        prefix: str = "rl:cache:",
    ):
        import redis as redis_lib

        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.prefix = prefix
        self.false_hit_log: list[dict[str, object]] = []
        self._redis: Any = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return bool(self._redis.ping())
        except Exception:
            return False

    def get(self, query: str) -> tuple[str | None, float]:
        """Look up a cached response from Redis."""
        if _is_uncacheable(query):
            return None, 0.0

        exact_key = f"{self.prefix}{self._query_hash(query)}"
        exact = self._redis.hget(exact_key, "response")
        if exact is not None:
            return str(exact), 1.0

        best_query: str | None = None
        best_value: str | None = None
        best_score = 0.0
        for key in self._redis.scan_iter(f"{self.prefix}*"):
            cached_query = self._redis.hget(key, "query")
            if cached_query is None:
                continue
            score = ResponseCache.similarity(query, str(cached_query))
            if score > best_score:
                best_score = score
                best_query = str(cached_query)
                best_value = self._redis.hget(key, "response")

        if best_value is None or best_score < self.similarity_threshold:
            return None, best_score

        if best_query is not None and _looks_like_false_hit(query, best_query):
            self.false_hit_log.append(
                {
                    "query": query,
                    "cached_key": best_query,
                    "score": best_score,
                    "reason": "date_or_number_mismatch",
                }
            )
            return None, best_score

        return str(best_value), best_score

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        """Store a response in Redis with TTL."""
        if _is_uncacheable(query):
            return
        key = f"{self.prefix}{self._query_hash(query)}"
        self._redis.hset(key, mapping={"query": query, "response": value})
        self._redis.expire(key, self.ttl_seconds)

    def flush(self) -> None:
        """Remove all entries with this cache prefix (for testing)."""
        for key in self._redis.scan_iter(f"{self.prefix}*"):
            self._redis.delete(key)

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            self._redis.close()

    @staticmethod
    def _query_hash(query: str) -> str:
        """Deterministic short hash for a query string."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]