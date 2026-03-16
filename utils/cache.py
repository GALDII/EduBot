import sys
import os
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Any
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import CACHE_TTL, MAX_CACHE_SIZE

_cache = {}
_cache_timestamps = {}


def _generate_key(query: str, context: dict = None) -> str:
    """Generate cache key from query and context."""
    key_data = {"query": query.lower().strip(), "context": context or {}}
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_response(query: str, context: dict = None) -> Optional[Any]:
    """Get cached LLM response if available and not expired."""
    try:
        key = _generate_key(query, context)
        if key in _cache:
            timestamp = _cache_timestamps.get(key)
            if timestamp and datetime.now() - timestamp < timedelta(seconds=CACHE_TTL):
                return _cache[key]
            else:
                # Expired, remove it
                _cache.pop(key, None)
                _cache_timestamps.pop(key, None)
        return None
    except Exception:
        return None


def cache_response(query: str, response: Any, context: dict = None) -> None:
    """Cache LLM response."""
    try:
        # Clean old entries if cache is too large
        if len(_cache) >= MAX_CACHE_SIZE:
            # Remove oldest 20% of entries
            sorted_items = sorted(_cache_timestamps.items(), key=lambda x: x[1])
            to_remove = int(MAX_CACHE_SIZE * 0.2)
            for key, _ in sorted_items[:to_remove]:
                _cache.pop(key, None)
                _cache_timestamps.pop(key, None)
        
        key = _generate_key(query, context)
        _cache[key] = response
        _cache_timestamps[key] = datetime.now()
    except Exception:
        pass


def clear_cache() -> None:
    """Clear all cached responses."""
    global _cache, _cache_timestamps
    _cache.clear()
    _cache_timestamps.clear()


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "size": len(_cache),
        "max_size": MAX_CACHE_SIZE,
        "ttl_seconds": CACHE_TTL
    }

