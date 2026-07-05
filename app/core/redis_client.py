"""
Redis client singleton for refresh tokens, caching, and Celery broker access.
"""
import redis
from app.core.config import get_settings

_redis_client = None


def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
    return _redis_client


def store_refresh_token(jti: str, user_id: int, expires_seconds: int):
    """Store a refresh token JTI in Redis with TTL."""
    client = get_redis_client()
    client.setex(f"refresh:{jti}", expires_seconds, str(user_id))


def validate_refresh_token(jti: str) -> str | None:
    """Return user_id if refresh token is valid, None otherwise."""
    client = get_redis_client()
    return client.get(f"refresh:{jti}")


def revoke_refresh_token(jti: str):
    """Delete a refresh token from Redis (logout)."""
    client = get_redis_client()
    client.delete(f"refresh:{jti}")


def blacklist_access_token(jti: str, expires_seconds: int):
    """Blacklist an access token JTI until it expires naturally."""
    client = get_redis_client()
    client.setex(f"blacklist:{jti}", expires_seconds, "1")


def is_token_blacklisted(jti: str) -> bool:
    """Check if an access token has been revoked."""
    client = get_redis_client()
    return client.exists(f"blacklist:{jti}") > 0
