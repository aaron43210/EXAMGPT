"""
In-process sliding-window rate limiter. Sufficient for a single-replica deployment
(HF Spaces); swap for Redis if you ever run more than one instance.
"""
import threading
import time
from collections import defaultdict, deque
from fastapi import HTTPException, Request


class RateLimiter:
    SWEEP_EVERY = 1000
    IDLE_SECONDS = 300

    def __init__(self):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._calls = 0

    def check(self, key: str, limit: int, window_seconds: int = 60) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            self._calls += 1
            if self._calls % self.SWEEP_EVERY == 0:
                # Forget idle keys so attacker-chosen keys (usernames, IPs) can't grow memory forever
                idle = [k for k, d in self._hits.items() if not d or d[-1] < now - self.IDLE_SECONDS]
                for k in idle:
                    del self._hits[k]
            hits = self._hits[key]
            while hits and hits[0] <= now - window_seconds:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = int(window_seconds - (now - hits[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please slow down and try again shortly.",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._calls = 0

    def tracked_keys(self) -> int:
        return len(self._hits)


limiter = RateLimiter()


def client_ip(request: Request) -> str:
    # HF Spaces sits behind a proxy; the left-most forwarded address is the client
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
