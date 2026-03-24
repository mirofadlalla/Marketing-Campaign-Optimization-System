import time
import logging
from fastapi import HTTPException, status
import redis

WINDOW  = 60
LIMIT = 100

try:
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True, socket_connect_timeout=2)
    redis_client.ping()
except Exception as e:
    logging.error("Redis Not Being Working")
    raise e


class RateLimiter:
    def __init__(self, user_id: str):
        self.user_id = user_id

    def _get_key(self):
        now = int(time.time())
        return f"rate:{self.user_id}:{now // WINDOW}"

    def check(self):
        try:
            key = self._get_key()

            current = redis_client.incr(key)

            if current == 1:
                redis_client.expire(key, WINDOW)

            if current > LIMIT:
                self._log_violation(current)
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded"
                )

            return current

        except redis.RedisError:
            logging.error("Redis error - skipping rate limit")
            return 0  # fallback: allow request

    def remaining(self):
        try:
            key = self._get_key()
            current = redis_client.get(key)

            if current is None:
                return LIMIT

            return max(LIMIT - int(current), 0)

        except redis.RedisError:
            return LIMIT

    def reset(self):
        try:
            key = self._get_key()
            redis_client.delete(key)
            return True
        except redis.RedisError:
            return False

    def _log_violation(self, current):
        try:
            now = int(time.time())
            key = f"violation:{self.user_id}:{now // WINDOW}"

            redis_client.incr(key)
            redis_client.expire(key, WINDOW * 5)

            logging.warning(
                f"Rate limit exceeded - user={self.user_id}, count={current}"
            )

        except Exception as e:
            logging.error(f"Logging error: {e}")