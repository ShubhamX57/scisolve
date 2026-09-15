"""HTTP to the Messages API, and the retry policy.

Raw `requests`, no SDK: it keeps the runtime deps at four and makes the wire
format visible in one file. The retry decision lives in one pure function,
:func:`retry_delay`, so it can be tested without a network or a clock.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

#: One constant. Don't scatter model strings through the codebase.
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 4096

API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

Transport = Callable[[dict], dict]

#: Worth another attempt: transient limits and capacity.
RETRYABLE = frozenset({429, 500, 502, 503, 504, 529})
#: Never worth another attempt. A bad request stays bad.
NEVER_RETRY = frozenset({400, 401, 403, 404, 413})


class TransportError(RuntimeError):
    """Any failure talking to the API."""


class ConnectionFailure(TransportError):
    """The request never produced a response. Usually transient."""


class APIStatusError(TransportError):
    """A non-200 response, with everything support will ask for."""

    def __init__(
        self,
        status: int,
        *,
        message: str = "",
        error_type: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.status = status
        self.error_type = error_type
        self.request_id = request_id
        self.retry_after = retry_after
        detail = f"HTTP {status}"
        if error_type:
            detail += f" {error_type}"
        if message:
            detail += f": {message}"
        if request_id:
            detail += f" (request-id {request_id})"
        super().__init__(detail)

    @property
    def is_spend_cap(self) -> bool:
        """A 429 with no retry-after is the spend-cap flavour: it won't clear."""
        return self.status == 429 and self.retry_after is None


class RetriesExhausted(TransportError):
    """Gave up after repeated retryable failures. Carries the last one."""

    def __init__(self, last: TransportError, attempts: int) -> None:
        self.last = last
        self.attempts = attempts
        hint = ""
        if isinstance(last, APIStatusError) and last.is_spend_cap:
            hint = (
                " This 429 carried no retry-after, which is how a usage-tier spend "
                "cap presents; waiting longer will not clear it."
            )
        super().__init__(f"gave up after {attempts} attempts: {last}{hint}")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    total_wait: float = 30.0
    #: 529 is shared capacity, not your account. Backing off tight makes it
    #: worse for everyone, so start further out than for a 429.
    overload_factor: float = 2.0


DEFAULT_POLICY = RetryPolicy()


def retry_delay(
    error: TransportError,
    *,
    attempt: int,
    waited: float = 0.0,
    policy: RetryPolicy = DEFAULT_POLICY,
    jitter: Callable[[], float] = random.random,
) -> float | None:
    """Seconds to wait before attempt ``attempt + 1``, or None to give up.

    ``attempt`` is how many attempts have already been made (1 after the first
    failure). ``waited`` is the total slept so far, which caps runaway backoff
    independently of the attempt count.
    """
    if attempt >= policy.max_attempts:
        return None

    status = getattr(error, "status", None)
    if status is not None and (status in NEVER_RETRY or status not in RETRYABLE):
        return None

    retry_after = getattr(error, "retry_after", None)
    if retry_after is not None:
        delay = float(retry_after)
    else:
        scale = policy.overload_factor if status == 529 else 1.0
        delay = min(policy.base_delay * scale * 2 ** (attempt - 1), policy.max_delay)
        delay *= 0.5 + jitter()  # full-ish jitter: 0.5x to 1.5x

    if waited + delay > policy.total_wait:
        return None
    return delay


def with_retries(
    send: Transport,
    *,
    policy: RetryPolicy = DEFAULT_POLICY,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> Transport:
    """Wrap a transport so retryable failures are retried per ``policy``."""

    def call(payload: dict) -> dict:
        attempt = 0
        waited = 0.0
        while True:
            try:
                return send(payload)
            except TransportError as exc:
                attempt += 1
                delay = retry_delay(
                    exc, attempt=attempt, waited=waited, policy=policy, jitter=jitter
                )
                if delay is None:
                    if attempt > 1 or _is_retryable(exc):
                        raise RetriesExhausted(exc, attempt) from exc
                    raise
                sleep(delay)
                waited += delay

    return call


def _is_retryable(exc: TransportError) -> bool:
    status = getattr(exc, "status", None)
    return status is None or status in RETRYABLE


class HTTPTransport:
    """POST to the Messages API. Raises; retries are someone else's job."""

    def __init__(
        self,
        *,
        api_key: str,
        url: str = API_URL,
        timeout: float = 120.0,
        session: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.timeout = timeout
        self._session = session or requests.Session()

    def __call__(self, payload: dict) -> dict:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        try:
            response = self._session.post(
                self.url, headers=headers, json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise ConnectionFailure(str(exc)) from exc

        request_id = response.headers.get("request-id")
        if response.status_code == 200:
            return response.json()

        raise APIStatusError(
            response.status_code,
            message=_error_message(response),
            error_type=_error_type(response),
            request_id=request_id,
            retry_after=_retry_after(response),
        )


def _body(response: Any) -> dict:
    try:
        parsed = response.json()
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _error_message(response: Any) -> str:
    error = _body(response).get("error")
    if isinstance(error, dict):
        return str(error.get("message", ""))
    return (getattr(response, "text", "") or "")[:200]


def _error_type(response: Any) -> str | None:
    error = _body(response).get("error")
    if isinstance(error, dict) and error.get("type"):
        return str(error["type"])
    return None


def _retry_after(response: Any) -> float | None:
    """Seconds from the retry-after header. HTTP-date form is ignored."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
