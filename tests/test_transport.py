"""Transport tests: the retry decision and the HTTP error shape. No network."""

from __future__ import annotations

import json

import pytest
import requests

from scisolve.transport import (
    APIStatusError,
    ConnectionFailure,
    HTTPTransport,
    RetriesExhausted,
    RetryPolicy,
    retry_delay,
    with_retries,
)

NO_JITTER = lambda: 0.5  # noqa: E731 -- makes delays deterministic (scale 1.0)


def status_error(status: int, **kw) -> APIStatusError:
    return APIStatusError(status, message="scripted", **kw)


# --- the retry decision ---------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
def test_retryable_statuses_get_a_delay(status):
    assert retry_delay(status_error(status), attempt=1, jitter=NO_JITTER) is not None


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
def test_never_retried_statuses_give_up_immediately(status):
    assert retry_delay(status_error(status), attempt=1, jitter=NO_JITTER) is None


def test_retry_after_is_honoured_exactly():
    delay = retry_delay(status_error(429, retry_after=7.0), attempt=1, jitter=NO_JITTER)
    assert delay == 7.0


def test_backoff_is_exponential_without_a_retry_after_header():
    first = retry_delay(status_error(500), attempt=1, jitter=NO_JITTER)
    second = retry_delay(status_error(500), attempt=2, jitter=NO_JITTER)
    third = retry_delay(status_error(500), attempt=3, jitter=NO_JITTER)
    assert (first, second, third) == (0.5, 1.0, 2.0)


def test_529_backs_off_further_than_a_429():
    overloaded = retry_delay(status_error(529), attempt=1, jitter=NO_JITTER)
    rate_limited = retry_delay(status_error(429), attempt=1, jitter=NO_JITTER)
    assert overloaded > rate_limited


def test_jitter_spreads_the_delay():
    low = retry_delay(status_error(500), attempt=1, jitter=lambda: 0.0)
    high = retry_delay(status_error(500), attempt=1, jitter=lambda: 0.99)
    assert low < high
    assert 0.25 <= low and high <= 0.75


def test_delay_is_capped_by_max_delay():
    policy = RetryPolicy(max_attempts=20, base_delay=1.0, max_delay=3.0, total_wait=1e9)
    delay = retry_delay(status_error(500), attempt=10, policy=policy, jitter=NO_JITTER)
    assert delay == 3.0


def test_attempts_are_capped():
    policy = RetryPolicy(max_attempts=3)
    assert retry_delay(status_error(500), attempt=3, policy=policy, jitter=NO_JITTER) is None


def test_total_wait_is_capped_even_when_attempts_remain():
    policy = RetryPolicy(max_attempts=10, total_wait=5.0)
    assert (
        retry_delay(status_error(429, retry_after=4.0), attempt=1, waited=4.0, policy=policy)
        is None
    )


def test_connection_failure_has_no_status_and_is_retried():
    assert retry_delay(ConnectionFailure("dns went away"), attempt=1, jitter=NO_JITTER) is not None


def test_spend_cap_429_is_distinguishable():
    assert status_error(429).is_spend_cap is True
    assert status_error(429, retry_after=2.0).is_spend_cap is False
    assert status_error(500).is_spend_cap is False


# --- the wrapper ----------------------------------------------------------


def test_with_retries_stops_and_names_the_spend_cap():
    def always_429(payload):
        raise status_error(429)

    with pytest.raises(RetriesExhausted) as caught:
        with_retries(always_429, policy=RetryPolicy(max_attempts=2), sleep=lambda s: None)({})

    assert "spend cap" in str(caught.value)
    assert caught.value.attempts == 2


def test_with_retries_reraises_a_400_untouched():
    def always_400(payload):
        raise status_error(400, error_type="invalid_request_error")

    with pytest.raises(APIStatusError) as caught:
        with_retries(always_400, sleep=lambda s: None)({})

    assert not isinstance(caught.value, RetriesExhausted)
    assert caught.value.status == 400


def test_with_retries_returns_the_first_success():
    calls = []

    def flaky(payload):
        calls.append(payload)
        if len(calls) < 3:
            raise status_error(529)
        return {"content": []}

    result = with_retries(flaky, sleep=lambda s: None, jitter=NO_JITTER)({"n": 1})
    assert result == {"content": []}
    assert len(calls) == 3


# --- HTTP plumbing --------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = json.dumps(body) if isinstance(body, dict) else str(body)

    def json(self):
        if isinstance(self._body, dict):
            return self._body
        raise ValueError("not json")


class FakeSession:
    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.calls = []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        if self.raises:
            raise self.raises
        return self.response


def test_successful_response_is_returned_as_json():
    session = FakeSession(FakeResponse(200, {"content": [{"type": "text"}]}))
    transport = HTTPTransport(api_key="sk-test", session=session)

    assert transport({"model": "m"}) == {"content": [{"type": "text"}]}
    url, kw = session.calls[0]
    assert kw["headers"]["x-api-key"] == "sk-test"
    assert kw["headers"]["anthropic-version"] == "2023-06-01"
    assert kw["json"] == {"model": "m"}


def test_error_body_and_request_id_are_surfaced():
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    session = FakeSession(
        FakeResponse(429, body, headers={"request-id": "req_abc", "retry-after": "12"})
    )
    transport = HTTPTransport(api_key="sk-test", session=session)

    with pytest.raises(APIStatusError) as caught:
        transport({})

    error = caught.value
    assert error.status == 429
    assert error.error_type == "rate_limit_error"
    assert error.request_id == "req_abc"
    assert error.retry_after == 12.0
    assert "req_abc" in str(error)
    assert error.is_spend_cap is False


def test_http_date_retry_after_is_ignored_rather_than_crashing():
    session = FakeSession(
        FakeResponse(429, {}, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    )
    transport = HTTPTransport(api_key="sk-test", session=session)

    with pytest.raises(APIStatusError) as caught:
        transport({})
    assert caught.value.retry_after is None


def test_connection_error_becomes_connection_failure():
    session = FakeSession(raises=requests.ConnectionError("no route to host"))
    transport = HTTPTransport(api_key="sk-test", session=session)

    with pytest.raises(ConnectionFailure, match="no route to host"):
        transport({})
