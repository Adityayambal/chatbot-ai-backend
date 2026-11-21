"""Fetches the content-bundle API. Server-side only — never called from the browser."""
from __future__ import annotations

import asyncio
import logging

import httpx
import orjson

logger = logging.getLogger(__name__)


class ContentBundleError(RuntimeError):
    pass


class ContentBundleClient:
    def __init__(self, url: str, token: str = "", *, max_retries: int = 3) -> None:
        self._url = url
        self._token = token
        self._max_retries = max_retries

    async def fetch(self, plan_ids: list[str]) -> bytes:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        payload = {"planIds": plan_ids}

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10)) as client:
            for attempt in range(self._max_retries):
                try:
                    response = await client.post(self._url, json=payload, headers=headers)
                    if response.status_code >= 500:
                        last_error = ContentBundleError(
                            f"content-bundle API returned {response.status_code}"
                        )
                        raise last_error
                    response.raise_for_status()
                    body = response.content
                    data = orjson.loads(body)
                    if not data.get("isSuccess") or data.get("statusCode") != 200:
                        raise ContentBundleError(
                            f"content-bundle API reported failure: {data.get('message')}"
                        )
                    return body
                except (httpx.TimeoutException, ContentBundleError, httpx.HTTPStatusError) as exc:
                    last_error = exc
                    if attempt < self._max_retries - 1:
                        backoff = 2**attempt
                        logger.warning(
                            "content-bundle fetch attempt %d failed: %s; retrying in %ds",
                            attempt + 1,
                            exc,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                    else:
                        break
        assert last_error is not None
        raise ContentBundleError(str(last_error)) from last_error
