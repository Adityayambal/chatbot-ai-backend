"""ContentBundleClient.fetch() against a mocked HTTP transport (respx) — no real network."""
import httpx
import orjson
import pytest
import respx

from app.content.client import ContentBundleClient, ContentBundleError

URL = "https://api.toppscholars.com/plan-and-promo/api/Plan/content-bundle"

SUCCESS_BODY = {"statusCode": 200, "isSuccess": True, "message": "Success", "result": []}


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def _instant_sleep(_seconds):
        return None

    monkeypatch.setattr("app.content.client.asyncio.sleep", _instant_sleep)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_succeeds_on_first_try():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=SUCCESS_BODY))
    client = ContentBundleClient(URL)

    body = await client.fetch(["plan-1"])

    assert orjson.loads(body) == SUCCESS_BODY
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_fetch_sends_bearer_token_when_configured():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=SUCCESS_BODY))
    client = ContentBundleClient(URL, token="secret-token")

    await client.fetch(["plan-1"])

    assert route.calls.last.request.headers["authorization"] == "Bearer secret-token"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_retries_on_5xx_then_succeeds():
    route = respx.post(URL).mock(
        side_effect=[httpx.Response(500), httpx.Response(200, json=SUCCESS_BODY)]
    )
    client = ContentBundleClient(URL, max_retries=3)

    body = await client.fetch(["plan-1"])

    assert orjson.loads(body) == SUCCESS_BODY
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_fetch_raises_after_exhausting_retries_on_persistent_5xx():
    route = respx.post(URL).mock(return_value=httpx.Response(503))
    client = ContentBundleClient(URL, max_retries=3)

    with pytest.raises(ContentBundleError):
        await client.fetch(["plan-1"])

    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_fetch_raises_on_is_success_false():
    route = respx.post(URL).mock(
        return_value=httpx.Response(
            200, json={"statusCode": 500, "isSuccess": False, "message": "boom", "result": []}
        )
    )
    client = ContentBundleClient(URL, max_retries=1)

    with pytest.raises(ContentBundleError):
        await client.fetch(["plan-1"])

    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_fetch_retries_on_timeout():
    route = respx.post(URL).mock(
        side_effect=[httpx.TimeoutException("timed out"), httpx.Response(200, json=SUCCESS_BODY)]
    )
    client = ContentBundleClient(URL, max_retries=3)

    body = await client.fetch(["plan-1"])

    assert orjson.loads(body) == SUCCESS_BODY
    assert route.call_count == 2
