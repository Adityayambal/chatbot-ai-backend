"""§4.9 failure-matrix coverage for the knowledge base's content-fetch resilience."""
import pytest

from app.content.client import ContentBundleError
from app.content.knowledge_base import KnowledgeBase
from tests.fakes import FIXTURE_PATH, FIXTURE_PLAN_ID


class _FailingClient:
    async def fetch(self, plan_ids: list[str]) -> bytes:
        raise ContentBundleError("simulated content-bundle API outage")


@pytest.mark.asyncio
async def test_fresh_load_reports_ready(settings):
    kb = KnowledgeBase(settings)
    kb._bundle_cache.write(FIXTURE_PLAN_ID, FIXTURE_PATH.read_bytes())  # noqa: SLF001

    await kb.ensure_loaded([FIXTURE_PLAN_ID])

    assert kb.get_plan_status(FIXTURE_PLAN_ID) == "ready"
    assert kb.get_plan(FIXTURE_PLAN_ID) is not None


@pytest.mark.asyncio
async def test_api_down_at_startup_with_no_cache_reports_error(settings):
    kb = KnowledgeBase(settings)
    kb._bundle_client = _FailingClient()  # noqa: SLF001

    await kb.ensure_loaded(["plan-with-no-cache"])

    assert kb.get_plan_status("plan-with-no-cache") == "error"
    assert kb.get_plan("plan-with-no-cache") is None


@pytest.mark.asyncio
async def test_api_down_at_startup_with_cache_serves_stale_and_reports_stale(settings):
    kb = KnowledgeBase(settings)
    kb._bundle_client = _FailingClient()  # noqa: SLF001
    kb._bundle_cache.write(FIXTURE_PLAN_ID, FIXTURE_PATH.read_bytes())  # noqa: SLF001
    # The TTL check only touches the cache's own freshness, not the (failing) client,
    # so is_fresh() being true here would short-circuit before ever calling fetch() —
    # force the "expired cache, API down" path explicitly.
    kb._bundle_cache.is_fresh = lambda *_a, **_kw: False  # noqa: SLF001

    await kb.ensure_loaded([FIXTURE_PLAN_ID])

    assert kb.get_plan_status(FIXTURE_PLAN_ID) == "stale"
    assert kb.get_plan(FIXTURE_PLAN_ID) is not None


@pytest.mark.asyncio
async def test_refresh_failure_after_prior_success_falls_back_to_stale_not_error(settings):
    kb = KnowledgeBase(settings)
    kb._bundle_cache.write(FIXTURE_PLAN_ID, FIXTURE_PATH.read_bytes())  # noqa: SLF001

    await kb.ensure_loaded([FIXTURE_PLAN_ID])
    assert kb.get_plan_status(FIXTURE_PLAN_ID) == "ready"

    # Simulate the cache expiring and the live API being unreachable on refresh.
    kb._bundle_client = _FailingClient()  # noqa: SLF001
    kb._bundle_cache.is_fresh = lambda *_a, **_kw: False  # noqa: SLF001

    await kb.reload([FIXTURE_PLAN_ID])

    assert kb.get_plan_status(FIXTURE_PLAN_ID) == "stale"
    assert kb.get_plan(FIXTURE_PLAN_ID) is not None
