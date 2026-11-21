"""API tests for catalog/outline (sessions endpoints are added in M2).

Builds a bare FastAPI app with just the catalog router + the fixture KB, instead of
app.main.create_app()'s full lifespan — that lifespan fetches the real content-bundle
API in the background, which tests must never do.
"""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.api import catalog as catalog_routes
from app.api.deps import get_kb
from tests.fakes import FIXTURE_PLAN_ID


def make_test_app(fixture_kb) -> FastAPI:
    app = FastAPI()
    app.include_router(catalog_routes.router, prefix="/api")
    app.dependency_overrides[get_kb] = lambda: fixture_kb

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            return JSONResponse(status_code=exc.status_code, content=detail)
        return JSONResponse(
            status_code=exc.status_code, content={"error": {"code": "error", "message": str(detail)}}
        )

    return app


@pytest.fixture
def client(fixture_kb) -> TestClient:
    return TestClient(make_test_app(fixture_kb))


def test_catalog_returns_options_for_plan(client: TestClient):
    resp = client.get(f"/api/catalog?planIds={FIXTURE_PLAN_ID}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["plans"][0]["planId"] == FIXTURE_PLAN_ID
    assert len(data["options"]) == 1
    option = data["options"][0]
    assert option["subjectName"] == "Science & Technology"
    assert option["mediumName"] == "English"
    assert option["chapterCount"] == 2
    assert option["conceptCount"] == 3
    assert option["isActive"] is False


def test_outline_returns_chapter_tree(client: TestClient):
    cp_id = "cp0000000000000000000001"
    resp = client.get(f"/api/subjects/{cp_id}/outline?planId={FIXTURE_PLAN_ID}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cpId"] == cp_id
    assert len(data["chapters"]) == 2
    first_chapter = data["chapters"][0]
    assert first_chapter["name"] == "Laws of Motion"
    assert first_chapter["counts"]["notes"] >= 1
    assert first_chapter["concepts"][0]["subconcepts"][0]["code"] == "1.1.1"


def test_outline_404_for_unknown_cp(client: TestClient):
    resp = client.get(f"/api/subjects/does-not-exist/outline?planId={FIXTURE_PLAN_ID}")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "cp_not_found"


class _AlwaysLoadingKB:
    """Stands in for a KnowledgeBase whose plan never finishes loading — no network calls."""

    def known_plan_ids(self) -> set[str]:
        return {"still-loading-plan"}

    def ensure_loading_started(self, plan_id: str) -> None:
        pass

    def get_plan_status(self, plan_id: str) -> str:
        return "loading"

    def get_plan(self, plan_id: str):
        return None

    def maybe_schedule_refresh(self, plan_id: str) -> None:
        pass


def test_catalog_returns_503_content_loading_when_plan_never_becomes_ready(monkeypatch):
    from app.api import deps as deps_module

    monkeypatch.setattr(deps_module, "PLAN_WAIT_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(deps_module, "PLAN_WAIT_POLL_SECONDS", 0.05)

    app = make_test_app(_AlwaysLoadingKB())
    client = TestClient(app)
    resp = client.get("/api/catalog?planIds=still-loading-plan")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "content_loading"
