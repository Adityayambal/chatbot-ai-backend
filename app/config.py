from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    router_model: str = "gpt-5.4-mini"
    answer_model: str = "gpt-5.5"
    router_effort: str = "low"
    answer_effort: str = "medium"
    router_max_tokens: int = 4000
    answer_max_tokens: int = 16000

    content_bundle_url: str = "https://api.toppscholars.com/plan-and-promo/api/Plan/content-bundle"
    content_bundle_token: str = ""
    default_plan_ids: str = "6b00418010a08d28bb37e089"
    bundle_cache_ttl_minutes: int = 360
    include_inactive_cps: bool = True
    ingest_ebook_pdfs: bool = False

    retrieval_top_k: int = 10
    max_context_tokens: int = 12000
    max_history_turns: int = 6
    strict_grounding: bool = False

    max_message_chars: int = 1500
    rate_limit_per_minute: int = 10
    cors_origins: str = "http://localhost:5173"
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    data_dir: str = "./data"
    admin_token: str = "change-me"
    debug: bool = True

    @property
    def default_plan_id_list(self) -> list[str]:
        return [p.strip() for p in self.default_plan_ids.split(",") if p.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


# USD per million tokens, OpenAI standard tier (gpt-5.5 at its <272K-context rate), from
# developers.openai.com/api/docs/pricing on 2026-11-21. Verify before relying on these for billing.
PRICES: dict[str, dict[str, float]] = {
    "gpt-5.5": {"input": 5.00, "output": 30.00, "cache_read": 0.50},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50, "cache_read": 0.075},
}
