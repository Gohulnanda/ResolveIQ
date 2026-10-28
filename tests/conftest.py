"""Shared fixtures. Tests never touch the network or require an API key."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models.schemas import TicketRequest
from app.services.container import build_container


def make_settings(tmp_path_factory: pytest.TempPathFactory, **overrides) -> Settings:
    defaults = dict(
        model_path=tmp_path_factory.mktemp("model") / "classifier.joblib",
        llm_provider="demo",
        embedding_backend="hybrid",
    )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # ignore any developer .env file


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return make_settings(tmp_path_factory)


@pytest.fixture(scope="session")
def container(settings: Settings):
    return build_container(settings)


@pytest.fixture(scope="session")
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def vpn_ticket() -> TicketRequest:
    return TicketRequest(
        title="Cannot connect to VPN from home",
        description="The VPN client times out after I enter my credentials. Home internet works.",
    )
