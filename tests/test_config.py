from pathlib import Path

import pytest
from pydantic import ValidationError

from qq_ollama_bot.config import Settings


def test_settings_parse_group_ids() -> None:
    settings = Settings(
        _env_file=None,
        onebot_access_token="0123456789abcdef",
        allowed_group_ids="123, 456,123",
        database_path=Path("test.db"),
    )
    assert settings.allowed_group_id_set == frozenset({123, 456})
    assert settings.profile_message_limit == 50
    assert settings.profile_character_budget == 6_000
    assert settings.profile_target_limit == 5
    assert settings.forward_max_depth == 3
    assert settings.forward_message_limit == 100
    assert settings.forward_character_budget == 6_000
    assert settings.forward_timeout_seconds == 30
    assert settings.forward_model_timeout_seconds == 600
    assert settings.image_model_timeout_seconds == 600
    assert settings.forward_queue_size == 20


def test_settings_reject_non_local_listener() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            onebot_access_token="0123456789abcdef",
            allowed_group_ids="123",
            bot_host="0.0.0.0",
        )


@pytest.mark.parametrize("options", [
    {"model_provider": "unknown"},
    {"model_provider": "deepseek"},
    {"model_provider": "deepseek", "deepseek_api_key": "   "},
    {"deepseek_base_url": "http://api.deepseek.com"},
    {"deepseek_base_url": "https://user:secret@api.deepseek.com"},
    {"deepseek_base_url": "https://api.deepseek.com?key=secret"},
    {"deepseek_max_tokens": 0},
])
def test_invalid_provider_configuration(options):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, onebot_access_token="0123456789abcdef", allowed_group_ids="1", **options)


def test_legacy_defaults_and_cloud_secret_masking():
    options = dict(_env_file=None, onebot_access_token="0123456789abcdef", allowed_group_ids="1")
    assert Settings(**options).model_provider == "ollama"
    cloud = Settings(**options, model_provider="deepseek", deepseek_api_key="private-api-key")
    assert "private-api-key" not in repr(cloud)
    assert cloud.deepseek_model == "deepseek-flash"
