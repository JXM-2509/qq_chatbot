from pathlib import Path

import pytest
from pydantic import ValidationError

from qq_ollama_bot.config import Settings


def test_settings_parse_group_ids() -> None:
    settings = Settings(
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
    assert settings.forward_queue_size == 20


def test_settings_reject_non_local_listener() -> None:
    with pytest.raises(ValidationError):
        Settings(
            onebot_access_token="0123456789abcdef",
            allowed_group_ids="123",
            bot_host="0.0.0.0",
        )
