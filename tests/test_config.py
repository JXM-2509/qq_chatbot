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


def test_settings_reject_non_local_listener() -> None:
    with pytest.raises(ValidationError):
        Settings(
            onebot_access_token="0123456789abcdef",
            allowed_group_ids="123",
            bot_host="0.0.0.0",
        )
