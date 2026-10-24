import os
import warnings
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from qq_ollama_bot.config import get_settings
from qq_ollama_bot.runtime import credentials


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("ONEBOT_ACCESS_TOKEN", "startup-test-token-0123456789")
    monkeypatch.setenv("ALLOWED_GROUP_IDS", "1")
    # Record the original state even when the key was absent: startup writes it
    # directly to os.environ, and teardown must remove that process-local value.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.setattr(credentials.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def forbid_prompt(*args, **kwargs):
    pytest.fail("An existing credential or noninteractive startup must not prompt")


def test_prompted_key_is_inherited_without_file_changes(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_PROVIDER=deepseek\n", encoding="utf-8")
    before = env_file.read_bytes()
    prompts = []

    def prompt(message):
        prompts.append(message)
        return "ephemeral-test-key"

    monkeypatch.setattr(credentials.getpass, "getpass", prompt)
    first = credentials.load_startup_settings()
    second = credentials.load_startup_settings()
    assert first is second
    assert first.deepseek_api_key.get_secret_value() == "ephemeral-test-key"
    assert os.environ["DEEPSEEK_API_KEY"] == "ephemeral-test-key"
    assert len(prompts) == 1
    assert env_file.read_bytes() == before
    assert list(tmp_path.iterdir()) == [env_file]
    captured = capsys.readouterr()
    assert "ephemeral-test-key" not in captured.out + captured.err


@pytest.mark.parametrize("source", ["environment", "dotenv"])
def test_configured_key_skips_prompt(source, tmp_path, monkeypatch):
    if source == "environment":
        monkeypatch.setenv("DEEPSEEK_API_KEY", "configured-test-key")
    else:
        (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=configured-test-key\n", encoding="utf-8")
    monkeypatch.setattr(credentials.getpass, "getpass", forbid_prompt)
    assert credentials.load_startup_settings().deepseek_api_key.get_secret_value() == "configured-test-key"


@pytest.mark.parametrize("interactive,tty", [(False, True), (True, False)])
def test_check_and_redirected_input_fail_without_prompt(interactive, tty, monkeypatch):
    monkeypatch.setattr(credentials.sys, "stdin", SimpleNamespace(isatty=lambda: tty))
    monkeypatch.setattr(credentials.getpass, "getpass", forbid_prompt)
    with pytest.raises(RuntimeError, match="缺少 DEEPSEEK_API_KEY"):
        credentials.load_startup_settings(interactive=interactive)
    assert "DEEPSEEK_API_KEY" not in os.environ


def test_empty_input_does_not_start_or_save_key(monkeypatch):
    monkeypatch.setattr(credentials.getpass, "getpass", lambda prompt: "  ")
    with pytest.raises(RuntimeError, match="未输入"):
        credentials.load_startup_settings()
    assert "DEEPSEEK_API_KEY" not in os.environ


def test_unavailable_secure_input_never_falls_back_to_echo(monkeypatch):
    def insecure_prompt(prompt):
        warnings.warn("Cannot control echo", credentials.getpass.GetPassWarning)
        pytest.fail("Echoed fallback must not be reached")

    monkeypatch.setattr(credentials.getpass, "getpass", insecure_prompt)
    with pytest.raises(RuntimeError, match="无法隐藏输入"):
        credentials.load_startup_settings()
    assert "DEEPSEEK_API_KEY" not in os.environ


def test_ollama_does_not_require_api_key(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setattr(credentials.getpass, "getpass", forbid_prompt)
    assert credentials.load_startup_settings().model_provider == "ollama"


def test_other_configuration_errors_do_not_prompt(monkeypatch):
    monkeypatch.setenv("BOT_PORT", "invalid")
    monkeypatch.setattr(credentials.getpass, "getpass", forbid_prompt)
    with pytest.raises(ValidationError):
        credentials.load_startup_settings()
