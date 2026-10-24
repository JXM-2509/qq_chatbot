import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from qq_ollama_bot.runtime.recovery import RecoveryPolicy
from qq_ollama_bot.runtime.supervisor import DiagnosticLog, NapCatLoginProbe, Supervisor, bot_diagnostic


def test_bot_diagnostics_keep_timings_without_content_or_credentials():
    line = ("request_context group_id=123 user_id=456 image_count=1 image_load_ms=33 "
            "model_timeout_seconds=600.0 history_count=2 prompt_fingerprint=secret "
            "token=secret https://private.example.com 聊天正文")
    assert bot_diagnostic(line) == (
        "bot：request_context image_count=1 image_load_ms=33 "
        "model_timeout_seconds=600.0 history_count=2"
    )
    assert bot_diagnostic("接收 <- 群聊 [123] request_completed elapsed_ms=1 聊天正文") is None
    assert bot_diagnostic("random secret https://private.example.com") is None


class Child:
    def __init__(self, alive=True):
        self.running = alive
        self.stopped = False
    def alive(self):
        return self.running
    def stop(self):
        self.stopped = True
        self.running = False


@pytest.fixture
def manager(tmp_path, monkeypatch):
    import qq_ollama_bot.runtime.supervisor as module
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    runtime = {"napcat_dir": str(tmp_path), "webui_port": 6099, "account": "1"}
    obj = Supervisor(runtime, SimpleNamespace(bot_port=8080))
    obj.children = {name: Child() for name in ("ollama", "bot", "napcat")}
    obj.health = lambda: {"ready": True, "onebot_connected": True, "qq_probe_ok": True, "qq_online": True}
    obj.model_health = lambda: (True, True)
    obj.login_probe.snapshot = lambda: None
    obj.started = {name: -1000 for name in obj.children}
    obj.restarted = []
    obj.start_component = lambda name: obj.restarted.append(name)
    obj.notify_login = lambda: None
    yield obj
    obj.close()


def test_busy_model_and_quiet_group_do_not_restart(manager):
    manager.tick(0)
    manager.tick(900)
    assert manager.restarted == []
    assert "运行正常" in manager.status


def test_bot_crash_does_not_restart_qq(manager):
    manager.children["bot"].running = False
    manager.health = lambda: None
    manager.tick(0)
    assert manager.restarted == ["bot"]


def test_silent_offline_waits_for_builtin_recovery(manager):
    manager.health = lambda: {"ready": True, "onebot_connected": True, "qq_probe_ok": True, "qq_online": False}
    manager.tick(0)
    manager.tick(59)
    assert not manager.restarted
    manager.tick(60)
    assert manager.restarted == ["napcat"]


def test_waiting_for_login_keeps_other_components_running(manager):
    manager.health = lambda: {"ready": True, "onebot_connected": False, "qq_online": None}
    manager.login_probe.snapshot = lambda: {"waiting": True, "online": False}
    manager.tick(0)
    manager.tick(5000)
    assert manager.restarted == []
    assert "等待扫码" in manager.status
    manager.health = lambda: {"ready": True, "onebot_connected": True, "qq_probe_ok": True, "qq_online": True}
    manager.login_probe.snapshot = lambda: {"waiting": False, "online": True}
    manager.tick(5001)
    assert not manager.waiting_login and "运行正常" in manager.status


def test_missing_model_does_not_loop_restarts(manager):
    manager.model_health = lambda: (True, False)
    manager.tick(0)
    manager.tick(1000)
    assert not manager.restarted and "模型不存在" in manager.status


def test_port_conflict_stops_startup_without_touching_processes(manager, monkeypatch):
    import qq_ollama_bot.runtime.supervisor as module
    monkeypatch.setattr(module, "port_open", lambda port: port == 8080)
    with pytest.raises(RuntimeError, match="旧机器人"):
        manager.preflight()
    assert all(not child.stopped for child in manager.children.values())


@pytest.mark.parametrize("remote", [
    {"model_available": None, "model_authenticated": False},
    {"model_available": False, "model_authenticated": True},
    {"model_available": None, "model_error": "balance"},
])
def test_cloud_outage_does_not_restart_local_components(manager, remote):
    manager.settings.model_provider = "deepseek"
    manager.model_health = lambda: pytest.fail("Cloud mode must not probe Ollama")
    manager.health = lambda: {"ready": True, "onebot_connected": True,
        "qq_probe_ok": True, "qq_online": True, **remote}
    manager.tick(0)
    manager.tick(1000)
    assert manager.restarted == []
    assert manager.components == ("bot", "napcat")
    assert "模型服务异常" in manager.status


def test_cloud_outage_does_not_block_bot_start(manager):
    manager.settings.model_provider = "deepseek"
    manager.children["bot"].running = False
    manager.started.pop("bot")
    manager.health = lambda: None
    manager.model_health = lambda: pytest.fail("Unexpected local model probe")
    manager.tick(0)
    assert manager.restarted == ["bot"]


def test_cloud_preflight_does_not_probe_local_model_or_port(manager, monkeypatch, tmp_path):
    import qq_ollama_bot.runtime.supervisor as module
    from pydantic import SecretStr
    manager.settings.model_provider = "deepseek"
    manager.settings.onebot_access_token = SecretStr("test-local-token")
    config = tmp_path / "config"
    config.mkdir()
    (config / "onebot11_1.json").write_text(json.dumps({"network": {"websocketClients": [{
        "enable": True, "url": "ws://127.0.0.1:8080/onebot/v11/ws", "token": "test-local-token"}]}}))
    checked_ports = []
    monkeypatch.setattr(module, "port_open", lambda port: checked_ports.append(port) or False)
    manager.model_health = lambda: pytest.fail("Unexpected local model probe")
    manager.preflight()
    assert checked_ports == [8080, 6099]


def test_cloud_runtime_does_not_require_ollama_executable(tmp_path):
    from qq_ollama_bot.runtime.supervisor import load_runtime
    for name in ("napcat.mjs", "NapCatWinBootMain.exe", "NapCatWinBootHook.dll", "qqnt.json", "loadNapCat.js", "qq.exe"):
        (tmp_path / name).touch()
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"napcat_dir": str(tmp_path), "qq_exe": str(tmp_path / "qq.exe"), "account": "1"}))
    assert load_runtime(manifest, model_provider="deepseek")["account"] == "1"


def test_old_worker_log_events_are_ignored(manager):
    manager.generation = 2
    manager.events.put((1, "waiting_login"))
    manager.drain_events()
    assert not manager.waiting_login


def test_login_phase_probe_never_returns_secret_urls(tmp_path):
    config = tmp_path / "webui.json"
    config.write_text(json.dumps({"token": "secret-local-token"}))
    calls = []
    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/auth/login"):
            assert "secret-local-token" not in request.content.decode()
            return httpx.Response(200, json={"code": 0, "data": {"Credential": "private-credential"}})
        return httpx.Response(200, json={"code": 0, "data": {"coreReady": False,
            "loginPhase": "waiting_qrcode", "qrcodeurl": "https://secret-qr.example.com"}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = NapCatLoginProbe(client, config, 6099).snapshot()
    assert result["waiting"]
    assert "secret" not in str(result) and "credential" not in str(result)
    assert len(calls) == 2


def test_log_rollover_preserves_previous_files(tmp_path):
    log = DiagnosticLog(tmp_path, max_bytes=1)
    log.write("运行正常")
    log.write("等待扫码")
    log.close()
    files = list(tmp_path.glob("*.log"))
    assert len(files) == 2
    assert "运行正常" in files[0].read_text(encoding="utf-8") or "运行正常" in files[1].read_text(encoding="utf-8")
