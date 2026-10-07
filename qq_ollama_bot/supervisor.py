"""Foreground project manager. No service installation, power changes or auto-updates."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import socket
import sys
import threading
import time

import httpx

from qq_ollama_bot.config import get_settings
from qq_ollama_bot.recovery import RecoveryPolicy, login_event

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT.parent / "runtime" / "project-runtime.json"


def bot_diagnostic(line: str) -> str | None:
    """Keep event names and numeric timings only; discard all identity/content fields."""
    if any(marker in line for marker in ("接收 <-", "发送 ->", "群聊 [", "私聊 [")):
        return None
    event = re.search(r"\b(request_context|request_completed|ollama_request_timeout|ollama_request_failed|image_load_failed)\b", line)
    if event is None:
        return None
    fields = re.findall(r"\b(image_count|image_load_ms|model_timeout_seconds|model_elapsed_ms|elapsed_ms|history_count)=([0-9]{1,12}(?:\.[0-9]{1,3})?)\b", line)
    return "bot：" + event[1] + " " + " ".join(f"{key}={value}" for key, value in fields)


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def load_runtime(path: Path = MANIFEST) -> dict:
    runtime = json.loads(path.read_text(encoding="utf-8-sig"))
    for field in ("napcat_dir", "qq_exe", "ollama_exe"):
        target = Path(runtime[field]).resolve()
        if not target.exists():
            raise RuntimeError(f"运行文件缺失：{field}。请重新运行部署工具。")
        runtime[field] = str(target)
    if not str(runtime["account"]).isdigit():
        raise RuntimeError("机器人账号配置无效。")
    napcat = Path(runtime["napcat_dir"])
    for name in ("napcat.mjs", "NapCatWinBootMain.exe", "NapCatWinBootHook.dll", "qqnt.json", "loadNapCat.js"):
        if not (napcat / name).is_file():
            raise RuntimeError(f"NapCat 运行文件缺失：{name}")
    for name, expected in runtime.get("hashes", {}).items():
        with Path(name).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                raise RuntimeError("运行文件校验失败，请检查固定版本部署。")
    return runtime


class NapCatLoginProbe:
    """Read login phase using existing credentials, even before OneBot is connected."""
    def __init__(self, client: httpx.Client, config: Path, port: int):
        self.client, self.config = client, config
        self.base = f"http://127.0.0.1:{port}/api"
        self.credential = None
        self.auth_after = 0.0

    def snapshot(self) -> dict | None:
        try:
            now = time.monotonic()
            if self.credential is None:
                if now < self.auth_after:
                    return None
                self.auth_after = now + 60
                token = json.loads(self.config.read_text(encoding="utf-8-sig")).get("token", "")
                digest = hashlib.sha256((token + ".napcat").encode()).hexdigest()
                reply = self.client.post(f"{self.base}/auth/login", json={"hash": digest}, timeout=3).json()
                self.credential = reply.get("data", {}).get("Credential")
                if not self.credential:
                    return None
            reply = self.client.post(f"{self.base}/QQLogin/CheckLoginStatus", json={},
                                     headers={"Authorization": f"Bearer {self.credential}"}, timeout=3).json()
            if reply.get("code") != 0:
                self.credential = None
                return None
            data = reply.get("data", {})
            # Never retain QR URLs or login error strings in state or diagnostics.
            return {"online": data.get("coreReady") is True,
                    "waiting": (bool(data.get("qrcodeurl")) and data.get("loginPhase") == "waiting_qrcode")
                    or data.get("loginPhase") in {"qrcode_scanned", "waiting_confirm", "captcha", "device_verify"},
                    "phase": data.get("loginPhase")}
        except (OSError, httpx.HTTPError, ValueError, TypeError, AttributeError):
            return None

    def reset(self):
        self.credential = None
        self.auth_after = 0


class DiagnosticLog:
    """Only fixed diagnostic events enter here; raw child output is never persisted."""
    def __init__(self, root: Path, max_bytes: int = 2 * 1024 * 1024):
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.max_bytes, self.index = root, max_bytes, 0
        self.stem = f"supervisor-{datetime.now():%Y%m%d-%H%M%S-%f}-{os.getpid()}"
        self.stream = None

    def write(self, message: str):
        if self.stream is None or self.stream.tell() >= self.max_bytes:
            if self.stream is not None:
                self.stream.close()
            self.index += 1
            self.stream = (self.root / f"{self.stem}-{self.index:03}.log").open("x", encoding="utf-8")
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
        self.stream.write(line + "\n")
        self.stream.flush()
        print(line, flush=True)

    def close(self):
        if self.stream is not None:
            self.stream.close()


class Supervisor:
    def __init__(self, runtime: dict, settings, process_factory=None):
        if process_factory is None:
            from qq_ollama_bot.windows_process import OwnedProcess
            process_factory = OwnedProcess
        self.runtime, self.settings, self.process_factory = runtime, settings, process_factory
        self.client = httpx.Client(timeout=8, trust_env=False)
        self.login_probe = NapCatLoginProbe(self.client, Path(runtime["napcat_dir"]) / "config" / "webui.json",
                                           runtime["webui_port"])
        self.children: dict = {}
        self.policies = {name: RecoveryPolicy() for name in ("ollama", "bot", "napcat")}
        self.events: queue.Queue[tuple[int, str]] = queue.Queue()
        self.diagnostics: queue.Queue[str] = queue.Queue(maxsize=1024)
        self.generation = 0
        self.waiting_login = False
        self.notification_sent = False
        self.shared_ollama = False
        self.status = ""
        self.started: dict[str, float] = {}
        self.log = DiagnosticLog(PROJECT_ROOT / "logs")
        self.stopping = False

    def health(self) -> dict | None:
        try:
            response = self.client.get(f"http://127.0.0.1:{self.settings.bot_port}/healthz")
            response.raise_for_status()
            health = response.json()
            if health.get("service") != "qq-ollama-bot":
                return None
            return health
        except (httpx.HTTPError, ValueError, AttributeError):
            return None

    def model_health(self) -> tuple[bool, bool | None]:
        base = self.settings.ollama_base_url
        try:
            response = self.client.get(f"{base}/api/version", timeout=3)
            response.raise_for_status()
            if not response.json().get("version"):
                return False, None
        except (httpx.HTTPError, ValueError, AttributeError):
            return False, None
        try:
            response = self.client.get(f"{base}/api/tags", timeout=3)
            response.raise_for_status()
            names = {item.get("name") for item in response.json()["models"]}
            model = self.settings.ollama_model
            return True, model in names or (":" not in model and f"{model}:latest" in names)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            return True, None

    def preflight(self):
        for port, label in ((self.settings.bot_port, "旧机器人"), (self.runtime["webui_port"], "旧 NapCat")):
            if port_open(port):
                raise RuntimeError(f"{label}占用端口 {port}，请先关闭后重新启动。不会自动结束外部进程。")
        reachable, model = self.model_health()
        if reachable and model is False:
            raise RuntimeError("配置的 Ollama 模型尚未安装，请先安装模型。")
        if not reachable and port_open(11434):
            raise RuntimeError("Ollama 端口被无响应的外部进程占用，请人工检查。")
        # Verify routing and credentials without displaying or changing them.
        import hmac
        cfg = Path(self.runtime["napcat_dir"]) / "config" / f"onebot11_{self.runtime['account']}.json"
        clients = json.loads(cfg.read_text(encoding="utf-8-sig"))["network"]["websocketClients"]
        expected = f"ws://127.0.0.1:{self.settings.bot_port}/onebot/v11/ws"
        if not any(item.get("enable") and item.get("url") == expected and
                   hmac.compare_digest(str(item.get("token", "")), self.settings.onebot_access_token.get_secret_value())
                   for item in clients):
            raise RuntimeError("NapCat 连接配置与机器人不一致，请检查本地配置。")

    def start_component(self, name: str):
        child = self.children.pop(name, None)
        if child is not None:
            child.stop()
        port = {"bot": self.settings.bot_port, "napcat": self.runtime["webui_port"], "ollama": 11434}[name]
        if port_open(port):
            if name == "ollama" and self.model_health()[0]:
                self.shared_ollama = True
                self.started[name] = time.monotonic()
                self.log.write("ollama：复用已有服务")
                return
            raise RuntimeError(f"{name}：端口被外部进程占用，暂停自动恢复。")
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
        cwd = PROJECT_ROOT
        if name == "bot":
            argv = [str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "qq_ollama_bot"]
        elif name == "ollama":
            argv = [self.runtime["ollama_exe"], "serve"]
            env["OLLAMA_HOST"] = "127.0.0.1:11434"
            self.shared_ollama = False
        else:
            cwd = Path(self.runtime["napcat_dir"])
            env.update({"NAPCAT_PATCH_PACKAGE": str(cwd / "qqnt.json"),
                        "NAPCAT_LOAD_PATH": str(cwd / "loadNapCat.js"),
                        "NAPCAT_INJECT_PATH": str(cwd / "NapCatWinBootHook.dll"),
                        "NAPCAT_MAIN_PATH": str(cwd / "napcat.mjs")})
            argv = [str(cwd / "NapCatWinBootMain.exe"), self.runtime["qq_exe"],
                    str(cwd / "NapCatWinBootHook.dll"), "-q", str(self.runtime["account"])]
            self.generation += 1
            self.waiting_login = self.notification_sent = False
            self.login_probe.reset()
        generation = self.generation

        def on_line(line):
            if name == "napcat" and (event := login_event(line)):
                self.events.put((generation, event))
            elif name == "bot" and (diagnostic := bot_diagnostic(line)):
                try:
                    self.diagnostics.put_nowait(diagnostic)
                except queue.Full:
                    pass

        self.children[name] = self.process_factory(argv, cwd, env, on_line)
        self.started[name] = time.monotonic()
        self.log.write(f"{name}：已启动项目进程")

    def alive(self, name: str) -> bool:
        child = self.children.get(name)
        return child is not None and child.alive()

    def drain_events(self):
        while not self.diagnostics.empty():
            self.log.write(self.diagnostics.get_nowait())
        while not self.events.empty():
            generation, event = self.events.get_nowait()
            if generation != self.generation:
                continue
            if event == "waiting_login":
                self.waiting_login = True
            elif event == "offline":
                self.log.write("napcat：账号离线，等待内置登录恢复")

    def notify_login(self):
        if self.notification_sent:
            return
        self.notification_sent = True
        self.log.write(f"需要手机确认：打开 http://127.0.0.1:{self.runtime['webui_port']}/webui 扫码或验证。")

        def show():
            ctypes.windll.user32.MessageBoxW(None, "机器人 QQ 需要重新扫码或安全验证。\n请打开项目 NapCat 网页完成登录，完成后自动恢复。",
                                             "QQ 项目需要登录", 0x40)
        threading.Thread(target=show, daemon=True).start()

    def tick(self, now: float | None = None):
        self.drain_events()
        now = time.monotonic() if now is None else now
        health = self.health()
        reachable, model = self.model_health()
        login = self.login_probe.snapshot() if self.alive("napcat") else None
        if login and login["waiting"]:
            self.waiting_login = True
        if health and health.get("qq_online") is True:
            self.waiting_login = self.notification_sent = False
        if model is False:
            self.set_status("故障：配置模型不存在，请安装模型；不循环重启")
            return
        checks = {"ollama": reachable,
                  "bot": self.alive("bot") and bool(health and health.get("ready")),
                  "napcat": self.alive("napcat") and bool(health and health.get("onebot_connected")
                             and health.get("qq_probe_ok") and health.get("qq_online"))}
        states = []
        for name in ("ollama", "bot", "napcat"):
            # Start downstream components only after their prerequisites are ready.
            if name == "bot" and not reachable and name not in self.started:
                states.append("recovering")
                continue
            if name == "napcat" and not checks["bot"] and name not in self.started:
                states.append("recovering")
                continue
            # A missing bot cannot prove QQ is broken. Fix the bot before touching QQ.
            if name == "napcat" and not checks["bot"] and self.alive("napcat"):
                self.policies[name].bad_since = None
                states.append("recovering")
                continue
            grace = 0 if not self.alive(name) and name != "ollama" else 60
            if name not in self.started:
                grace = 0
            if name == "napcat" and login and login.get("phase") in {"reconnecting", "initializing", "generating_qrcode"}:
                grace = 120  # Built-in 60s recovery plus login service initialization.
            if now - self.started.get(name, now) < 90 and self.alive(name):
                grace = 90
            waiting = name == "napcat" and self.waiting_login and self.alive(name)
            decision = self.policies[name].observe(checks[name], now, waiting_login=waiting, grace=grace)
            if decision == "restart":
                self.policies[name].record_restart(now)
                self.log.write(f"{name}：健康检查失败，恢复尝试 {len(self.policies[name].failures)}/3")
                try:
                    self.start_component(name)
                except (RuntimeError, OSError):
                    self.policies[name].paused = True
                    self.log.write(f"{name}：启动失败或端口冲突，已暂停；按 R 重试")
                decision = "recovering"
            states.append(decision)
        if "waiting_login" in states:
            self.notify_login()
        if "paused" in states:
            self.set_status("故障：自动恢复已暂停，修复原因后按 R 重试")
        elif "waiting_login" in states:
            self.set_status("等待扫码：其他组件保持运行")
        elif model is None or "recovering" in states:
            self.set_status("正在恢复：等待组件就绪")
        else:
            self.set_status("运行正常：QQ 在线，机器人和模型可用")

    def set_status(self, status: str):
        if status != self.status:
            self.status = status
            self.log.write(status)

    def wait(self, seconds: float) -> bool:
        import msvcrt
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key == "q":
                    return False
                if key == "r":
                    for policy in self.policies.values():
                        policy.reset()
                    self.log.write("手动重试：已解除恢复暂停")
                    return True
            time.sleep(0.1)
        return True

    def run(self):
        self.preflight()
        self.log.write("项目管理已启动；R 重试，Q 或 Ctrl+C 停止；关闭终端结束项目")
        for name in ("ollama", "bot", "napcat"):
            self.start_component(name)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                ready = (self.model_health()[0] if name == "ollama" else
                         bool((health := self.health()) and health.get("ready")) if name == "bot" else
                         port_open(self.runtime["webui_port"]))
                if ready:
                    break
                if not self.alive(name):
                    break  # Normal recovery policy handles startup crashes.
                if not self.wait(1):
                    return
            if name != "napcat" and not ready:
                self.log.write(f"{name}：尚未就绪，转入自动恢复；下游暂缓启动")
                break
        while True:
            cycle_start = time.monotonic()
            self.tick()
            if not self.wait(max(0.1, 15 - (time.monotonic() - cycle_start))):
                return

    def close(self):
        self.stopping = True
        for name in ("napcat", "bot", "ollama"):
            child = self.children.pop(name, None)
            if child is not None:
                try:
                    child.stop()
                except OSError:
                    child.close()
        self.client.close()
        self.log.write("项目已停止；外部 QQ 和复用的 Ollama 保持原状态")
        self.log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="QQ 项目手动启动管理")
    parser.add_argument("--check", action="store_true", help="只检查依赖与端口，不启动服务")
    args = parser.parse_args()
    if os.name != "nt":
        print("管理程序仅支持 Windows。")
        return 1
    os.chdir(PROJECT_ROOT)
    manager = lock = None
    try:
        from qq_ollama_bot.windows_process import ProjectLock
        lock = ProjectLock(PROJECT_ROOT)
        runtime = load_runtime()
        try:
            settings = get_settings()
        except Exception:
            raise RuntimeError("机器人配置无效，请检查现有 .env；配置内容不会输出。") from None
        manager = Supervisor(runtime, settings)
        if args.check:
            manager.preflight()
            manager.log.write("检查通过：独立运行环境、模型与连接配置可用")
        else:
            manager.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        # All RuntimeErrors above have fixed safe text; other errors expose no config.
        print(str(exc) if isinstance(exc, RuntimeError) else "运行环境检查失败，请检查部署文件。", flush=True)
        return 1
    finally:
        if manager is not None:
            manager.close()
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
