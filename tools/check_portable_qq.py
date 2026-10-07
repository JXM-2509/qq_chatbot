"""No-login portable QQ smoke check: isolated ports/config, no OneBot delivery."""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import time

import httpx

from qq_ollama_bot.supervisor import NapCatLoginProbe, load_runtime
from qq_ollama_bot.windows_process import OwnedProcess


def main():
    root = Path(__file__).resolve().parents[1]
    runtime = load_runtime()
    directory = root / "data" / f"portable-check-{datetime.now():%Y%m%d-%H%M%S}"
    shutil.copytree(runtime["napcat_dir"], directory)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config = directory / "config" / "webui.json"
    webui = json.loads(config.read_text(encoding="utf-8-sig"))
    if webui.get("autoLoginAccount"):
        raise RuntimeError("自动登录已配置，停止无登录测试。")
    webui.update(port=port, host="127.0.0.1")
    config.write_text(json.dumps(webui, ensure_ascii=False), encoding="utf-8")
    for file in (directory / "config").glob("onebot11_*.json"):
        onebot = json.loads(file.read_text(encoding="utf-8-sig"))
        for connections in onebot["network"].values():
            if isinstance(connections, list):
                for connection in connections:
                    connection["enable"] = False
        file.write_text(json.dumps(onebot, ensure_ascii=False), encoding="utf-8")
    env = os.environ.copy()
    env.update(NAPCAT_PATCH_PACKAGE=str(directory / "qqnt.json"),
               NAPCAT_LOAD_PATH=str(directory / "loadNapCat.js"),
               NAPCAT_INJECT_PATH=str(directory / "NapCatWinBootHook.dll"),
               NAPCAT_MAIN_PATH=str(directory / "napcat.mjs"))
    # Deliberately omit -q. This check does not authenticate any QQ account.
    child = OwnedProcess([str(directory / "NapCatWinBootMain.exe"), runtime["qq_exe"],
                          str(directory / "NapCatWinBootHook.dll")], directory, env, lambda line: None)
    passed = False
    try:
        with httpx.Client(timeout=3, trust_env=False) as client:
            probe = NapCatLoginProbe(client, config, port)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                state = probe.snapshot()
                if state and state["waiting"]:
                    passed = True
                    break
                if not child.alive():
                    break
                time.sleep(1)
    finally:
        child.stop(grace=2)
    report = {"napcat": runtime["napcat_version"], "qq": runtime["qq_version"],
              "login_service_ready": passed, "account_login_attempted": False,
              "onebot_connections_enabled": False, "owned_processes_stopped": not child.alive()}
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
