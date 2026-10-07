"""Prepare pinned portable binaries without running either system installer.

Downloads are optional. Existing installs, account data and .env are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "runtime"
ARTIFACTS = {
    "NapCat.Shell.Windows.OneKey-4.18.33.zip": (
        "https://github.com/NapNeko/NapCatQQ/releases/download/v4.18.33/NapCat.Shell.Windows.OneKey.zip",
        "46c3ec155aa61f12e89bbf97013a21243d00b0019b5565992a88852d7627f39c"),
    "NapCat.Shell-4.18.33.zip": (
        "https://github.com/NapNeko/NapCatQQ/releases/download/v4.18.33/NapCat.Shell.zip",
        "4b8e20e6d22288586d99eb0b34fff6c7ee1d88b00353039326f8c459a86df667"),
    "QQ_9.9.33_260813_x64_01.exe": (
        "https://qqdl.gtimg.cn/qqfile/QQNT/9.9.33/release/497e2f1f/QQ_9.9.33_260813_x64_01.exe",
        "b25c0d3ce9df764074a9118d0ded927e1b2d7ebf60e306112e8df18a040ec492"),
}


def digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(download: bool = False):
    RUNTIME.mkdir(exist_ok=True)
    downloads = RUNTIME / "downloads"
    downloads.mkdir(exist_ok=True)
    for name, (url, expected) in ARTIFACTS.items():
        path = downloads / name
        if not path.exists() and download:
            print(f"下载固定版本：{name}", flush=True)
            # Do not replace an existing artifact or remove a failed download.
            with urllib.request.urlopen(url, timeout=60) as response, path.open("xb") as output:
                shutil.copyfileobj(response, output)
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"附件缺失或校验失败：{name}。原文件保留，未部署。")
    one_key, napcat, qq = (RUNTIME / "onekey-4.18.33", RUNTIME / "napcat-4.18.33",
                          RUNTIME / "qq-9.9.33-52230")
    for archive, target in (("NapCat.Shell.Windows.OneKey-4.18.33.zip", one_key),
                            ("NapCat.Shell-4.18.33.zip", napcat)):
        if not target.exists():
            with zipfile.ZipFile(downloads / archive) as file:
                # All members must remain inside the intended portable directory.
                for member in file.namelist():
                    if not (target / member).resolve().is_relative_to(target.resolve()):
                        raise RuntimeError("压缩包路径校验失败。")
                file.extractall(target)
    qq_exe = qq / "Files" / "QQ.exe"
    if not qq_exe.exists():
        subprocess.run([str(one_key / "7z.exe"), "x", str(downloads / "QQ_9.9.33_260813_x64_01.exe"),
                        f"-o{qq}", "Files\\*", "-y", "-bso0", "-bsp0"], check=True)
    package = qq / "Files" / "versions" / "9.9.33-52230" / "resources" / "app" / "package.json"
    qq_meta = json.loads(package.read_text(encoding="utf-8"))
    if qq_meta["version"] != "9.9.33-52230":
        raise RuntimeError("QQ 运行版本不匹配。")
    source = (napcat / "napcat.mjs").read_text(encoding="utf-8")
    if '"4.18.33"' not in source or '"SelfOffline"' not in source:
        raise RuntimeError("NapCat 版本或掉线恢复实现不匹配。")

    manifest_path = RUNTIME / "project-runtime.json"
    if manifest_path.exists():
        print("附件校验通过，已有运行配置保留。")
        return
    old = ROOT.parent / "napcat" / "config"
    accounts = sorted(old.glob("onebot11_*.json"))
    if len(accounts) != 1:
        raise RuntimeError("需要唯一的机器人账号配置，未自动选择账号。")
    account = accounts[0].stem.removeprefix("onebot11_")
    cfg = napcat / "config"
    cfg.mkdir(exist_ok=True)
    # Copy credentials byte-for-byte first; only non-secret fields below change.
    for source_file in old.glob("*.json"):
        target_file = cfg / source_file.name
        if not target_file.exists():
            shutil.copy2(source_file, target_file)
    for name in ("napcat.json", f"napcat_{account}.json"):
        target = cfg / name
        data = json.loads(target.read_text(encoding="utf-8-sig"))
        data["fileLog"] = False  # The manager stores filtered diagnostics, not raw chat logs.
        write_json(target, data)
    target = cfg / f"onebot11_{account}.json"
    data = json.loads(target.read_text(encoding="utf-8-sig"))
    for client in data["network"]["websocketClients"]:
        if client.get("enable"):
            client.update(heartInterval=15_000, reconnectInterval=5_000)
    for group in ("httpServers", "websocketServers"):
        for server in data["network"].get(group, []):
            server["host"] = "127.0.0.1"
    write_json(target, data)
    target = cfg / "webui.json"
    data = json.loads(target.read_text(encoding="utf-8-sig"))
    data.update(host="127.0.0.1", port=6099)
    write_json(target, data)
    # Patch metadata matches the independent QQ, rather than the package's example QQ.
    qq_meta["main"] = "./loadNapCat.js"
    write_json(napcat / "qqnt.json", qq_meta)
    ollama = shutil.which("ollama") or str(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Ollama" / "ollama.exe")
    manifest = {"napcat_version": "4.18.33", "qq_version": "9.9.33-52230", "account": account,
                "napcat_dir": str(napcat), "qq_exe": str(qq_exe), "ollama_exe": ollama, "webui_port": 6099,
                "hashes": {str(path): digest(path) for path in (
                    napcat / "napcat.mjs", napcat / "NapCatWinBootMain.exe", napcat / "NapCatWinBootHook.dll", qq_exe)}}
    write_json(manifest_path, manifest)
    print("独立运行环境部署完成；原 QQ、NapCat、登录数据和 .env 保留。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.download)
    except Exception as exc:
        print(str(exc) if isinstance(exc, RuntimeError) else "部署失败，请检查本地附件和运行目录。", file=sys.stderr)
        sys.exit(1)
