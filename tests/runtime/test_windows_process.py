import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows process ownership")


def test_project_mutex_rejects_duplicate_and_releases(tmp_path):
    from qq_ollama_bot.runtime.windows_process import ProjectLock
    lock = ProjectLock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="重复启动"):
            ProjectLock(tmp_path)
    finally:
        lock.close()
    second = ProjectLock(tmp_path)
    second.close()


def test_job_owns_descendants_even_after_root_exits(tmp_path):
    from qq_ollama_bot.runtime.windows_process import OwnedProcess
    lines = []
    code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print('root-exited', flush=True)"
    child = OwnedProcess([sys.executable, "-c", code], tmp_path, os.environ.copy(), lines.append)
    try:
        deadline = time.monotonic() + 10
        while "root-exited" not in lines and time.monotonic() < deadline:
            time.sleep(0.05)
        assert "root-exited" in lines
        assert child.alive()
        job = child.job
        child.stop(grace=0)
        assert child.job is None and not child.alive()
    finally:
        child.close()


def test_manager_abrupt_exit_kills_owned_tree(tmp_path):
    pid_file = tmp_path / "child-pid.txt"
    script = """import os,sys,time
from pathlib import Path
from qq_ollama_bot.runtime.windows_process import OwnedProcess
child = OwnedProcess([sys.executable,'-c','import time;time.sleep(60)'], Path.cwd(), os.environ.copy(), lambda line: None)
Path(sys.argv[1]).write_text(str(child.pid))
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", script, str(pid_file)], check=True, timeout=15)
    pid = int(pid_file.read_text())
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000, False, pid)
    if handle:
        try:
            assert kernel.WaitForSingleObject(handle, 5000) == 0
        finally:
            kernel.CloseHandle(handle)
