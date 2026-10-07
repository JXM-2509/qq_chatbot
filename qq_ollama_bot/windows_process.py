"""Owned Windows process trees: assign suspended children before they can spawn."""
from __future__ import annotations

import ctypes as c
from ctypes import wintypes as w
import hashlib
import msvcrt
import os
from pathlib import Path
import subprocess
import threading
import time
from collections.abc import Callable


class StartupInfo(c.Structure):
    _fields_ = [("cb", w.DWORD), ("lpReserved", w.LPWSTR), ("lpDesktop", w.LPWSTR),
                ("lpTitle", w.LPWSTR), ("dwX", w.DWORD), ("dwY", w.DWORD),
                ("dwXSize", w.DWORD), ("dwYSize", w.DWORD), ("dwXCountChars", w.DWORD),
                ("dwYCountChars", w.DWORD), ("dwFillAttribute", w.DWORD),
                ("dwFlags", w.DWORD), ("wShowWindow", w.WORD), ("cbReserved2", w.WORD),
                ("lpReserved2", c.POINTER(c.c_byte)), ("hStdInput", w.HANDLE),
                ("hStdOutput", w.HANDLE), ("hStdError", w.HANDLE)]


class ProcessInfo(c.Structure):
    _fields_ = [("hProcess", w.HANDLE), ("hThread", w.HANDLE),
                ("dwProcessId", w.DWORD), ("dwThreadId", w.DWORD)]


class BasicLimits(c.Structure):
    _fields_ = [("PerProcessUserTimeLimit", c.c_int64), ("PerJobUserTimeLimit", c.c_int64),
                ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", c.c_size_t),
                ("MaximumWorkingSetSize", c.c_size_t), ("ActiveProcessLimit", w.DWORD),
                ("Affinity", c.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]


class IoCounters(c.Structure):
    _fields_ = [(name, c.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount",
                "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class ExtendedLimits(c.Structure):
    _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", c.c_size_t), ("JobMemoryLimit", c.c_size_t),
                ("PeakProcessMemoryUsed", c.c_size_t), ("PeakJobMemoryUsed", c.c_size_t)]


class Accounting(c.Structure):
    _fields_ = [(name, c.c_int64) for name in ("TotalUserTime", "TotalKernelTime",
                "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")] + [
                (name, w.DWORD) for name in ("TotalPageFaultCount", "TotalProcesses",
                "ActiveProcesses", "TotalTerminatedProcesses")]


def kernel():
    k = c.WinDLL("kernel32", use_last_error=True)
    prototypes = {
        "CreateJobObjectW": ([c.c_void_p, w.LPCWSTR], w.HANDLE),
        "SetInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
        "QueryInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL),
        "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
        "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
        "CloseHandle": ([w.HANDLE], w.BOOL),
        "ResumeThread": ([w.HANDLE], w.DWORD),
        "CreateMutexW": ([c.c_void_p, w.BOOL, w.LPCWSTR], w.HANDLE),
        "GenerateConsoleCtrlEvent": ([w.DWORD, w.DWORD], w.BOOL),
        "TerminateProcess": ([w.HANDLE, w.UINT], w.BOOL),
        "CreateProcessW": ([w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p, w.BOOL,
                            w.DWORD, c.c_void_p, w.LPCWSTR, c.POINTER(StartupInfo),
                            c.POINTER(ProcessInfo)], w.BOOL),
    }
    for name, (args, result) in prototypes.items():
        getattr(k, name).argtypes, getattr(k, name).restype = args, result
    return k


def require(success):
    if not success:
        raise c.WinError(c.get_last_error())
    return success


class ProjectLock:
    def __init__(self, root: Path):
        self.k = kernel()
        name = "Local\\QQOllamaSupervisor-" + hashlib.sha256(str(root.resolve()).casefold().encode()).hexdigest()[:24]
        self.handle = require(self.k.CreateMutexW(None, False, name))
        if c.get_last_error() == 183:
            self.close()
            raise RuntimeError("项目管理终端已经运行，请勿重复启动。")

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None


class OwnedProcess:
    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str], on_line: Callable[[str], None]):
        self.k, self.handle, self.job, self.pid = kernel(), None, None, 0
        self.job = require(self.k.CreateJobObjectW(None, None))
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        try:
            require(self.k.SetInformationJobObject(self.job, 9, c.byref(limits), c.sizeof(limits)))
            read_fd, write_fd = os.pipe()
            try:
                os.set_inheritable(write_fd, True)
                with open(os.devnull, "rb") as stdin:
                    os.set_inheritable(stdin.fileno(), True)
                    info = StartupInfo()
                    info.cb, info.dwFlags, info.wShowWindow = c.sizeof(info), 0x101, 0
                    info.hStdInput = msvcrt.get_osfhandle(stdin.fileno())
                    info.hStdOutput = info.hStdError = msvcrt.get_osfhandle(write_fd)
                    proc = ProcessInfo()
                    cmd = c.create_unicode_buffer(subprocess.list2cmdline(argv))
                    env_block = c.create_unicode_buffer("\0".join(f"{key}={value}" for key, value in sorted(env.items())) + "\0\0")
                    # Suspended + own process group + Unicode environment; inherits the manager console.
                    require(self.k.CreateProcessW(argv[0], cmd, None, None, True, 0x4 | 0x200 | 0x400,
                                                  env_block, str(cwd), c.byref(info), c.byref(proc)))
                    self.handle, self.pid = proc.hProcess, proc.dwProcessId
                    try:
                        require(self.k.AssignProcessToJobObject(self.job, self.handle))
                        if self.k.ResumeThread(proc.hThread) == 0xFFFFFFFF:
                            raise c.WinError(c.get_last_error())
                    except BaseException:
                        self.k.TerminateProcess(self.handle, 1)
                        raise
                    finally:
                        self.k.CloseHandle(proc.hThread)
            except BaseException:
                os.close(read_fd)
                raise
            finally:
                os.close(write_fd)
            self.reader = threading.Thread(target=self._read, args=(read_fd, on_line), daemon=True)
            self.reader.start()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _read(fd: int, callback: Callable[[str], None]):
        with os.fdopen(fd, "rb") as stream:
            while line := stream.readline(8192):
                try:
                    callback(line.decode("utf-8", errors="replace").rstrip())
                except Exception:
                    # Output handling must never block or terminate the child.
                    continue

    def alive(self) -> bool:
        if not self.job:
            return False
        accounting = Accounting()
        require(self.k.QueryInformationJobObject(self.job, 1, c.byref(accounting), c.sizeof(accounting), None))
        return accounting.ActiveProcesses > 0

    def stop(self, grace: float = 5):
        if self.alive():
            self.k.GenerateConsoleCtrlEvent(1, self.pid)  # CTRL_BREAK, only this process group
            deadline = time.monotonic() + grace
            while self.alive() and time.monotonic() < deadline:
                time.sleep(0.1)
            if self.alive():
                require(self.k.TerminateJobObject(self.job, 1))
        self.close()

    def close(self):
        for attr in ("job", "handle"):
            handle = getattr(self, attr, None)
            if handle:
                self.k.CloseHandle(handle)
                setattr(self, attr, None)
