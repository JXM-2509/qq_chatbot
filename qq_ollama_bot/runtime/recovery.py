"""Pure recovery policy, with monotonic clocks supplied by the supervisor."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RecoveryPolicy:
    failures: list[float] = field(default_factory=list)
    bad_since: float | None = None
    retry_at: float = 0
    healthy_since: float | None = None
    paused: bool = False

    def observe(self, healthy: bool, now: float, *, waiting_login: bool = False,
                grace: float = 60) -> str:
        # A QR challenge is a stable waiting state, never a crash/restart signal.
        if waiting_login:
            self.bad_since = None
            return "waiting_login"
        self.failures = [stamp for stamp in self.failures if now - stamp < 900]
        if healthy:
            self.paused = False
            self.bad_since = None
            if self.healthy_since is None:
                self.healthy_since = now
            if now - self.healthy_since >= 300:
                self.failures.clear()
            return "healthy"
        self.healthy_since = None
        if self.bad_since is None:
            self.bad_since = now
        # Circuit stays open until a successful health probe or explicit retry.
        if self.paused:
            return "paused"
        if now - self.bad_since < grace or now < self.retry_at:
            return "recovering"
        if len(self.failures) >= 3:
            self.paused = True
            return "paused"
        return "restart"

    def record_restart(self, now: float) -> None:
        self.failures = [stamp for stamp in self.failures if now - stamp < 900]
        self.failures.append(now)
        self.retry_at = now + (5, 15, 60)[min(len(self.failures), 3) - 1]
        self.bad_since = None

    def reset(self) -> None:
        self.failures.clear()
        self.bad_since = self.healthy_since = None
        self.retry_at = 0
        self.paused = False


def login_event(line: str) -> str | None:
    """Consume only framework diagnostics, never quoted chat content or URLs."""
    if any(word in line for word in ("接收 <-", "发送 ->", "群聊 [", "私聊 [")):
        return None
    if "[KickedOffLine]" in line:
        return "offline"
    if "账号状态变更为离线" in line:
        return "offline"
    if any(word in line for word in ("请扫描下面的二维码", "请使用手机QQ扫描", "二维码已被扫描", "等待确认", "需要安全验证", "需要验证码")):
        return "waiting_login"
    return None
