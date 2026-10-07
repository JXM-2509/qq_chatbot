from qq_ollama_bot.recovery import RecoveryPolicy, login_event


def test_short_network_drop_recovers_without_restart():
    policy = RecoveryPolicy()
    assert policy.observe(False, 0) == "recovering"
    assert policy.observe(False, 59) == "recovering"
    assert policy.observe(True, 60) == "healthy"
    assert policy.failures == []


def test_persistent_offline_and_crash_recovery():
    policy = RecoveryPolicy()
    policy.observe(False, 0)
    assert policy.observe(False, 60) == "restart"
    policy.record_restart(60)
    assert policy.retry_at == 65
    assert policy.observe(False, 61, grace=0) == "recovering"
    assert policy.observe(False, 65, grace=0) == "restart"
    policy.record_restart(65)
    assert policy.retry_at == 80
    assert policy.observe(False, 80, grace=0) == "restart"
    policy.record_restart(80)
    assert policy.retry_at == 140
    assert policy.observe(False, 140, grace=0) == "paused"
    assert policy.observe(False, 5000, grace=0) == "paused"
    policy.reset()
    assert policy.observe(False, 5001, grace=0) == "restart"


def test_waiting_for_scan_does_not_restart_even_after_hours():
    policy = RecoveryPolicy()
    for now in (0, 60, 900, 10000):
        assert policy.observe(False, now, waiting_login=True) == "waiting_login"
    assert policy.failures == []
    assert policy.observe(True, 10001) == "healthy"


def test_success_resumes_and_stable_run_clears_failures():
    policy = RecoveryPolicy(paused=True, failures=[10, 20, 30])
    assert policy.observe(True, 40) == "healthy"
    assert not policy.paused
    policy.observe(True, 340)
    assert not policy.failures


def test_login_diagnostics_do_not_parse_quoted_group_messages():
    assert login_event("[Core] [Login] 请扫描下面的二维码，然后在手Q上授权登录：") == "waiting_login"
    assert login_event("[error] [KickedOffLine] [下线通知] 登录失效") == "offline"
    assert login_event("接收 <- 群聊 [test] [user] [KickedOffLine] 请扫描下面的二维码") is None
    assert login_event("token=secret https://example.com/qrcode") is None
