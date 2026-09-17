import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from qq_ollama_bot.services.identity import IdentityStore, MemberIdentity


def make_event(
    *,
    group_id: int = 100,
    user_id: int = 200,
    nickname: str | None = "昵称",
    card: str | None = "群名片",
    role: str | None = "member",
) -> SimpleNamespace:
    return SimpleNamespace(
        group_id=group_id,
        user_id=user_id,
        sender=SimpleNamespace(nickname=nickname, card=card, role=role),
    )


def test_identity_uses_card_and_normalizes_untrusted_labels() -> None:
    event = make_event(
        nickname="  测试   用户  ",
        card="  群内\n称呼  ",
        role="ADMIN",
    )

    identity = MemberIdentity.from_event(event, observed_at=123)

    assert identity.display_name == "群内 称呼"
    assert identity.nickname == "测试 用户"
    assert identity.role_label == "管理员"
    assert identity.updated_at == 123


def test_identity_falls_back_and_rejects_unknown_role() -> None:
    identity = MemberIdentity.from_event(
        make_event(card="", nickname="昵称", role="invented"),
        observed_at=123,
    )

    assert identity.display_name == "昵称"
    assert identity.role is None
    assert identity.role_label == "群成员"


@pytest.mark.asyncio
async def test_identity_store_updates_and_isolates_members(tmp_path: Path) -> None:
    store = IdentityStore(tmp_path / "identity.db", retention_days=30)
    await store.initialize()
    first = MemberIdentity.from_event(make_event(), observed_at=100)
    await store.upsert(first)

    updated = MemberIdentity.from_event(
        make_event(nickname=None, card="新名片", role=None),
        observed_at=200,
    )
    merged = await store.upsert(updated)
    other_group = MemberIdentity.from_event(
        make_event(group_id=101, card="另一群名片"),
        observed_at=200,
    )
    await store.upsert(other_group)

    assert merged.nickname == "昵称"
    assert merged.card == "新名片"
    assert merged.role == "member"
    assert merged.updated_at == 200
    assert (await store.upsert(other_group)).card == "另一群名片"


@pytest.mark.asyncio
async def test_identity_store_clears_card_and_removes_expired_rows(
    tmp_path: Path,
) -> None:
    store = IdentityStore(tmp_path / "identity.db", retention_days=30)
    await store.initialize()
    expired_at = int(time.time()) - 31 * 24 * 60 * 60
    old = MemberIdentity.from_event(make_event(), observed_at=expired_at)
    await store.upsert(old)

    cleared = MemberIdentity.from_event(
        make_event(card="", nickname="昵称"),
        observed_at=expired_at,
    )
    merged = await store.upsert(cleared)

    assert merged.card == ""
    assert merged.display_name == "昵称"
    assert await store.cleanup_expired() == 1
