from pathlib import Path

import aiosqlite
import pytest

from qq_ollama_bot.conversation.group_messages import (
    GroupMessageRecord,
    GroupMessageStore,
    trim_recent_messages,
)


@pytest.mark.asyncio
async def test_archive_is_idempotent_and_saves_unique_mentions(tmp_path: Path) -> None:
    database_path = tmp_path / "messages.db"
    store = GroupMessageStore(database_path)
    await store.initialize()

    assert await store.append_event(
        group_id=100,
        message_id=10,
        user_id=200,
        content="你好",
        created_at=1000,
        mentioned_user_ids=[300, 300, 400],
    )
    assert not await store.append_event(
        group_id=100,
        message_id=10,
        user_id=200,
        content="重复事件",
        created_at=1001,
        mentioned_user_ids=[500],
    )

    async with aiosqlite.connect(database_path) as connection:
        messages = await (await connection.execute(
            "SELECT content FROM group_messages"
        )).fetchall()
        mentions = await (await connection.execute(
            """
            SELECT target_user_id FROM group_message_mentions
            ORDER BY target_user_id
            """
        )).fetchall()

    assert messages == [("你好",)]
    assert mentions == [(300,), (400,)]


@pytest.mark.asyncio
async def test_recent_messages_are_member_and_group_scoped(tmp_path: Path) -> None:
    store = GroupMessageStore(tmp_path / "messages.db")
    await store.initialize()
    records = [
        (100, 1, 200, "旧消息", 100),
        (100, 2, 200, "新消息", 300),
        (100, 3, 200, "中间消息", 200),
        (100, 4, 201, "其他人", 400),
        (101, 5, 200, "其他群", 500),
        (100, 6, 200, "", 600),
    ]
    for group_id, message_id, user_id, content, created_at in records:
        await store.append_event(
            group_id=group_id,
            message_id=message_id,
            user_id=user_id,
            content=content,
            created_at=created_at,
            mentioned_user_ids=[],
        )

    recent = await store.get_recent_messages(100, 200, limit=2)

    assert [(item.content, item.created_at) for item in recent] == [
        ("中间消息", 200),
        ("新消息", 300),
    ]


def test_character_budget_keeps_newest_messages() -> None:
    records = [
        GroupMessageRecord("旧消息", 100),
        GroupMessageRecord("中消息", 200),
        GroupMessageRecord("最新消息", 300),
    ]

    selected = trim_recent_messages(records, character_budget=7)

    assert [(item.content, item.created_at) for item in selected] == [
        ("中消息", 200),
        ("最新消息", 300),
    ]
