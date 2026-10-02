from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import aiosqlite


@dataclass(frozen=True, slots=True)
class GroupMessageRecord:
    content: str
    created_at: int


def trim_recent_messages(
    messages: Sequence[GroupMessageRecord],
    character_budget: int,
) -> tuple[GroupMessageRecord, ...]:
    selected: list[GroupMessageRecord] = []
    remaining = character_budget
    for message in reversed(messages):
        if remaining <= 0:
            break
        if len(message.content) <= remaining:
            selected.append(message)
            remaining -= len(message.content)
            continue
        selected.append(
            GroupMessageRecord(
                content=message.content[:remaining],
                created_at=message.created_at,
            )
        )
        break
    selected.reverse()
    return tuple(selected)


class GroupMessageStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    async def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._database_path) as connection:
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute("PRAGMA foreign_keys=ON")
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS group_messages (
                    group_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (group_id, message_id)
                )
                """
            )
            await connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_group_messages_member_time
                ON group_messages (group_id, user_id, created_at DESC, message_id DESC)
                """
            )
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS group_message_mentions (
                    group_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    target_user_id INTEGER NOT NULL,
                    PRIMARY KEY (group_id, message_id, target_user_id),
                    FOREIGN KEY (group_id, message_id)
                        REFERENCES group_messages (group_id, message_id)
                        ON DELETE CASCADE
                )
                """
            )
            await connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_group_mentions_target
                ON group_message_mentions (group_id, target_user_id, message_id)
                """
            )
            await connection.commit()

    async def append_event(
        self,
        *,
        group_id: int,
        message_id: int,
        user_id: int,
        content: str,
        created_at: int,
        mentioned_user_ids: Sequence[int],
    ) -> bool:
        unique_mentions = tuple(dict.fromkeys(mentioned_user_ids))
        async with aiosqlite.connect(self._database_path) as connection:
            await connection.execute("PRAGMA foreign_keys=ON")
            cursor = await connection.execute(
                """
                INSERT OR IGNORE INTO group_messages (
                    group_id, message_id, user_id, content, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (group_id, message_id, user_id, content, created_at),
            )
            inserted = cursor.rowcount > 0
            if inserted and unique_mentions:
                await connection.executemany(
                    """
                    INSERT INTO group_message_mentions (
                        group_id, message_id, target_user_id
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        (group_id, message_id, target_user_id)
                        for target_user_id in unique_mentions
                    ),
                )
            await connection.commit()
        return inserted

    async def get_recent_messages(
        self,
        group_id: int,
        user_id: int,
        limit: int,
    ) -> list[GroupMessageRecord]:
        async with aiosqlite.connect(self._database_path) as connection:
            cursor = await connection.execute(
                """
                SELECT content, created_at
                FROM group_messages
                WHERE group_id = ? AND user_id = ? AND length(trim(content)) > 0
                ORDER BY created_at DESC, message_id DESC
                LIMIT ?
                """,
                (group_id, user_id, limit),
            )
            rows = await cursor.fetchall()

        return [
            GroupMessageRecord(content=content, created_at=created_at)
            for content, created_at in reversed(rows)
        ]
