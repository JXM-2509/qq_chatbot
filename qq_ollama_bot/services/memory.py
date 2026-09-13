from __future__ import annotations

import time
from pathlib import Path

import aiosqlite


class MemoryStore:
    def __init__(
        self,
        database_path: Path,
        *,
        max_turns: int,
        character_budget: int,
        retention_days: int,
    ) -> None:
        self._database_path = database_path
        self._max_messages = max_turns * 2
        self._character_budget = character_budget
        self._retention_days = retention_days

    async def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._database_path) as connection:
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            await connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages (group_id, user_id, id DESC)
                """
            )
            await connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_created_at
                ON messages (created_at)
                """
            )
            await connection.commit()

    async def get_history(self, group_id: int, user_id: int) -> list[dict[str, str]]:
        async with aiosqlite.connect(self._database_path) as connection:
            cursor = await connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE group_id = ? AND user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (group_id, user_id, self._max_messages),
            )
            rows = await cursor.fetchall()

        selected: list[dict[str, str]] = []
        used_characters = 0
        for role, content in rows:
            remaining = self._character_budget - used_characters
            if remaining <= 0:
                break
            if len(content) > remaining:
                if not selected:
                    selected.append({"role": role, "content": content[-remaining:]})
                break
            selected.append({"role": role, "content": content})
            used_characters += len(content)
        selected.reverse()
        return selected

    async def append_exchange(
        self,
        group_id: int,
        user_id: int,
        user_content: str,
        assistant_content: str,
    ) -> None:
        created_at = int(time.time())
        async with aiosqlite.connect(self._database_path) as connection:
            await connection.executemany(
                """
                INSERT INTO messages (group_id, user_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (group_id, user_id, "user", user_content, created_at),
                    (group_id, user_id, "assistant", assistant_content, created_at),
                ),
            )
            await connection.execute(
                """
                DELETE FROM messages
                WHERE group_id = ? AND user_id = ?
                  AND id NOT IN (
                      SELECT id
                      FROM messages
                      WHERE group_id = ? AND user_id = ?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (group_id, user_id, group_id, user_id, self._max_messages),
            )
            await connection.commit()

    async def reset_session(self, group_id: int, user_id: int) -> None:
        async with aiosqlite.connect(self._database_path) as connection:
            await connection.execute(
                "DELETE FROM messages WHERE group_id = ? AND user_id = ?",
                (group_id, user_id),
            )
            await connection.commit()

    async def cleanup_expired(self) -> int:
        cutoff = int(time.time()) - self._retention_days * 24 * 60 * 60
        async with aiosqlite.connect(self._database_path) as connection:
            cursor = await connection.execute(
                "DELETE FROM messages WHERE created_at < ?",
                (cutoff,),
            )
            await connection.commit()
            return max(cursor.rowcount, 0)

