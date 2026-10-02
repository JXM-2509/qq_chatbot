from __future__ import annotations

import re
from pathlib import Path

import aiosqlite

from qq_ollama_bot.services.ollama import SYSTEM_PROMPT


PERSONA_NAME = re.compile(r"[a-z0-9_-]{1,32}\Z")


class PersonaCatalog:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def names(self) -> list[str]:
        return [
            "default",
            *sorted(
                path.stem
                for path in self._directory.glob("*.md")
                if path.is_file()
                and path.stem != "default"
                and PERSONA_NAME.fullmatch(path.stem)
                and path.read_text(encoding="utf-8").strip()
            ),
        ]

    def get_prompt(self, name: str) -> str | None:
        if name == "default":
            return SYSTEM_PROMPT
        if not PERSONA_NAME.fullmatch(name):
            return None
        path = self._directory / f"{name}.md"
        if not path.is_file():
            return None
        prompt = path.read_text(encoding="utf-8").strip()
        return prompt or None


class PersonaStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    async def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._database_path) as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS persona_selections (
                    group_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    persona_name TEXT NOT NULL,
                    PRIMARY KEY (group_id, user_id)
                )
                """
            )
            await connection.commit()

    async def get_active(self, group_id: int, user_id: int) -> str:
        async with aiosqlite.connect(self._database_path) as connection:
            cursor = await connection.execute(
                """
                SELECT persona_name FROM persona_selections
                WHERE group_id = ? AND user_id = ?
                """,
                (group_id, user_id),
            )
            row = await cursor.fetchone()
        return row[0] if row else "default"

    async def set_active(self, group_id: int, user_id: int, name: str) -> None:
        async with aiosqlite.connect(self._database_path) as connection:
            await connection.execute(
                """
                INSERT INTO persona_selections (group_id, user_id, persona_name)
                VALUES (?, ?, ?)
                ON CONFLICT (group_id, user_id)
                DO UPDATE SET persona_name = excluded.persona_name
                """,
                (group_id, user_id, name),
            )
            await connection.commit()
