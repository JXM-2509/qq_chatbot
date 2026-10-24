from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from qq_ollama_bot.models.model_common import SYSTEM_PROMPT


PERSONA_NAME = re.compile(r"[a-z0-9_-]{1,32}\Z")


@dataclass(frozen=True, slots=True)
class PersonaSelection:
    name: str
    revision: int


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
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS persona_selections (
                    group_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    persona_name TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (group_id, user_id)
                )
                """
            )
            columns = await (
                await connection.execute("PRAGMA table_info(persona_selections)")
            ).fetchall()
            if "revision" not in {column[1] for column in columns}:
                await connection.execute(
                    "ALTER TABLE persona_selections ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
                )
            await connection.commit()

    async def get_active(self, group_id: int, user_id: int) -> str:
        return (await self.get_selection(group_id, user_id)).name

    async def get_selection(self, group_id: int, user_id: int) -> PersonaSelection:
        async with aiosqlite.connect(self._database_path) as connection:
            cursor = await connection.execute(
                """
                SELECT persona_name, revision FROM persona_selections
                WHERE group_id = ? AND user_id = ?
                """,
                (group_id, user_id),
            )
            row = await cursor.fetchone()
        return PersonaSelection(*row) if row else PersonaSelection("default", 0)

    async def set_active(self, group_id: int, user_id: int, name: str) -> None:
        async with aiosqlite.connect(self._database_path) as connection:
            await connection.execute(
                """
                INSERT INTO persona_selections (group_id, user_id, persona_name, revision)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (group_id, user_id)
                DO UPDATE SET persona_name = excluded.persona_name,
                              revision = persona_selections.revision + 1
                WHERE persona_selections.persona_name != excluded.persona_name
                """,
                (group_id, user_id, name, int(name != "default")),
            )
            await connection.commit()
