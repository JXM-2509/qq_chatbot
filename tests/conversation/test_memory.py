from pathlib import Path

import aiosqlite
import pytest

from qq_ollama_bot.conversation.memory import MemoryStore


@pytest.mark.asyncio
async def test_memory_isolated_and_trimmed(tmp_path: Path) -> None:
    store = MemoryStore(
        tmp_path / "memory.db",
        max_turns=2,
        character_budget=10_000,
        retention_days=30,
    )
    await store.initialize()
    for index in range(3):
        await store.append_exchange(100, 200, f"u{index}", f"a{index}")
    await store.append_exchange(100, 201, "other", "answer")

    assert await store.get_history(100, 200) == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert await store.get_history(100, 201) == [
        {"role": "user", "content": "other"},
        {"role": "assistant", "content": "answer"},
    ]


@pytest.mark.asyncio
async def test_reset_only_removes_one_session(tmp_path: Path) -> None:
    store = MemoryStore(
        tmp_path / "memory.db",
        max_turns=8,
        character_budget=10_000,
        retention_days=30,
    )
    await store.initialize()
    await store.append_exchange(100, 200, "one", "answer")
    await store.append_exchange(101, 200, "two", "answer")
    await store.reset_session(100, 200)

    assert await store.get_history(100, 200) == []
    assert len(await store.get_history(101, 200)) == 2


@pytest.mark.asyncio
async def test_history_and_trimming_are_independent_per_persona(tmp_path: Path) -> None:
    store = MemoryStore(
        tmp_path / "memory.db", max_turns=2, character_budget=10_000, retention_days=30
    )
    await store.initialize()
    await store.append_exchange(100, 200, "cat question", "cat answer")
    await store.append_exchange(101, 200, "other group", "answer", persona_name="assistant")
    await store.append_exchange(100, 201, "other user", "answer", persona_name="assistant")
    for index in range(3):
        await store.append_exchange(
            100, 200, f"u{index}", f"a{index}", persona_name="assistant"
        )

    assert await store.get_history(100, 200, persona_name="assistant") == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert await store.get_history(100, 200) == [
        {"role": "user", "content": "cat question"},
        {"role": "assistant", "content": "cat answer"},
    ]
    assert (await store.get_history(101, 200, persona_name="assistant"))[0]["content"] == "other group"
    assert (await store.get_history(100, 201, persona_name="assistant"))[0]["content"] == "other user"


@pytest.mark.asyncio
async def test_persona_reset_and_reset_all_keep_other_sessions(tmp_path: Path) -> None:
    store = MemoryStore(
        tmp_path / "memory.db", max_turns=8, character_budget=10_000, retention_days=30
    )
    await store.initialize()
    for group_id, user_id, persona in [
        (100, 200, "default"), (100, 200, "assistant"),
        (100, 201, "assistant"), (101, 200, "assistant"),
    ]:
        await store.append_exchange(group_id, user_id, "question", "answer", persona_name=persona)

    await store.reset_session(100, 200, persona_name="assistant")
    assert await store.get_history(100, 200, persona_name="assistant") == []
    assert len(await store.get_history(100, 200)) == 2
    await store.reset_all_personas(100, 200)
    assert await store.get_history(100, 200) == []
    assert len(await store.get_history(100, 201, persona_name="assistant")) == 2
    assert len(await store.get_history(101, 200, persona_name="assistant")) == 2


@pytest.mark.asyncio
async def test_legacy_migration_preserves_records_and_selection(tmp_path: Path) -> None:
    from qq_ollama_bot.conversation.persona import PersonaStore

    database_path = tmp_path / "legacy.db"
    async with aiosqlite.connect(database_path) as connection:
        await connection.executescript(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, created_at INTEGER NOT NULL
            );
            INSERT INTO messages VALUES (42, 100, 200, 'user', 'hello', 123);
            INSERT INTO messages VALUES (43, 100, 200, 'assistant', 'cat answer', 123);
            CREATE TABLE persona_selections (
                group_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
                persona_name TEXT NOT NULL, PRIMARY KEY (group_id, user_id)
            );
            INSERT INTO persona_selections VALUES (100, 200, 'assistant');
            """
        )
        await connection.commit()
    store = MemoryStore(database_path, max_turns=8, character_budget=10_000, retention_days=30)
    personas = PersonaStore(database_path)
    for _ in range(2):
        await store.initialize()
        await personas.initialize()

    assert await personas.get_active(100, 200) == "assistant"
    assert await store.get_history(100, 200, persona_name="assistant") == []
    assert (await store.get_history(100, 200))[1]["content"] == "cat answer"
    async with aiosqlite.connect(database_path) as connection:
        rows = await (await connection.execute(
            "SELECT id, group_id, user_id, role, content, created_at, persona_name FROM messages ORDER BY id"
        )).fetchall()
    assert rows == [
        (42, 100, 200, "user", "hello", 123, "default"),
        (43, 100, 200, "assistant", "cat answer", 123, "default"),
    ]
