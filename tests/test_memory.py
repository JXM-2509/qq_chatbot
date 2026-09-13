from pathlib import Path

import pytest

from qq_ollama_bot.services.memory import MemoryStore


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

