from pathlib import Path

import pytest

from qq_ollama_bot.services.persona import PersonaCatalog, PersonaStore


def test_catalog_loads_presets_and_rejects_paths(tmp_path: Path) -> None:
    (tmp_path / "writer.md").write_text("  你是作家。\n", encoding="utf-8")
    (tmp_path / "empty.md").write_text(" \n", encoding="utf-8")
    catalog = PersonaCatalog(tmp_path)

    assert catalog.names() == ["default", "writer"]
    assert catalog.get_prompt("writer") == "你是作家。"
    assert catalog.get_prompt("../writer") is None
    assert catalog.get_prompt("empty") is None
    assert catalog.get_prompt("default")


@pytest.mark.asyncio
async def test_persona_selection_persists_per_group_and_user(tmp_path: Path) -> None:
    database_path = tmp_path / "personas.db"
    store = PersonaStore(database_path)
    await store.initialize()
    assert await store.get_active(100, 200) == "default"

    await store.set_active(100, 200, "writer")
    reloaded = PersonaStore(database_path)
    assert await reloaded.get_active(100, 200) == "writer"
    assert await reloaded.get_active(100, 201) == "default"
    assert await reloaded.get_active(101, 200) == "default"

    await reloaded.set_active(100, 200, "default")
    assert await store.get_active(100, 200) == "default"


@pytest.mark.asyncio
async def test_selection_detects_switch_away_and_back(tmp_path: Path) -> None:
    store = PersonaStore(tmp_path / "personas.db")
    await store.initialize()
    initial = await store.get_selection(100, 200)
    await store.set_active(100, 200, "default")
    assert await store.get_selection(100, 200) == initial
    await store.set_active(100, 200, "assistant")
    assistant = await store.get_selection(100, 200)
    await store.set_active(100, 200, "assistant")
    assert await store.get_selection(100, 200) == assistant
    await store.set_active(100, 200, "default")
    returned = await store.get_selection(100, 200)
    assert returned.name == initial.name
    assert returned.revision > initial.revision
