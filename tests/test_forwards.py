import asyncio
import json
from pathlib import Path

import pytest
from nonebot.adapters.onebot.v11 import MessageSegment

from qq_ollama_bot.services.forwards import ForwardInputError, ForwardReader, prepare_forward


def text(value):
    return {"type": "text", "data": {"text": value}}


def node(content, name="Alice", **kwargs):
    return {"sender": {"nickname": name, "user_id": 123456789}, "message": content, **kwargs}


def forward(ref, content=None):
    data = {"id": ref}
    if content is not None:
        data["content"] = content
    return MessageSegment("forward", data)


class Bot:
    def __init__(self, records=None, images=None):
        self.records = records or {}
        self.images = images or {}
        self.calls = []

    async def call_api(self, api, **kwargs):
        self.calls.append((api, kwargs))
        if api == "get_image":
            result = self.images[kwargs["file"]]
            if isinstance(result, Exception):
                raise result
            return {"file": str(result.resolve())}
        result = self.records[kwargs["message_id"]]
        if isinstance(result, Exception):
            raise result
        return {"messages": result} if api == "get_forward_msg" else result


async def read(bot, forwards, **limits):
    reader = ForwardReader(bot, **{
        "max_depth": 3, "max_messages": 100,
        "character_budget": 6000, "timeout_seconds": 1, **limits,
    })
    evidence = await reader.read(forwards)
    return await prepare_forward(
        bot, evidence, max_images=4, image_max_bytes=1024,
        image_total_max_bytes=2048, timeout_seconds=1,
    )


@pytest.mark.asyncio
async def test_api_and_inline_nodes_preserve_order_and_metadata():
    bot = Bot({"one": [node([text("first")], time=1700000000)]})
    inline = {"type": "node", "data": {"nickname": "Bob", "content": [text("second")]}}
    prepared = await read(bot, [forward("one"), forward("two", [inline])])
    payload = json.loads(prepared.context)
    assert [m["sender"] for m in payload["messages"]] == ["Alice", "Bob"]
    assert payload["messages"][0]["sent_at_unix"] == 1700000000
    assert payload["messages"][1]["sent_at_unix"] is None
    assert [m["content"][0]["text"] for m in payload["messages"]] == ["first", "second"]
    assert bot.calls == [("get_forward_msg", {"message_id": "one"})]
    assert "123456789" not in prepared.context


@pytest.mark.asyncio
async def test_nested_forward_keeps_position_and_does_not_execute_commands():
    nested = {"type": "forward", "data": {"id": "inner"}}
    bot = Bot({
        "outer": [node([text("before"), nested, text("after")])],
        "inner": [node([text("/reset"), {"type": "at", "data": {"qq": "999"}}], "Bob")],
    })
    prepared = await read(bot, [forward("outer")])
    messages = json.loads(prepared.context)["messages"]
    assert [item["content"][0]["text"] for item in messages] == ["before", "/reset", "after"]
    assert "@ 提及" in prepared.context
    assert "999" not in prepared.context


@pytest.mark.asyncio
async def test_reference_node_and_missing_metadata():
    bot = Bot({"42": {"message": [text("lookup")]}})
    # Node IDs are numeric, unlike forward resource IDs.
    bot.records[42] = bot.records.pop("42")
    prepared = await read(bot, [forward("outer", [{"type": "node", "data": {"id": "42"}}])])
    assert "未知发送者" in prepared.context and "lookup" in prepared.context
    assert bot.calls == [("get_msg", {"message_id": 42})]


@pytest.mark.asyncio
async def test_cycles_partial_errors_and_invalid_nodes_do_not_lose_good_content():
    bot = Bot({"outer": [node([
        text("usable"), {"type": "forward", "data": {"id": "outer"}},
        {"type": "forward", "data": {"id": "expired"}},
    ]), "bad", {"message": [{"type": "text", "data": 3}]}], "expired": RuntimeError()})
    prepared = await read(bot, [forward("outer")])
    assert "usable" in prepared.context
    assert any("循环" in note for note in prepared.notes)
    assert any("已过期" in note for note in prepared.notes)
    assert any("格式无效" in note for note in prepared.notes)
    assert len(bot.calls) == 2


@pytest.mark.asyncio
async def test_depth_count_and_escaped_character_budget():
    records = [node([text("keep"), {"type": "forward", "data": {"id": "inner"}}])]
    prepared = await read(Bot(), [forward("outer", records)], max_depth=1)
    assert "keep" in prepared.context
    assert any("层数" in note for note in prepared.notes)
    prepared = await read(Bot(), [forward("outer", [node([text("one")]), node([text("two")])])], max_messages=1)
    assert "one" in prepared.context and '"text":"two"' not in prepared.context
    assert any("条数" in note for note in prepared.notes)
    prepared = await read(Bot(), [forward("outer", [node([text('"\\\n' * 1000)], name="x" * 100)])], character_budget=500)
    assert len(prepared.context) <= 500
    assert any("截断" in note for note in prepared.notes)


@pytest.mark.asyncio
@pytest.mark.parametrize("records", [[], [node([])], RuntimeError("expired"), [node([{"type": "record", "data": {}}])]])
async def test_unreadable_record_is_not_given_to_model(records):
    with pytest.raises(ForwardInputError, match="无法读取"):
        await read(Bot({"outer": records}), [forward("outer")])


@pytest.mark.asyncio
async def test_forward_api_timeout_is_reported():
    class SlowBot:
        async def call_api(self, api, **kwargs):
            await asyncio.Event().wait()

    with pytest.raises(ForwardInputError, match="超时"):
        await read(SlowBot(), [forward("outer")], timeout_seconds=0.01)


def image(file_id):
    return {"type": "image", "data": {"file": file_id}}


@pytest.mark.asyncio
async def test_images_map_to_speakers_with_success_only_numbering(tmp_path: Path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
    bot = Bot(images={"bad": RuntimeError(), "good": path})
    prepared = await read(bot, [forward("outer", [
        node([text("first"), image("bad")], "Alice"),
        node([image("good")], "Bob"),
    ])])
    messages = json.loads(prepared.context)["messages"]
    assert "未读取" in messages[0]["content"][1]["image"]
    assert messages[1]["content"][0]["image"] == 1
    assert len(prepared.images) == 1
    assert "部分图片无法读取" in prepared.notes[0]


@pytest.mark.asyncio
async def test_pure_images_and_image_count_limit(tmp_path: Path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
    bot = Bot(images={"good": path})
    prepared = await read(bot, [forward("outer", [node([image("good")] * 5)])])
    assert len(prepared.images) == 4
    assert len(bot.calls) == 4
    assert "超过图片数量限制" in prepared.context


@pytest.mark.asyncio
async def test_image_size_and_total_budget_leave_text_readable(tmp_path: Path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 800)
    large = tmp_path / "large.png"
    large.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 2000)
    bot = Bot(images={"good": path, "large": large})
    prepared = await read(bot, [forward("outer", [node([
        text("still readable"), image("large"), *([image("good")] * 3),
    ])])])
    assert len(prepared.images) == 2
    assert "still readable" in prepared.context
    assert "未读取" in prepared.context


@pytest.mark.asyncio
async def test_all_images_failed_is_an_error():
    with pytest.raises(ForwardInputError):
        await read(Bot(images={"bad": RuntimeError()}), [forward("outer", [node([image("bad")])])])


@pytest.mark.asyncio
async def test_failed_reference_nodes_cannot_bypass_message_limit():
    bot = Bot({1: RuntimeError(), 2: RuntimeError(), 3: RuntimeError()})
    reader = ForwardReader(bot, max_depth=3, max_messages=2, character_budget=6000, timeout_seconds=1)
    evidence = await reader.read([forward("outer", [
        {"type": "node", "data": {"id": str(index)}} for index in (1, 2, 3)
    ])])
    assert len(bot.calls) == 2
    assert any("条数限制" in note for note in evidence.notes)


@pytest.mark.asyncio
async def test_inline_message_segments_and_repeated_sibling_forwards():
    inline = MessageSegment.node_custom(123, "Alice", "inline text")
    prepared = await read(Bot(), [forward("same", [inline]), forward("same", [inline])])
    assert len(json.loads(prepared.context)["messages"]) == 2
    assert not prepared.notes
