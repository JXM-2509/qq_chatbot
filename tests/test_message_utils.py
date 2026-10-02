from nonebot.adapters.onebot.v11 import Message, MessageSegment

from qq_ollama_bot.message_utils import (
    extract_text,
    get_at_user_ids,
    has_explicit_at,
    should_reply_to_group_message,
    split_reply,
)


def test_explicit_at_and_text_extraction() -> None:
    message = Message(
        [
            MessageSegment.reply(1),
            MessageSegment.at(123),
            MessageSegment.text(" 你好 "),
        ]
    )
    assert has_explicit_at(message, 123)
    assert not has_explicit_at(message, 456)
    assert extract_text(message) == "你好"


def test_split_reply_prefers_sentence_boundary() -> None:
    chunks = split_reply("第一句话。第二句话。第三句话。", limit=10)
    assert "".join(chunks) == "第一句话。第二句话。第三句话。"
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_at_targets_preserve_order_deduplicate_and_exclude() -> None:
    message = Message(
        [
            MessageSegment.at(999),
            MessageSegment.at(123),
            MessageSegment.at(123),
            MessageSegment("at", {"qq": "all"}),
            MessageSegment.at(456),
        ]
    )

    assert get_at_user_ids(message, exclude={999}) == [123, 456]


def test_reply_trigger_accepts_at_or_reply_but_not_ordinary_message() -> None:
    ordinary = Message([MessageSegment.text("普通消息")])
    mentioned = Message([MessageSegment.at(999), MessageSegment.text("问题")])

    assert not should_reply_to_group_message(ordinary, 999, to_me=False)
    assert should_reply_to_group_message(ordinary, 999, to_me=True)
    assert should_reply_to_group_message(mentioned, 999, to_me=False)
