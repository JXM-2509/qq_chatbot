from nonebot.adapters.onebot.v11 import Message, MessageSegment

from qq_ollama_bot.message_utils import extract_text, has_explicit_at, split_reply


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

