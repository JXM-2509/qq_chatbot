from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import aiosqlite


MAX_IDENTITY_LABEL_LENGTH = 100
ROLE_LABELS = {
    "owner": "群主",
    "admin": "管理员",
    "member": "群成员",
}


class GroupMessageLike(Protocol):
    group_id: int
    user_id: int
    sender: object


def _normalize_label(value: object) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).split())[:MAX_IDENTITY_LABEL_LENGTH]


def _normalize_role(value: object) -> str | None:
    if value is None:
        return None
    role = str(value).strip().casefold()
    return role if role in ROLE_LABELS else None


@dataclass(frozen=True, slots=True)
class MemberIdentity:
    group_id: int
    user_id: int
    nickname: str | None
    card: str | None
    role: str | None
    updated_at: int

    @classmethod
    def from_event(
        cls,
        event: GroupMessageLike,
        *,
        observed_at: int | None = None,
    ) -> "MemberIdentity":
        sender = event.sender
        return cls(
            group_id=event.group_id,
            user_id=event.user_id,
            nickname=_normalize_label(getattr(sender, "nickname", None)),
            card=_normalize_label(getattr(sender, "card", None)),
            role=_normalize_role(getattr(sender, "role", None)),
            updated_at=int(time.time()) if observed_at is None else observed_at,
        )

    @classmethod
    def from_group_member_info(
        cls,
        group_id: int,
        user_id: int,
        data: Mapping[str, object],
        *,
        observed_at: int | None = None,
    ) -> "MemberIdentity":
        return cls(
            group_id=group_id,
            user_id=user_id,
            nickname=_normalize_label(data.get("nickname")),
            card=_normalize_label(data.get("card")),
            role=_normalize_role(data.get("role")),
            updated_at=int(time.time()) if observed_at is None else observed_at,
        )

    @property
    def display_name(self) -> str:
        return self.card or self.nickname or "群成员"

    @property
    def role_label(self) -> str:
        return ROLE_LABELS.get(self.role, "群成员")


class IdentityStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    async def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._database_path) as connection:
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS group_members (
                    group_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    nickname TEXT,
                    card TEXT,
                    role TEXT,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (group_id, user_id)
                )
                """
            )
            await connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_group_members_updated_at
                ON group_members (updated_at)
                """
            )
            await connection.commit()

    async def upsert(self, identity: MemberIdentity) -> MemberIdentity:
        async with aiosqlite.connect(self._database_path) as connection:
            await connection.execute(
                """
                INSERT INTO group_members (
                    group_id, user_id, nickname, card, role, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (group_id, user_id) DO UPDATE SET
                    nickname = COALESCE(excluded.nickname, group_members.nickname),
                    card = COALESCE(excluded.card, group_members.card),
                    role = COALESCE(excluded.role, group_members.role),
                    updated_at = excluded.updated_at
                WHERE excluded.updated_at >= group_members.updated_at
                """,
                (
                    identity.group_id,
                    identity.user_id,
                    identity.nickname,
                    identity.card,
                    identity.role,
                    identity.updated_at,
                ),
            )
            cursor = await connection.execute(
                """
                SELECT nickname, card, role, updated_at
                FROM group_members
                WHERE group_id = ? AND user_id = ?
                """,
                (identity.group_id, identity.user_id),
            )
            row = await cursor.fetchone()
            await connection.commit()

        if row is None:
            raise RuntimeError("identity upsert did not return a row")
        nickname, card, role, updated_at = row
        return MemberIdentity(
            group_id=identity.group_id,
            user_id=identity.user_id,
            nickname=nickname,
            card=card,
            role=role,
            updated_at=updated_at,
        )

    async def get_many(
        self,
        group_id: int,
        user_ids: Sequence[int],
    ) -> dict[int, MemberIdentity]:
        unique_user_ids = tuple(dict.fromkeys(user_ids))
        if not unique_user_ids:
            return {}

        placeholders = ", ".join("?" for _ in unique_user_ids)
        async with aiosqlite.connect(self._database_path) as connection:
            cursor = await connection.execute(
                f"""
                SELECT user_id, nickname, card, role, updated_at
                FROM group_members
                WHERE group_id = ? AND user_id IN ({placeholders})
                """,
                (group_id, *unique_user_ids),
            )
            rows = await cursor.fetchall()

        return {
            user_id: MemberIdentity(
                group_id=group_id,
                user_id=user_id,
                nickname=nickname,
                card=card,
                role=role,
                updated_at=updated_at,
            )
            for user_id, nickname, card, role, updated_at in rows
        }
