"""把 apply 造成的改名寫進 GitHub Gitless Sync 的清單。

Gitless Sync 只會上傳 ``github-sync-metadata.json`` 裡列到的路徑，
不會在啟動時掃描整個庫。Obsidian 關著時改名，外掛看不到 create/delete，
之後再開庫、按同步，新圖仍不會上傳。因此 apply 必須自己更新這份清單。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blogseo.seo.applier import ApplyPlan

_METADATA_RELATIVE = Path(".obsidian") / "github-sync-metadata.json"


@dataclass
class GitlessSyncOutcome:
    """一次 metadata 更新的結果，給 CLI 顯示用。"""

    metadata_path: Path | None = None
    marked_deleted: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    marked_dirty: list[str] = field(default_factory=list)
    warning: str | None = None


def find_gitless_metadata(start: Path) -> Path | None:
    """從文章路徑往上找 ``.obsidian/github-sync-metadata.json``。

    Args:
        start: 文章檔或其所在目錄。

    Returns:
        metadata 路徑；找不到則 ``None``。
    """
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / _METADATA_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def _vault_relative(vault: Path, path: Path) -> str:
    return path.resolve().relative_to(vault.resolve()).as_posix()


def _mark_deleted(files: dict[str, Any], relative: str, now: int) -> bool:
    entry = files.get(relative)
    if not isinstance(entry, dict):
        return False
    if entry.get("deleted") is True:
        return False
    if not entry.get("sha"):
        # 遠端樹裡本來就沒有這個檔，標刪除會讓外掛對 undefined 寫 sha。
        files.pop(relative, None)
        return False
    entry["deleted"] = True
    entry["deletedAt"] = now
    entry["dirty"] = True
    entry["justDownloaded"] = False
    return True


def _mark_new_or_dirty(files: dict[str, Any], relative: str, now: int) -> str:
    entry = files.get(relative)
    if not isinstance(entry, dict):
        files[relative] = {
            "path": relative,
            "sha": None,
            "dirty": True,
            "justDownloaded": False,
            "lastModified": now,
        }
        return "added"
    entry["dirty"] = True
    entry["justDownloaded"] = False
    entry["lastModified"] = now
    entry.pop("deleted", None)
    entry.pop("deletedAt", None)
    if entry.get("sha") is None:
        return "added"
    return "dirty"


def patch_gitless_sync_metadata(plan: ApplyPlan) -> GitlessSyncOutcome:
    """依 apply 計畫更新 Gitless Sync metadata。

    找不到清單時靜默略過。清單損壞時不中斷 apply，只回傳警告。

    Args:
        plan: 已經寫入磁碟的套用計畫。

    Returns:
        更新結果；庫裡沒有 Gitless Sync 時 ``metadata_path`` 為 ``None``。
    """
    metadata_path = find_gitless_metadata(plan.article_path)
    if metadata_path is None:
        return GitlessSyncOutcome()

    try:
        raw = metadata_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        return GitlessSyncOutcome(
            metadata_path=metadata_path,
            warning=f"讀不到 Gitless Sync 清單，未更新：{exc}",
        )

    if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
        return GitlessSyncOutcome(
            metadata_path=metadata_path,
            warning="Gitless Sync 清單格式不符，未更新",
        )

    files: dict[str, Any] = payload["files"]
    vault = metadata_path.parent.parent
    now = int(time.time() * 1000)
    outcome = GitlessSyncOutcome(metadata_path=metadata_path)

    for rename in plan.renames:
        old_rel = _vault_relative(vault, rename.source)
        new_rel = _vault_relative(vault, rename.dest)
        if _mark_deleted(files, old_rel, now):
            outcome.marked_deleted.append(old_rel)
        kind = _mark_new_or_dirty(files, new_rel, now)
        if kind == "added":
            outcome.added.append(new_rel)
        else:
            outcome.marked_dirty.append(new_rel)

    article_rel = _vault_relative(vault, plan.article_path)
    if plan.new_text != plan.original_text or plan.renames:
        kind = _mark_new_or_dirty(files, article_rel, now)
        if kind == "added":
            outcome.added.append(article_rel)
        elif article_rel not in outcome.marked_dirty:
            outcome.marked_dirty.append(article_rel)

    if not (outcome.marked_deleted or outcome.added or outcome.marked_dirty):
        return outcome

    try:
        _atomic_write_json(metadata_path, payload)
    except OSError as exc:
        outcome.warning = f"無法寫入 Gitless Sync 清單：{exc}"
    return outcome


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    temp = path.with_name(f"{path.stem}.blogseo-new{path.suffix}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temp.replace(path)
