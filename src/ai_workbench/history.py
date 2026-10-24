from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .domain import Proposal
from .proposals import validate_relative_path

DEFAULT_HISTORY_DIR = Path.home() / ".local" / "share" / "ai-workbench" / "history"


class HistoryService:
    """Persists complete pre-transaction snapshots outside project repositories."""

    def __init__(self, history_root: Path = DEFAULT_HISTORY_DIR) -> None:
        self.history_root = history_root.expanduser()

    def snapshot(self, root: Path, proposal: Proposal) -> Path:
        resolved_root = root.resolve()
        project_id = hashlib.sha256(os.fsencode(resolved_root)).hexdigest()[:24]
        transaction_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}-{uuid.uuid4().hex}"
        project_history = self.history_root / project_id
        project_history.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=project_history))
        final = project_history / transaction_id
        records: list[dict[str, object]] = []
        try:
            snapshots = temporary / "files"
            snapshots.mkdir()
            for index, proposed in enumerate(proposal.files):
                relative = validate_relative_path(resolved_root, proposed.relative_path.as_posix())
                target = resolved_root / relative
                existed = target.is_file()
                snapshot_name = f"{index:04d}.snapshot" if existed else None
                if existed and snapshot_name is not None:
                    (snapshots / snapshot_name).write_bytes(target.read_bytes())
                records.append({
                    "path": relative.as_posix(),
                    "existed": existed,
                    "snapshot": snapshot_name,
                })
            manifest = {
                "version": 1,
                "project_root": str(resolved_root),
                "transaction_id": transaction_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": records,
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, final)
            return final
        except Exception:
            for child in sorted(temporary.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            temporary.rmdir()
            raise

    def restore(self, snapshot: Path) -> Path:
        """Restore every file in one recorded transaction without deleting its audit record."""
        resolved_snapshot = snapshot.resolve()
        try:
            resolved_snapshot.relative_to(self.history_root.resolve())
        except ValueError as exc:
            raise OSError("Snapshot is outside the configured history directory.") from exc
        manifest_path = resolved_snapshot / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            root = Path(manifest["project_root"]).resolve()
            records = manifest["files"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise OSError(f"Invalid history snapshot: {resolved_snapshot}") from exc
        if not isinstance(records, list):
            raise OSError(f"Invalid history snapshot: {resolved_snapshot}")

        restore_files: list[tuple[Path, bytes]] = []
        remove_files: list[Path] = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise OSError(f"Invalid history snapshot record: {resolved_snapshot}")
            relative = validate_relative_path(root, record["path"])
            target = root / relative
            existed = record.get("existed")
            snapshot_name = record.get("snapshot")
            if existed is True and isinstance(snapshot_name, str):
                if Path(snapshot_name).name != snapshot_name:
                    raise OSError(f"Invalid history file snapshot name: {resolved_snapshot}")
                source = resolved_snapshot / "files" / snapshot_name
                if not source.is_file():
                    raise OSError(f"Missing history file snapshot: {source}")
                restore_files.append((target, source.read_bytes()))
            elif existed is False and snapshot_name is None:
                if target.exists() and not target.is_file():
                    raise OSError(f"Cannot remove non-file rollback target: {target}")
                remove_files.append(target)
            else:
                raise OSError(f"Invalid history snapshot record: {resolved_snapshot}")

        staged: list[tuple[Path, Path]] = []
        try:
            for target, content in restore_files:
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{target.name}.restore.", dir=target.parent
                )
                temporary = Path(temporary_name)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                staged.append((temporary, target))
            for temporary, target in staged:
                os.replace(temporary, target)
            for target in remove_files:
                target.unlink(missing_ok=True)
        finally:
            for temporary, _target in staged:
                temporary.unlink(missing_ok=True)
        return root
