from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .domain import Proposal
from .history import HistoryService
from .proposals import validate_relative_path


class WorkspaceService:
    def __init__(self, history: HistoryService | None = None) -> None:
        self.history = history or HistoryService()

    def accept(self, root: Path, proposal: Proposal) -> Path:
        targets: list[tuple[Path, str]] = []
        for proposed in proposal.files:
            relative = validate_relative_path(root, proposed.relative_path.as_posix())
            targets.append((root / relative, proposed.content))

        snapshot = self.history.snapshot(root, proposal)

        staged: list[tuple[Path, Path]] = []
        try:
            for target, content in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
                temporary = Path(temporary_name)
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                staged.append((temporary, target))
            for temporary, target in staged:
                os.replace(temporary, target)
        finally:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
        return snapshot

    def restore(self, snapshot: Path) -> Path:
        return self.history.restore(snapshot)
