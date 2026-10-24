from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from .context import ContextBuilder
from .domain import ContextDocument


class AutopilotContextError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AutopilotContext:
    managed_paths: tuple[Path, ...]
    project_paths: tuple[Path, ...]
    protected_paths: tuple[Path, ...]
    unmatched_patterns: tuple[str, ...]
    content: str


class AutopilotContextService:
    MAX_PROJECT_CONTEXT_BYTES = 2_000_000

    def __init__(self, builder: ContextBuilder | None = None) -> None:
        self.builder = builder or ContextBuilder()

    def build(
        self,
        root: Path,
        controls: list[ContextDocument],
        patterns: tuple[str, ...],
        selected: tuple[Path, ...] | list[Path],
        protected: Callable[[Path], bool],
    ) -> AutopilotContext:
        root = root.resolve()
        resolved: set[Path] = set()
        unmatched: list[str] = []
        for pattern in (*patterns, *(path.as_posix() for path in selected)):
            self._validate_pattern(pattern)
            matches = self._matches(root, pattern)
            if not matches:
                unmatched.append(pattern)
            resolved.update(matches)
        paths = tuple(sorted(resolved, key=lambda path: path.as_posix()))
        total_bytes = 0
        for relative_path in paths:
            path = root / relative_path
            try:
                payload = path.read_bytes()
                payload.decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise AutopilotContextError(
                    f"Could not read Autopilot context path as UTF-8: {relative_path.as_posix()}"
                ) from exc
            total_bytes += len(payload)
            if total_bytes > self.MAX_PROJECT_CONTEXT_BYTES:
                raise AutopilotContextError(
                    "Autopilot project context exceeds the 2000000-byte limit at: "
                    + relative_path.as_posix()
                )
        try:
            content = self.builder.build(root, controls, list(paths))
        except (OSError, UnicodeDecodeError) as exc:
            path = getattr(exc, "filename", None) or "configured project context"
            raise AutopilotContextError(f"Could not read Autopilot context path: {path}") from exc
        managed = tuple(document.relative_path for document in controls)
        protected_paths = tuple(path for path in paths if protected(path))
        return AutopilotContext(managed, paths, protected_paths, tuple(dict.fromkeys(unmatched)), content)

    @staticmethod
    def _validate_pattern(pattern: str) -> None:
        candidate = PurePosixPath(pattern)
        if not pattern.strip() or candidate.is_absolute() or ".." in candidate.parts:
            raise AutopilotContextError(f"Unsafe Autopilot context path pattern: {pattern}")

    @staticmethod
    def _matches(root: Path, pattern: str) -> tuple[Path, ...]:
        matches: list[Path] = []
        try:
            # pathlib treats a terminal ``/**`` as directories only. Project
            # policy uses the familiar "everything below here" meaning.
            glob_pattern = pattern + "/*" if pattern.endswith("/**") else pattern
            candidates = root.glob(glob_pattern)
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                try:
                    relative = resolved.relative_to(root)
                except ValueError as exc:
                    raise AutopilotContextError(
                        f"Autopilot context path escapes the project root: {pattern}"
                    ) from exc
                matches.append(relative)
        except (OSError, ValueError) as exc:
            if isinstance(exc, AutopilotContextError):
                raise
            raise AutopilotContextError(f"Invalid Autopilot context pattern: {pattern}") from exc
        return tuple(sorted(set(matches), key=lambda path: path.as_posix()))
