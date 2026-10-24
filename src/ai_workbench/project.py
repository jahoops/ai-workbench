from __future__ import annotations

from pathlib import Path
import re
from dataclasses import dataclass

from .domain import ContextDocument

CONTROL_FILES = (
    ("Project context", Path(".ai/context.md")),
    ("Project memory", Path(".ai/memory.md")),
    ("Next steps", Path(".ai/next-steps.md")),
    ("Project goal", Path(".ai/goal.md")),
)
CONTROL_PATHS = frozenset(path for _label, path in CONTROL_FILES)
EXCLUDED_PARTS = {".git", ".ai", ".venv", "venv", "__pycache__", ".pytest_cache"}


class ProjectError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NextStepState:
    current: str | None
    after: str | None = None
    complete: bool = False

    @property
    def display(self) -> str:
        if self.complete:
            return "CURRENT\nNone\n\nSTATUS\nProject complete"
        lines = ["CURRENT", self.current or "None"]
        if self.after:
            lines.extend(["", "AFTER", self.after])
        return "\n".join(lines)


class ProjectService:
    def open(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise ProjectError(f"Not a project directory: {path}")
        return path

    def load_control_documents(self, root: Path) -> list[ContextDocument]:
        documents: list[ContextDocument] = []
        for label, relative_path in CONTROL_FILES:
            path = root / relative_path
            if path.is_file():
                documents.append(ContextDocument(label, relative_path, path.read_text(encoding="utf-8")))
        return documents

    def current_next_step(self, root: Path) -> str | None:
        return self.next_step_state(root).current

    def next_step_state(self, root: Path) -> NextStepState:
        path = root / ".ai/next-steps.md"
        if not path.is_file():
            return NextStepState(None)
        lines = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        fields: dict[str, list[str]] = {"current": [], "after": []}
        active: str | None = None
        structured = False
        unstructured: list[str] = []
        for line in lines:
            match = re.match(r"^\s*(CURRENT|AFTER)\s*:?[ \t]*(.*)$", line, re.IGNORECASE)
            if match:
                active = match.group(1).casefold()
                structured = True
                if match.group(2).strip():
                    fields[active].append(match.group(2).strip())
            elif structured and active:
                fields[active].append(line.strip())
            else:
                unstructured.append(line.strip())
        current_text = "\n".join(fields["current"] if structured else unstructured).strip()
        after = "\n".join(fields["after"]).strip() or None
        current = self.actionable_next_step(current_text)
        complete = bool(current_text) and current is None and self._is_completion_marker(
            self._normalize_step(current_text)
        )
        return NextStepState(current, after, complete)

    @staticmethod
    def actionable_next_step(text: str) -> str | None:
        task = text.strip()
        if not task:
            return None
        normalized = ProjectService._normalize_step(task)
        if ProjectService._is_completion_marker(normalized):
            return None
        return task

    def next_step_is_complete(self, root: Path) -> bool:
        return self.next_step_state(root).complete

    @staticmethod
    def _normalize_step(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()

    @staticmethod
    def _is_completion_marker(normalized: str) -> bool:
        return normalized in {
            "none",
            "done",
            "complete",
            "completed",
            "project complete",
            "project completed",
            "done project complete",
            "done project completed",
        }

    def selectable_files(self, root: Path, limit: int | None = None) -> list[Path]:
        results: list[Path] = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part in EXCLUDED_PARTS for part in relative.parts) or not path.is_file():
                continue
            try:
                path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            results.append(relative)
            if limit is not None and len(results) >= limit:
                break
        return results
