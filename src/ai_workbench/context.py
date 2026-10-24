from pathlib import Path

from .domain import ContextDocument


class ContextBuilder:
    def build(self, root: Path, controls: list[ContextDocument], selected: list[Path]) -> str:
        sections: list[str] = []
        for document in controls:
            sections.append(self._section(document.label, document.relative_path, document.content))
        for relative_path in sorted(selected):
            content = (root / relative_path).read_text(encoding="utf-8")
            sections.append(self._section("Selected source file", relative_path, content))
        return "\n\n".join(sections)

    @staticmethod
    def _section(label: str, path: Path, content: str) -> str:
        return f"===== {label}: {path.as_posix()} =====\n{content.rstrip()}\n===== END {path.as_posix()} ====="
