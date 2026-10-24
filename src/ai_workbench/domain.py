from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProposedFile:
    relative_path: Path
    content: str


@dataclass(frozen=True, slots=True)
class Proposal:
    files: tuple[ProposedFile, ...]


@dataclass(frozen=True, slots=True)
class ContextDocument:
    label: str
    relative_path: Path
    content: str
