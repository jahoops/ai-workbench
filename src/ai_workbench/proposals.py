from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from .domain import Proposal, ProposedFile

BLOCK = re.compile(r"^<<<FILE ([^\r\n]+)>>>\r?\n(.*?)^<<<END FILE>>>[ \t]*$", re.MULTILINE | re.DOTALL)


class ProposalError(ValueError):
    pass


def validate_relative_path(root: Path, raw_path: str) -> Path:
    if "\\" in raw_path:
        raise ProposalError(f"Backslashes are not allowed in proposal paths: {raw_path}")
    pure = PurePosixPath(raw_path)
    if not raw_path.strip() or pure.is_absolute() or ".." in pure.parts or pure == PurePosixPath("."):
        raise ProposalError(f"Unsafe proposal path: {raw_path}")
    relative = Path(*pure.parts)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ProposalError(f"Path escapes project root: {raw_path}") from exc
    return relative


def parse_proposal(root: Path, response: str) -> Proposal:
    matches = list(BLOCK.finditer(response))
    if not matches:
        raise ProposalError("No complete <<<FILE path>>> ... <<<END FILE>>> blocks found.")

    residue = BLOCK.sub("", response).strip()
    if residue:
        raise ProposalError("Response contains text outside proposal blocks.")

    files: list[ProposedFile] = []
    seen: set[Path] = set()
    for match in matches:
        relative = validate_relative_path(root, match.group(1).strip())
        if relative in seen:
            raise ProposalError(f"Duplicate proposed path: {relative.as_posix()}")
        seen.add(relative)
        files.append(ProposedFile(relative, match.group(2)))
    return Proposal(tuple(files))
