from pathlib import Path

import pytest

from ai_workbench.proposals import ProposalError, parse_proposal


def test_parses_multiple_files(tmp_path: Path) -> None:
    response = "<<<FILE one.py>>>\none\n<<<END FILE>>>\n<<<FILE lib/two.py>>>\ntwo\n<<<END FILE>>>"
    proposal = parse_proposal(tmp_path, response)
    assert [item.relative_path.as_posix() for item in proposal.files] == ["one.py", "lib/two.py"]


@pytest.mark.parametrize("path", ["../secret", "/tmp/secret", "a/../../secret", "a\\b.py"])
def test_rejects_unsafe_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(ProposalError):
        parse_proposal(tmp_path, f"<<<FILE {path}>>>\nx\n<<<END FILE>>>")


def test_rejects_text_outside_blocks(tmp_path: Path) -> None:
    with pytest.raises(ProposalError):
        parse_proposal(tmp_path, "commentary\n<<<FILE one.py>>>\none\n<<<END FILE>>>")


def test_rejects_duplicate_paths(tmp_path: Path) -> None:
    response = "<<<FILE one.py>>>\na\n<<<END FILE>>>\n<<<FILE one.py>>>\nb\n<<<END FILE>>>"
    with pytest.raises(ProposalError):
        parse_proposal(tmp_path, response)
