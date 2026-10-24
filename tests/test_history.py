import json
from pathlib import Path

from ai_workbench.domain import Proposal, ProposedFile
from ai_workbench.history import HistoryService


def test_snapshot_records_existing_and_missing_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "existing.bin").write_bytes(b"before\x00contents")
    service = HistoryService(tmp_path / "history")
    proposal = Proposal((
        ProposedFile(Path("existing.bin"), "replacement"),
        ProposedFile(Path("new.txt"), "new"),
    ))

    transaction = service.snapshot(project, proposal)
    manifest = json.loads((transaction / "manifest.json").read_text(encoding="utf-8"))

    assert transaction.is_relative_to(tmp_path / "history")
    assert manifest["project_root"] == str(project.resolve())
    assert manifest["files"] == [
        {"path": "existing.bin", "existed": True, "snapshot": "0000.snapshot"},
        {"path": "new.txt", "existed": False, "snapshot": None},
    ]
    assert (transaction / "files/0000.snapshot").read_bytes() == b"before\x00contents"
