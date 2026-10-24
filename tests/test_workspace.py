from pathlib import Path

from ai_workbench.domain import Proposal, ProposedFile
from ai_workbench.history import HistoryService
from ai_workbench.workspace import WorkspaceService


def test_accept_writes_new_and_existing_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "existing.txt").write_text("old", encoding="utf-8")
    proposal = Proposal((
        ProposedFile(Path("existing.txt"), "new"),
        ProposedFile(Path("nested/new.txt"), "created"),
    ))

    WorkspaceService(HistoryService(tmp_path / "outside-history")).accept(project, proposal)

    assert (project / "existing.txt").read_text(encoding="utf-8") == "new"
    assert (project / "nested/new.txt").read_text(encoding="utf-8") == "created"
