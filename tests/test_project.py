from pathlib import Path

import pytest

from ai_workbench.project import ProjectService


def test_loads_ai_control_documents_and_excludes_them_from_selectable_files(tmp_path: Path) -> None:
    control_contents = {
        "context.md": "project context",
        "memory.md": "durable memory",
        "next-steps.md": "current next step",
    }
    control_dir = tmp_path / ".ai"
    control_dir.mkdir()
    for name, content in control_contents.items():
        (control_dir / name).write_text(content, encoding="utf-8")
    (control_dir / "internal.txt").write_text("also hidden", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('visible')\n", encoding="utf-8")

    service = ProjectService()
    documents = service.load_control_documents(tmp_path)
    selectable = service.selectable_files(tmp_path)

    assert [document.relative_path.as_posix() for document in documents] == [
        ".ai/context.md",
        ".ai/memory.md",
        ".ai/next-steps.md",
    ]
    assert [document.content for document in documents] == list(control_contents.values())
    assert selectable == [Path("main.py")]


def test_goal_is_automatic_context_control_and_not_selectable(tmp_path: Path) -> None:
    control_dir = tmp_path / ".ai"
    control_dir.mkdir()
    (control_dir / "goal.md").write_text("Ship when acceptance tests pass.", encoding="utf-8")

    service = ProjectService()
    documents = service.load_control_documents(tmp_path)

    assert [(item.relative_path.as_posix(), item.content) for item in documents] == [
        (".ai/goal.md", "Ship when acceptance tests pass.")
    ]
    assert service.selectable_files(tmp_path) == []


def test_current_next_step_reads_actionable_content(tmp_path: Path) -> None:
    control_dir = tmp_path / ".ai"
    control_dir.mkdir()
    (control_dir / "next-steps.md").write_text(
        "# Next step\n\nImplement the prompt workflow.\nThen run tests.\n",
        encoding="utf-8",
    )

    assert ProjectService().current_next_step(tmp_path) == (
        "Implement the prompt workflow.\nThen run tests."
    )


def test_current_next_step_returns_none_when_missing_or_only_a_heading(tmp_path: Path) -> None:
    service = ProjectService()
    assert service.current_next_step(tmp_path) is None

    control_dir = tmp_path / ".ai"
    control_dir.mkdir()
    (control_dir / "next-steps.md").write_text("# Next step\n\n## Queue\n", encoding="utf-8")
    assert service.current_next_step(tmp_path) is None


def test_structured_next_step_state_has_current_and_after_without_markdown_heading(
    tmp_path: Path,
) -> None:
    control = tmp_path / ".ai"
    control.mkdir()
    (control / "next-steps.md").write_text(
        "# Next Steps\n\nCURRENT\nAdd --exclude WORD.\n\nAFTER\nDONE\n",
        encoding="utf-8",
    )

    state = ProjectService().next_step_state(tmp_path)

    assert state.current == "Add --exclude WORD."
    assert state.after == "DONE"
    assert state.complete is False
    assert state.display == "CURRENT\nAdd --exclude WORD.\n\nAFTER\nDONE"
    assert "# Next Steps" not in state.display


def test_structured_current_without_after_displays_only_current(tmp_path: Path) -> None:
    control = tmp_path / ".ai"
    control.mkdir()
    (control / "next-steps.md").write_text(
        "# Plan\nCURRENT: Implement JSON output.\n", encoding="utf-8"
    )

    state = ProjectService().next_step_state(tmp_path)

    assert state.current == "Implement JSON output."
    assert state.after is None
    assert state.display == "CURRENT\nImplement JSON output."


@pytest.mark.parametrize("marker", ("DONE", "Project complete", "None"))
def test_completion_markers_share_canonical_plan_interpretation(
    tmp_path: Path,
    marker: str,
) -> None:
    control = tmp_path / ".ai"
    control.mkdir()
    (control / "next-steps.md").write_text(
        f"# Next Steps\n\n{marker}\n", encoding="utf-8"
    )

    service = ProjectService()
    state = service.next_step_state(tmp_path)

    assert state.current is None
    assert state.complete is True
    assert state.display == "CURRENT\nNone\n\nSTATUS\nProject complete"
    assert service.current_next_step(tmp_path) is None
    assert service.next_step_is_complete(tmp_path) is True
