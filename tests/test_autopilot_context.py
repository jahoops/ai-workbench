from pathlib import Path

import pytest

from ai_workbench.autopilot import ProtectedPathPolicy
from ai_workbench.autopilot_context import AutopilotContextError, AutopilotContextService
from ai_workbench.project import ProjectService


def test_context_paths_resolve_globs_deterministically_and_deduplicate(tmp_path: Path) -> None:
    (tmp_path / ".ai").mkdir()
    (tmp_path / ".ai/context.md").write_text("managed\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/z.py").write_text("z = 1\n", encoding="utf-8")
    (tmp_path / "src/a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("secret = False\n", encoding="utf-8")
    controls = ProjectService().load_control_documents(tmp_path)

    context = AutopilotContextService().build(
        tmp_path,
        controls,
        ("src/*.py", "src/a.py", "future.py"),
        (),
        ProtectedPathPolicy(("src/z.py",)).is_protected,
    )

    assert context.project_paths == (Path("src/a.py"), Path("src/z.py"))
    assert context.protected_paths == (Path("src/z.py"),)
    assert context.unmatched_patterns == ("future.py",)
    assert "a = 1" in context.content and "z = 1" in context.content
    assert "secret = False" not in context.content


@pytest.mark.parametrize("pattern", ("../outside.py", "/tmp/outside.py"))
def test_context_path_traversal_is_rejected(tmp_path: Path, pattern: str) -> None:
    with pytest.raises(AutopilotContextError, match="Unsafe"):
        AutopilotContextService().build(tmp_path, [], (pattern,), (), lambda _path: False)


def test_context_path_symlink_escape_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("outside\n", encoding="utf-8")
    (tmp_path / "linked.py").symlink_to(outside)

    with pytest.raises(AutopilotContextError, match="escapes"):
        AutopilotContextService().build(
            tmp_path, [], ("linked.py",), (), lambda _path: False
        )
