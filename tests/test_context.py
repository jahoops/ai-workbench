from pathlib import Path

from ai_workbench.context import ContextBuilder
from ai_workbench.domain import ContextDocument


def test_context_contains_controls_and_selected_files(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    controls = [ContextDocument("Project context", Path(".ai/context.md"), "Rules")]

    result = ContextBuilder().build(tmp_path, controls, [Path("main.py")])

    assert "Project context: .ai/context.md" in result
    assert "Selected source file: main.py" in result
    assert "print('hello')" in result
