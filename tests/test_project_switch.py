from pathlib import Path

import flet as ft

from ai_workbench.app import WorkbenchUI


def test_replacing_project_files_clears_old_controls_and_selection(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    ui = object.__new__(WorkbenchUI)
    ui.files = ft.Column()
    original_controls = ui.files.controls

    ui.replace_file_controls(first, [Path("old.py"), Path("shared.py")])
    ui.files.controls[0].value = True
    ui.replace_file_controls(second, [Path("new.py"), Path("shared.py")])

    assert ui.files.controls is original_controls
    assert [control.label for control in ui.files.controls] == [
        "new.py    Writable", "shared.py    Writable"
    ]
    assert [control.data for control in ui.files.controls] == [Path("new.py"), Path("shared.py")]
    assert all(control.value is False for control in ui.files.controls)
    assert all(str(second) in str(control.key) for control in ui.files.controls)
    assert ui.selected_files() == []
