import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import flet as ft

from ai_workbench.app import WorkbenchUI
from ai_workbench.config import ConfigurationService, UserSettings


class FakePage:
    def __init__(self) -> None:
        self.title = ""
        self.padding = 0
        self.controls: list[object] = []
        self.services: list[object] = []
        self.update_count = 0

    def add(self, *controls: object) -> None:
        self.controls.extend(controls)

    def update(self) -> None:
        self.update_count += 1


def test_startup_restores_valid_project_model_files_and_context(tmp_path: Path) -> None:
    project = tmp_path / "remembered-project"
    control_dir = project / ".ai"
    control_dir.mkdir(parents=True)
    (control_dir / "context.md").write_text("remembered context", encoding="utf-8")
    (project / "visible.py").write_text("print('visible')\n", encoding="utf-8")
    configuration = ConfigurationService(tmp_path / "user-config")
    configuration.ensure_user_files()
    configuration.save_settings(UserSettings(
        last_project=str(project),
        last_model="deepseek-v4-flash",
        recent_projects=(str(project),),
    ))
    page = FakePage()

    ui = WorkbenchUI(page, configuration=configuration)  # type: ignore[arg-type]
    ui.build()

    assert ui.root == project.resolve()
    assert ui.model_selector.value == "deepseek-v4-flash"
    assert [control.label for control in ui.files.controls] == ["visible.py    Writable"]
    assert "remembered context" in (ui.context_preview.value or "")
    assert configuration.load_settings().last_project == str(project.resolve())


def test_transcript_expands_between_header_and_fixed_composer(tmp_path: Path) -> None:
    page = FakePage()
    ui = WorkbenchUI(
        page, configuration=ConfigurationService(tmp_path / "user-config")
    )  # type: ignore[arg-type]

    ui.build()

    assert page.controls == [ui.layout]
    transcript_index = ui.layout.controls.index(ui.transcript)
    composer = ui.layout.controls[1]
    assert isinstance(composer, ft.Column)
    assert composer.controls[0] is ui.prompt
    action_row = composer.controls[1]
    assert isinstance(action_row, ft.Row)
    assert action_row.controls[:3] == [ui.ask_button, ui.change_button, ui.continue_button]
    assert transcript_index > ui.layout.controls.index(composer)
    assert ui.transcript.expand is True
    assert ui.context_preview.max_lines == 4
    assert page.scroll is None


def test_autopilot_actions_are_above_variable_details_and_sections_are_distinct(
    tmp_path: Path,
) -> None:
    ui = WorkbenchUI(
        FakePage(), configuration=ConfigurationService(tmp_path / "user-config")
    )  # type: ignore[arg-type]

    ui.build()

    controls = ui.autopilot_panel.content.controls
    assert isinstance(controls[2], ft.Row)
    assert controls[2].controls == [ui.autopilot_button, ui.autopilot_run_button]
    assert isinstance(controls[3], ft.Row)
    assert controls[3].controls[:2] == [ui.autopilot_pause_button, ui.autopilot_stop_button]
    assert ui.autopilot_button.visible is True
    assert ui.autopilot_run_button.visible is True
    assert ui.project_context_panel.bgcolor != ui.autopilot_panel.bgcolor
    assert ui.prompt.expand is True


def test_missing_remembered_project_fails_harmlessly(tmp_path: Path) -> None:
    missing = tmp_path / "gone"
    configuration = ConfigurationService(tmp_path / "user-config")
    configuration.ensure_user_files()
    configuration.save_settings(UserSettings(last_project=str(missing)))
    page = FakePage()

    ui = WorkbenchUI(page, configuration=configuration)  # type: ignore[arg-type]
    ui.build()

    assert ui.root is None
    assert ui.path.value in (None, "")
    assert "no longer available" in (ui.status.value or "")


def test_open_project_shows_autopilot_readiness(tmp_path: Path) -> None:
    project = tmp_path / "ready"
    control = project / ".ai"
    control.mkdir(parents=True)
    (control / "goal.md").write_text("Ship the project.\n", encoding="utf-8")
    (control / "next-steps.md").write_text("Implement the next step.\n", encoding="utf-8")
    (control / "project.yml").write_text(
        "verify:\n  commands:\n    - [pytest, -q]\n"
        "autopilot:\n"
        "  max_steps: 25\n"
        "  max_consecutive_failures: 4\n"
        "  protected_paths:\n    - .ai/goal.md\n"
        "  verification_contract_paths:\n    - tests/**\n",
        encoding="utf-8",
    )
    ui = WorkbenchUI(
        FakePage(), configuration=ConfigurationService(tmp_path / "user-config")
    )  # type: ignore[arg-type]
    ui.build()

    assert ui.open_project_path(str(project)) is True

    assert ui.autopilot_status.value.startswith("STATUS: READY — WARNING:")
    assert ui.autopilot_button.disabled is False
    assert ui.autopilot_run_button.disabled is False

    (control / "next-steps.md").write_text("# Next Steps\n\nDONE\n", encoding="utf-8")
    ui.refresh_autopilot_status()

    assert ui.autopilot_status.value.startswith("STATUS: NOT READY")
    assert ui.autopilot_button.disabled is True
    assert ui.autopilot_run_button.disabled is True
    assert ui.autopilot_button.bgcolor == "#334155"
    assert ui.autopilot_run_button.bgcolor == "#334155"

    (control / "next-steps.md").write_text("CURRENT\nImplement it.\n", encoding="utf-8")
    ui.refresh_autopilot_status()

    assert ui.autopilot_button.disabled is False
    assert ui.autopilot_run_button.disabled is False


def test_open_project_updates_and_caps_recent_projects(tmp_path: Path) -> None:
    configuration = ConfigurationService(tmp_path / "user-config")
    configuration.ensure_user_files()
    old_projects = tuple(str(tmp_path / f"old-{number}") for number in range(8))
    configuration.save_settings(UserSettings(recent_projects=old_projects))
    project = tmp_path / "current"
    project.mkdir()
    ui = WorkbenchUI(FakePage(), configuration=configuration)  # type: ignore[arg-type]
    ui.build()

    assert ui.open_project_path(str(project)) is True

    settings = configuration.load_settings()
    assert settings.last_project == str(project.resolve())
    assert settings.recent_projects == (str(project.resolve()), *old_projects[:7])


def test_model_change_is_saved_automatically(tmp_path: Path) -> None:
    configuration = ConfigurationService(tmp_path / "user-config")
    configuration.ensure_user_files()
    ui = WorkbenchUI(FakePage(), configuration=configuration)  # type: ignore[arg-type]
    ui.build()

    ui.model_selector.value = "deepseek-v4-flash"
    ui.model_changed(None)  # type: ignore[arg-type]

    assert configuration.load_settings().last_model == "deepseek-v4-flash"


def test_folder_picker_uses_the_shared_project_open_path(tmp_path: Path) -> None:
    configuration = ConfigurationService(tmp_path / "user-config")
    configuration.ensure_user_files()
    project = tmp_path / "picked"
    project.mkdir()
    page = FakePage()
    ui = WorkbenchUI(page, configuration=configuration)  # type: ignore[arg-type]
    ui.build()
    ui.file_picker.get_directory_path = AsyncMock(return_value=str(project))  # type: ignore[method-assign]

    asyncio.run(ui.browse_project(None))  # type: ignore[arg-type]

    ui.file_picker.get_directory_path.assert_awaited_once_with(  # type: ignore[attr-defined]
        dialog_title="Open AI Workbench project",
        initial_directory=None,
    )
    assert ui.root == project.resolve()
    assert configuration.load_settings().last_project == str(project.resolve())
    assert ui.file_picker in page.services
