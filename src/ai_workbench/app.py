from __future__ import annotations

import asyncio
import difflib
import logging
from dataclasses import dataclass
from pathlib import Path

import flet as ft

from .autopilot import AutopilotReadinessService, ProtectedPathPolicy
from .autopilot_context import AutopilotContextError
from .autopilot_loop import (
    AutopilotLoopEvent,
    AutopilotLoopRunner,
    AutopilotRunResult,
    AutopilotRunStatus,
)
from .autopilot_runner import AutopilotRunner, AutopilotStepResult, AutopilotStepStatus
from .config import ConfigurationError, ConfigurationService, ModelConfiguration, UserSettings
from .context import ContextBuilder
from .domain import Proposal
from .models import FakeModelAdapter, ModelAdapter, ModelRequestError, OpenAICompatibleAdapter
from .project import CONTROL_PATHS, ProjectService
from .project_config import ProjectConfigurationError, ProjectConfigurationService
from .proposals import ProposalError, parse_proposal
from .workspace import WorkspaceService
from .workflows import PromptWorkflow

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProposalTransaction:
    root: Path
    proposal: Proposal
    container: ft.Container
    status: ft.Text
    accept_button: ft.Button
    reject_button: ft.OutlinedButton
    file_contents: dict[Path, str]
    diffs: dict[Path, str]
    resolved: bool = False


class WorkbenchUI:
    def __init__(
        self,
        page: ft.Page,
        adapter: ModelAdapter | None = None,
        configuration: ConfigurationService | None = None,
    ) -> None:
        self.page = page
        self.fake_adapter = adapter or FakeModelAdapter()
        self.configuration = configuration or ConfigurationService()
        self.models: dict[str, ModelConfiguration] = {}
        self.settings = UserSettings()
        self.configuration_error: str | None = None
        try:
            self.configuration.ensure_user_files()
            self.models = self.configuration.load_models()
            self.settings = self.configuration.load_settings()
        except ConfigurationError as exc:
            self.configuration_error = str(exc)
        self.projects = ProjectService()
        self.context_builder = ContextBuilder()
        self.workspace = WorkspaceService()
        self.workflow = PromptWorkflow()
        self.autopilot_readiness = AutopilotReadinessService(projects=self.projects)
        self.project_configuration = ProjectConfigurationService()
        self.autopilot_runner = AutopilotRunner(
            projects=self.projects,
            readiness=self.autopilot_readiness,
            context_builder=self.context_builder,
            workflow=self.workflow,
            workspace=self.workspace,
        )
        self.autopilot_loop = AutopilotLoopRunner(
            self.autopilot_runner,
            projects=self.projects,
        )
        self.root: Path | None = None
        self.controls = []
        self.proposal: Proposal | None = None
        self.transactions: list[ProposalTransaction] = []
        self.busy = False
        self.autopilot_ready = False

        self.path = ft.TextField(label="Project directory", expand=True)
        model_names = ["fake", *self.models]
        selected_model = self.settings.last_model if self.settings.last_model in model_names else model_names[0]
        self.model_selector = ft.Dropdown(
            label="Model configuration",
            value=selected_model,
            options=[ft.DropdownOption(name) for name in model_names],
            on_select=self.model_changed,
            width=300,
        )
        self.model_status = ft.Text()
        self.autopilot_status = ft.Text("STATUS: NO PROJECT", weight=ft.FontWeight.BOLD)
        self.recent_projects = ft.Dropdown(
            label="Recent projects",
            options=[],
            on_select=self.recent_project_selected,
            width=300,
        )
        self.file_picker = ft.FilePicker()
        self.files = ft.ListView(
            scroll=ft.ScrollMode.ALWAYS, height=130, spacing=2, expand=True
        )
        self.context_summary = ft.Text(
            "0 project files selected · 0 writable · 0 protected · 0 managed AI files",
            weight=ft.FontWeight.BOLD,
            color="#7DD3FC",
        )
        self.project_files_heading = ft.Text(
            "Project files    0 selected / 0 files", weight=ft.FontWeight.BOLD
        )
        self.autopilot_goal = ft.Text("No project open", max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
        self.autopilot_plan = ft.Text("No current plan", max_lines=6, overflow=ft.TextOverflow.ELLIPSIS)
        self.autopilot_model = ft.Text("fake")
        self.autopilot_context_summary = ft.Text("0 files")
        self.autopilot_verification = ft.Text("Not configured")
        self.context_preview = ft.TextField(
            label="Assembled context",
            multiline=True,
            min_lines=3,
            max_lines=4,
            read_only=True,
        )
        self.prompt = ft.TextField(
            label="Prompt",
            hint_text="Ask about the project or describe a change",
            multiline=True,
            min_lines=4,
            max_lines=6,
            expand=True,
            bgcolor="#111A24",
            border_color="#31506B",
            focused_border_color="#38BDF8",
        )
        self.transcript = ft.ListView(
            expand=True,
            spacing=12,
            padding=10,
            scroll=ft.ScrollMode.AUTO,
            auto_scroll=False,
        )
        self.status = ft.Text()
        self.ask_button = ft.Button(
            "ASK", on_click=self.ask, bgcolor="#155E75", color=ft.Colors.WHITE
        )
        self.change_button = ft.Button(
            "CHANGE", on_click=self.change, bgcolor="#1D4ED8", color=ft.Colors.WHITE
        )
        self.continue_button = ft.Button(
            "CONTINUE", on_click=self.continue_next_step,
            bgcolor="#6D28D9", color=ft.Colors.WHITE,
        )
        self.autopilot_button = ft.Button(
            "Run One Step",
            on_click=self.run_one_autopilot_step,
            visible=True,
            disabled=True,
            bgcolor="#334155",
            color="#94A3B8",
        )
        self.autopilot_run_button = ft.Button(
            "Run Autopilot",
            on_click=self.run_autopilot,
            visible=True,
            disabled=True,
            bgcolor="#334155",
            color="#94A3B8",
        )
        self.autopilot_pause_button = ft.OutlinedButton(
            "Pause",
            on_click=self.pause_autopilot,
            visible=True,
            disabled=True,
        )
        self.autopilot_stop_button = ft.OutlinedButton(
            "Stop",
            on_click=self.stop_autopilot,
            visible=True,
            disabled=True,
        )
        self.progress = ft.ProgressRing(visible=False, width=20, height=20)

    def build(self) -> None:
        self.page.title = "AI Workbench"
        self.page.padding = 20
        self.page.scroll = None
        self.page.bgcolor = "#0B1118"
        # FilePicker is a Service in current Flet releases and must be mounted
        # on the page before one of its native dialogs can be invoked.
        if self.file_picker not in self.page.services:
            self.page.services.append(self.file_picker)
        header = ft.Column([
            ft.Row([
                ft.Text("AI Workbench", size=24, weight=ft.FontWeight.BOLD),
                self.model_selector,
                self.model_status,
            ], wrap=True),
            ft.Row([
                self.path,
                ft.OutlinedButton("Browse / Open Folder", on_click=self.browse_project),
                ft.Button("Open project", on_click=self.open_project),
                self.recent_projects,
            ]),
            self.status,
        ], spacing=6)
        project_context = self.project_context_panel = ft.Container(
            content=ft.Column([
                ft.Text(
                    "AUTOPILOT CONTEXT", size=18, weight=ft.FontWeight.BOLD,
                    color="#38BDF8",
                ),
                self.context_summary,
                self.project_files_heading,
                ft.Row([
                    self.files,
                    ft.Column([
                        ft.Text("Managed automatically:", weight=ft.FontWeight.BOLD),
                        ft.Text("✓ .ai/goal.md"),
                        ft.Text("✓ .ai/context.md"),
                        ft.Text("✓ .ai/memory.md"),
                        ft.Text("✓ .ai/next-steps.md"),
                    ], width=210, spacing=3),
                ]),
                ft.OutlinedButton(
                    "Inspect assembled context",
                    on_click=self.inspect_assembled_context,
                ),
            ], spacing=6),
            height=300,
            expand=True,
            padding=8,
            bgcolor="#101A24",
            border=ft.Border.all(1, "#2A6F97"),
            border_radius=8,
        )
        autopilot_panel = self.autopilot_panel = ft.Container(
            content=ft.Column([
                ft.Text(
                    "AUTOPILOT", size=18, weight=ft.FontWeight.BOLD,
                    color="#A78BFA",
                ),
                self.autopilot_status,
                ft.Row([self.autopilot_button, self.autopilot_run_button], wrap=True),
                ft.Row([
                    self.autopilot_pause_button,
                    self.autopilot_stop_button,
                    self.progress,
                ]),
                ft.Divider(color="#3F3A63", height=8),
                ft.Row([
                    ft.Column([
                        ft.Text("GOAL", weight=ft.FontWeight.BOLD),
                        self.autopilot_goal,
                        ft.Text("PLAN", weight=ft.FontWeight.BOLD),
                        self.autopilot_plan,
                    ], expand=True),
                    ft.Column([
                        ft.Text("MODEL", weight=ft.FontWeight.BOLD),
                        self.autopilot_model,
                        ft.Text("CONTEXT", weight=ft.FontWeight.BOLD),
                        self.autopilot_context_summary,
                        ft.Text("VERIFICATION", weight=ft.FontWeight.BOLD),
                        self.autopilot_verification,
                    ], expand=True),
                ], vertical_alignment=ft.CrossAxisAlignment.START),
            ], spacing=5),
            height=300,
            width=380,
            padding=8,
            bgcolor="#171526",
            border=ft.Border.all(1, "#6D5BD0"),
            border_radius=8,
        )
        composer = ft.Column([
            self.prompt,
            ft.Row([
                self.ask_button,
                self.change_button,
                self.continue_button,
            ]),
        ], spacing=6)
        self.layout = ft.Column([
            header,
            composer,
            ft.Row([project_context, autopilot_panel], expand=False),
            ft.Divider(color="#294052"),
            ft.Text(
                "TRANSCRIPT", size=18, weight=ft.FontWeight.BOLD,
                color="#34D399",
            ),
            self.transcript,
        ], expand=True, spacing=8)
        self.page.add(self.layout)
        self.refresh_recent_projects()
        self.refresh_model_status()
        if self.configuration_error:
            self.message(self.configuration_error)
        remembered = self.settings.last_project
        if remembered:
            if Path(remembered).expanduser().is_dir():
                self.path.value = remembered
                self.open_project_path(remembered)
            else:
                self.message(f"Remembered project is no longer available: {remembered}")

    def message(self, text: str) -> None:
        self.status.value = text
        self.page.update()

    def model_changed(self, _event: ft.ControlEvent) -> None:
        self.refresh_model_status()
        self.refresh_autopilot_overview()
        self.persist_settings(last_model=self.model_selector.value or "fake")
        self.page.update()

    def refresh_model_status(self) -> None:
        name = self.model_selector.value or "fake"
        if name == "fake":
            self.model_status.value = "fake — built-in test adapter — credential not required"
            return
        configuration = self.models.get(name)
        if configuration is None:
            self.model_status.value = f"{name} — unavailable configuration"
            return
        try:
            available = self.configuration.credential(configuration) is not None
        except ConfigurationError:
            available = False
        status = "credential available" if available else "credential unavailable"
        self.model_status.value = f"{configuration.name} — {configuration.model} — {status}"

    def open_project(self, _event: ft.ControlEvent) -> None:
        self.open_project_path(self.path.value or "")

    def open_project_path(self, raw_path: str) -> bool:
        try:
            new_root = self.projects.open(raw_path)
            new_controls = self.projects.load_control_documents(new_root)
            new_files = self.projects.selectable_files(new_root)
            self.root = new_root
            self.autopilot_run_button.text = "Run Autopilot"
            self.path.value = str(new_root)
            self.controls = new_controls
            self.replace_file_controls(new_root, new_files)
            self.proposal = None
            self.context_preview.value = self.assemble_context()
            self.refresh_autopilot_status()
            recent = (str(new_root), *(path for path in self.settings.recent_projects if path != str(new_root)))
            self.settings = UserSettings(
                last_project=str(new_root),
                last_model=self.model_selector.value or "fake",
                recent_projects=recent[:8],
            )
            self.refresh_recent_projects()
            try:
                self.configuration.save_settings(self.settings)
            except OSError:
                self.message(
                    f"Opened {self.root}, but could not save user settings to "
                    f"{self.configuration.settings_path}."
                )
                return True
            self.message(f"Opened {self.root}")
            return True
        except (ValueError, OSError) as exc:
            self.message(str(exc))
            return False

    def refresh_autopilot_status(self) -> None:
        if self.root is None:
            self.autopilot_status.value = "STATUS: NO PROJECT"
            self.autopilot_status.color = "#94A3B8"
            self.autopilot_ready = False
            self.autopilot_button.disabled = True
            self.autopilot_run_button.disabled = True
            self.style_autopilot_run_buttons()
            self.autopilot_button.tooltip = "Open an Autopilot-ready project first."
            self.autopilot_run_button.tooltip = "Open an Autopilot-ready project first."
            return
        readiness = self.autopilot_readiness.inspect(self.root)
        self.autopilot_ready = readiness.ready
        if readiness.ready:
            self.autopilot_status.value = (
                f"STATUS: READY — WARNING: {readiness.warnings[0].message}"
                if readiness.warnings else "STATUS: READY"
            )
            self.autopilot_status.color = "#4ADE80" if not readiness.warnings else "#FBBF24"
            self.autopilot_button.disabled = self.busy
            self.autopilot_run_button.disabled = self.busy
            self.style_autopilot_run_buttons()
            self.autopilot_button.tooltip = "Run exactly one supervised Autopilot step."
            self.autopilot_run_button.tooltip = "Run the bounded supervised Autopilot loop."
        else:
            reason = readiness.problems[0].message
            self.autopilot_status.value = f"STATUS: NOT READY — {reason}"
            self.autopilot_status.color = "#FB7185"
            self.autopilot_button.disabled = True
            self.autopilot_run_button.disabled = True
            self.style_autopilot_run_buttons()
            self.autopilot_button.tooltip = reason
            self.autopilot_run_button.tooltip = reason
        self.refresh_autopilot_overview()

    def refresh_autopilot_overview(self) -> None:
        name = self.model_selector.value or "fake"
        configuration = self.models.get(name)
        self.autopilot_model.value = configuration.model if configuration else name
        if self.root is None:
            self.autopilot_goal.value = "No project open"
            self.autopilot_plan.value = "No current plan"
            self.autopilot_context_summary.value = "0 files"
            self.autopilot_verification.value = "Not configured"
            return
        try:
            goal = self.root / ".ai" / "goal.md"
            self.autopilot_goal.value = goal.read_text(encoding="utf-8").strip() or "Not defined"
        except OSError:
            self.autopilot_goal.value = "Unavailable"
        try:
            self.autopilot_plan.value = self.projects.next_step_state(self.root).display
        except OSError:
            self.autopilot_plan.value = "Unavailable"
        self.refresh_context_summary()
        try:
            project_configuration = self.project_configuration.load(self.root)
            commands = project_configuration.verify.commands
            self.autopilot_verification.value = (
                "\n".join(" ".join(command) for command in commands)
                if commands else "Not configured"
            )
        except ProjectConfigurationError:
            self.autopilot_verification.value = "Configuration unavailable"

    async def browse_project(self, _event: ft.ControlEvent) -> None:
        try:
            initial = str(self.root) if self.root else None
            selected = await self.file_picker.get_directory_path(
                dialog_title="Open AI Workbench project",
                initial_directory=initial,
            )
            if selected:
                self.open_project_path(selected)
        except Exception:
            self.message("The folder picker is unavailable. Enter a project path manually.")

    def recent_project_selected(self, _event: ft.ControlEvent) -> None:
        selected = self.recent_projects.value
        if selected:
            self.open_project_path(selected)

    def refresh_recent_projects(self) -> None:
        self.recent_projects.options.clear()
        self.recent_projects.options.extend(
            ft.DropdownOption(path) for path in self.settings.recent_projects
        )
        self.recent_projects.value = None

    def persist_settings(self, *, last_model: str) -> None:
        self.settings = UserSettings(
            last_project=str(self.root) if self.root else self.settings.last_project,
            last_model=last_model,
            recent_projects=self.settings.recent_projects,
        )
        try:
            self.configuration.save_settings(self.settings)
        except OSError:
            self.status.value = f"Could not save user settings to {self.configuration.settings_path}."

    def replace_file_controls(self, root: Path, paths: list[Path]) -> None:
        """Repopulate the mounted list while discarding all prior selection state."""
        configured: set[Path] = set()
        protected: set[Path] = set()
        try:
            if not hasattr(self, "autopilot_runner"):
                raise ProjectConfigurationError("Autopilot services are not initialized.")
            project_configuration = getattr(
                self, "project_configuration", ProjectConfigurationService()
            )
            configuration = project_configuration.load(root)
            policy = ProtectedPathPolicy((
                *configuration.autopilot.protected_paths,
                *configuration.autopilot.verification_contract_paths,
                ".ai/project.yml",
            ))
            description = self.autopilot_runner.autopilot_context.build(
                root,
                self.controls,
                configuration.autopilot.context_paths,
                (),
                policy.is_protected,
            )
            configured.update(description.project_paths)
            protected.update(description.protected_paths)
        except (ProjectConfigurationError, AutopilotContextError, OSError, ValueError):
            pass
        self.files.controls.clear()
        self.files.controls.extend(
            ft.Checkbox(
                label=(
                    f"{path.as_posix()}    "
                    f"{'Protected' if path in protected else 'Writable'}"
                ),
                data=path,
                value=path in configured,
                on_change=self.autopilot_context_selection_changed,
                key=f"{root}::{path.as_posix()}",
            )
            for path in paths
        )
        self.refresh_context_summary(protected)

    def refresh_context_summary(self, protected: set[Path] | None = None) -> None:
        if not hasattr(self, "context_summary"):
            return
        selected = self.selected_files()
        if protected is None:
            protected = set()
            if self.root is not None:
                try:
                    configuration = self.project_configuration.load(self.root)
                    policy = ProtectedPathPolicy((
                        *configuration.autopilot.protected_paths,
                        *configuration.autopilot.verification_contract_paths,
                        ".ai/project.yml",
                    ))
                    protected = {path for path in selected if policy.is_protected(path)}
                except ProjectConfigurationError:
                    pass
        protected_count = sum(path in protected for path in selected)
        writable_count = len(selected) - protected_count
        managed_count = len(self.controls) if hasattr(self, "controls") else 0
        self.context_summary.value = (
            f"{len(selected)} project files selected · {writable_count} writable · "
            f"{protected_count} protected · {managed_count} managed AI files"
        )
        self.project_files_heading.value = (
            f"Project files    {len(selected)} selected / {len(self.files.controls)} files"
        )
        self.autopilot_context_summary.value = (
            f"{len(selected) + managed_count} files, "
            f"{len(selected)} project + {managed_count} managed"
        )

    def autopilot_context_selection_changed(self, _event: ft.ControlEvent) -> None:
        if self.root is None:
            return
        paths = tuple(path.as_posix() for path in self.selected_files())
        try:
            self.project_configuration.save_context_paths(self.root, paths)
            self.status.value = (
                f"Autopilot project context updated: {len(paths)} file(s)."
                if paths else
                "Autopilot project context is empty; new-file steps may proceed, but repair context is limited."
            )
            self.refresh_context_summary()
        except (ProjectConfigurationError, OSError) as exc:
            self.log_workflow_failure("autopilot-context-selection", exc)
            self.status.value = "Could not save Autopilot project context selection."
        self.page.update()

    def selected_files(self) -> list[Path]:
        return [control.data for control in self.files.controls if isinstance(control, ft.Checkbox) and control.value]

    def build_context(self, _event: ft.ControlEvent) -> None:
        if self.root is None:
            self.message("Open a project first.")
            return
        try:
            self.context_preview.value = self.assemble_context()
            self.page.update()
        except OSError as exc:
            self.message(f"Could not build context: {exc}")

    def inspect_assembled_context(self, _event: ft.ControlEvent) -> None:
        if self.root is None:
            self.message("Open a project first.")
            return
        try:
            context = self.assemble_context()
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            self.log_workflow_failure("inspect-context", exc)
            self.message("Could not assemble project context for inspection.")
            return
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Assembled model context"),
            content=ft.Container(
                content=ft.TextField(
                    value=context,
                    multiline=True,
                    read_only=True,
                    min_lines=16,
                    max_lines=24,
                ),
                width=850,
            ),
            actions=[ft.TextButton("Close", on_click=lambda _event: self.page.pop_dialog())],
        )
        self.page.show_dialog(dialog)

    def assemble_context(self) -> str:
        if self.root is None:
            raise ValueError("Open a project first.")
        context = self.context_builder.build(self.root, self.controls, self.selected_files())
        self.context_preview.value = context
        return context

    def selected_adapter(self) -> tuple[str, ModelAdapter]:
        name = self.model_selector.value or "fake"
        adapter: ModelAdapter = self.fake_adapter
        if name == "fake":
            return name, adapter
        configuration = self.models[name]
        credential = self.configuration.credential(configuration)
        if not credential:
            raise ConfigurationError(self.configuration.credential_setup_message(configuration))
        if configuration.adapter != "openai-compatible":
            raise ConfigurationError(f"Unsupported adapter type: {configuration.adapter}")
        return name, OpenAICompatibleAdapter(configuration, credential)

    def set_busy(self, busy: bool, status: str | None = None) -> None:
        self.busy = busy
        self.ask_button.disabled = busy
        self.change_button.disabled = busy
        self.continue_button.disabled = busy
        self.autopilot_button.disabled = busy or not self.autopilot_ready
        self.autopilot_run_button.disabled = busy or not self.autopilot_ready
        self.style_autopilot_run_buttons()
        self.progress.visible = busy
        if status is not None:
            self.status.value = status
        self.page.update()

    def style_autopilot_run_buttons(self) -> None:
        disabled_color = "#334155"
        disabled_text = "#94A3B8"
        self.autopilot_button.bgcolor = disabled_color if self.autopilot_button.disabled else "#0F766E"
        self.autopilot_run_button.bgcolor = (
            disabled_color if self.autopilot_run_button.disabled else "#2563EB"
        )
        self.autopilot_button.color = disabled_text if self.autopilot_button.disabled else ft.Colors.WHITE
        self.autopilot_run_button.color = (
            disabled_text if self.autopilot_run_button.disabled else ft.Colors.WHITE
        )

    def append_user_entry(self, mode: str, task: str) -> ft.Container:
        entry = ft.Container(
            content=ft.Column([
                ft.Text(f"You · {mode.upper()}", weight=ft.FontWeight.BOLD),
                ft.Markdown(task, selectable=True),
            ], spacing=4),
            padding=12,
            border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.BLUE),
            data={"kind": "user", "mode": mode, "task": task},
        )
        self.transcript.controls.append(entry)
        self.request_transcript_scroll()
        return entry

    def append_model_entry(self, title: str, body: str) -> ft.Container:
        entry = ft.Container(
            content=ft.Column([
                ft.Text(title, weight=ft.FontWeight.BOLD),
                ft.Markdown(body, selectable=True),
            ], spacing=6),
            padding=12,
            border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.06, ft.Colors.GREY),
            data={"kind": "model", "title": title, "body": body},
        )
        self.transcript.controls.append(entry)
        self.request_transcript_scroll()
        return entry

    def append_activity_entry(self, text: str) -> ft.Container:
        entry = ft.Container(
            content=ft.Text(text, italic=True),
            padding=10,
            data={"kind": "activity", "text": text},
        )
        self.transcript.controls.append(entry)
        self.request_transcript_scroll()
        return entry

    def request_transcript_scroll(self) -> None:
        if isinstance(self.page, ft.Page):
            self.page.run_task(self.scroll_transcript_to_end)

    async def scroll_transcript_to_end(self) -> None:
        try:
            # Let the page update carrying newly appended controls reach the
            # Flet client before asking that client to resolve the end offset.
            await asyncio.sleep(0.05)
            await self.transcript.scroll_to(offset=-1, duration=200)
        except (AssertionError, RuntimeError):
            # A very early startup message can be appended before ListView is
            # mounted. The next transcript append will scroll normally.
            pass

    def update_activity_entry(self, entry: ft.Container, text: str, *, error: bool = False) -> None:
        entry.content = ft.Text(text, color=ft.Colors.RED if error else None)
        entry.data = {"kind": "error" if error else "activity", "text": text}

    def model_display_name(self, name: str) -> str:
        if name == "fake":
            return "Fake model"
        configuration = self.models.get(name)
        value = configuration.name if configuration else name
        return value.replace("-", " ").title()

    async def ask(self, _event: ft.ControlEvent) -> None:
        await self.run_prompt_workflow("ask")

    async def change(self, _event: ft.ControlEvent) -> None:
        await self.run_prompt_workflow("change")

    async def continue_next_step(self, _event: ft.ControlEvent) -> None:
        await self.run_prompt_workflow("continue")

    async def run_autopilot(self, _event: ft.ControlEvent) -> None:
        if self.busy:
            return
        if self.root is None:
            self.message("Open an Autopilot-ready project first.")
            return
        resume = (
            self.autopilot_run_button.text == "Resume Autopilot"
            and self.autopilot_loop.last_result is not None
            and self.autopilot_loop.last_result.paused
        )
        activity = self.append_activity_entry(
            "Resuming Autopilot at the next safe boundary..."
            if resume else "Starting bounded Autopilot run..."
        )
        self.autopilot_pause_button.visible = True
        self.autopilot_stop_button.visible = True
        self.autopilot_pause_button.disabled = False
        self.autopilot_stop_button.disabled = False
        self.autopilot_run_button.text = "Run Autopilot"
        self.set_busy(True, "Autopilot is running...")
        try:
            name, adapter = self.selected_adapter()
            method = self.autopilot_loop.resume if resume else self.autopilot_loop.run
            selected = tuple(self.selected_files())
            queue: asyncio.Queue[AutopilotLoopEvent] = asyncio.Queue()
            event_loop = asyncio.get_running_loop()

            def receive_event(event: AutopilotLoopEvent) -> None:
                event_loop.call_soon_threadsafe(queue.put_nowait, event)

            worker = asyncio.create_task(asyncio.to_thread(
                method,
                self.root,
                adapter,
                selected,
                on_event=receive_event,
            ))
            while not worker.done():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                self.append_autopilot_event(event, name, selected)
                self.page.update()
            result = await worker
            while not queue.empty():
                self.append_autopilot_event(queue.get_nowait(), name, selected)
            critical = result.stop_reason.value == "ROLLBACK_ERROR"
            self.update_activity_entry(
                activity,
                self.format_autopilot_run_summary(result),
                error=critical,
            )
            if result.last_step_result is not None:
                self.append_model_entry(
                    f"Autopilot last attempt · {result.last_step_result.status.value}",
                    self.format_autopilot_result(result.last_step_result),
                )
            if result.total_attempts:
                self.controls = self.projects.load_control_documents(self.root)
                self.replace_file_controls(self.root, self.projects.selectable_files(self.root))
                self.context_preview.value = self.assemble_context()
            if result.status is AutopilotRunStatus.PAUSED:
                self.autopilot_run_button.text = "Resume Autopilot"
            self.refresh_autopilot_status()
            summary = self.format_autopilot_run_summary(result)
            self.message(f"CRITICAL: {summary} Filesystem state may be uncertain." if critical else summary)
        except (ConfigurationError, KeyError, OSError, ValueError) as exc:
            self.log_workflow_failure("autopilot-loop", exc)
            text = "AI Workbench: Autopilot could not start. Check project configuration and credentials."
            self.update_activity_entry(activity, text, error=True)
            self.message(text)
        except Exception as exc:
            self.log_workflow_failure("autopilot-loop", exc)
            text = "AI Workbench: Autopilot failed unexpectedly. Check the local log."
            self.update_activity_entry(activity, text, error=True)
            self.message(text)
        finally:
            self.autopilot_pause_button.visible = True
            self.autopilot_stop_button.visible = True
            self.autopilot_pause_button.disabled = True
            self.autopilot_stop_button.disabled = True
            self.set_busy(False)

    def pause_autopilot(self, _event: ft.ControlEvent) -> None:
        self.autopilot_loop.request_pause()
        self.status.value = "Pause requested; Autopilot will pause after the current attempt."
        self.autopilot_pause_button.disabled = True
        self.page.update()

    def stop_autopilot(self, _event: ft.ControlEvent) -> None:
        self.autopilot_loop.request_stop()
        self.status.value = "Stop requested; Autopilot will stop after the current attempt."
        self.autopilot_stop_button.disabled = True
        self.page.update()

    def append_autopilot_event(
        self,
        event: AutopilotLoopEvent,
        model_name: str,
        selected: tuple[Path, ...],
    ) -> None:
        if event.kind == "attempt_started":
            configuration = self.models.get(model_name)
            model = configuration.model if configuration is not None else model_name
            body = [
                f"**Model:** `{model}`",
                "**Plan:**",
                event.plan or "(no durable plan text)",
                f"**Current step:** {event.step or '(no actionable step)'}",
                "**Retry context:** " + (
                    "present — previous attempt failed and was rolled back"
                    if event.has_retry_context else "none"
                ),
                "**Status:** assembling explicit model context...",
            ]
            self.append_model_entry(f"Autopilot attempt {event.attempt}", "\n".join(body))
        elif event.kind == "context_ready" and event.context is not None:
            context = event.context
            protected = set(context.protected_paths)
            body = ["Managed context:"]
            body.extend(f"  {path.as_posix()}" for path in context.managed_paths)
            body.append("Project context:")
            body.extend(
                f"  {path.as_posix()}  {'protected' if path in protected else 'writable'}"
                for path in context.project_paths
            )
            if not context.project_paths:
                body.append("  none — explicit project context is empty")
            if context.unmatched_patterns:
                body.append("Configured paths currently matching no file:")
                body.extend(f"  {pattern}" for pattern in context.unmatched_patterns)
            self.append_activity_entry("\n".join(body))
        elif event.kind == "waiting_for_model":
            self.append_activity_entry("Requesting model response...")
        elif event.kind == "model_response_received":
            self.append_activity_entry("Model response received.")
        elif event.kind == "proposal_ready":
            paths = "\n".join(f"  {path.as_posix()}" for path in event.paths) or "  none"
            self.append_activity_entry("Proposal files:\n" + paths)
        elif event.kind == "validating_paths":
            self.append_activity_entry("Validating proposal paths...")
        elif event.kind == "applying_transaction":
            self.append_activity_entry("Applying speculative transaction...")
        elif event.kind == "verifying":
            self.append_activity_entry("Running verification...")
        elif event.kind.startswith("verification_"):
            self.append_activity_entry(
                "Verification " + event.kind.removeprefix("verification_").upper() + "."
            )
        elif event.kind == "restoring_snapshot":
            self.append_activity_entry("Restoring snapshot...")
        elif event.kind == "rollback_complete":
            self.append_activity_entry("Rollback complete.")
        elif event.kind == "attempt_finished" and event.result is not None:
            result = event.result
            paths = "\n".join(f"- `{path.as_posix()}`" for path in result.changed_paths) or "- none"
            verification = result.verification.status.value if result.verification else "not run"
            rollback = (
                "complete" if result.rollback_succeeded else
                "failed" if result.rollback_attempted else
                "not needed"
            )
            action = (
                "advance to next step" if result.changes_kept else
                "retry same logical step" if result.rollback_succeeded else
                "stop"
            )
            self.append_model_entry(
                f"Attempt {event.attempt} result · {result.status.value}",
                f"**Step:** {event.step or '(none)'}\n\n"
                f"**Proposed files:**\n{paths}\n\n"
                f"**Transaction:** {'applied' if result.applied else 'not applied'}\n\n"
                f"**Verification:** {verification}\n\n"
                f"**Rollback:** {rollback}\n\n"
                f"**Result:** {action}",
            )
        elif event.kind == "retry_scheduled":
            self.append_activity_entry(
                f"Attempt {event.attempt}: rollback complete; retrying the same step."
            )

    @staticmethod
    def format_autopilot_run_summary(result: AutopilotRunResult) -> str:
        verification = (
            result.last_step_result.verification.status.value
            if result.last_step_result and result.last_step_result.verification
            else "not run"
        )
        return (
            f"Autopilot {result.status.value.lower()}: {result.stop_reason.value}; "
            f"{result.total_attempts} attempt(s), {result.successful_steps} passed step(s); "
            f"final verification: {verification}; next step: {result.current_step or 'none'}."
        )

    async def run_one_autopilot_step(self, _event: ft.ControlEvent) -> None:
        if self.busy:
            return
        if self.root is None:
            self.message("Open an Autopilot-ready project first.")
            return
        try:
            step = self.projects.current_next_step(self.root)
        except OSError as exc:
            self.log_workflow_failure("autopilot", exc)
            step = None
        if not step:
            self.append_model_entry(
                "AI Workbench",
                "Autopilot step was not run: no actionable next step is available.",
            )
            self.refresh_autopilot_status()
            self.page.update()
            return

        self.append_user_entry(
            "autopilot",
            f"Run exactly one automatically supervised step:\n\n{step}",
        )
        activity = self.append_activity_entry("Preparing one Autopilot step...")
        self.set_busy(True, "Running one Autopilot step...")
        try:
            _name, adapter = self.selected_adapter()
            result = await asyncio.to_thread(
                self.autopilot_runner.run,
                self.root,
                adapter,
                tuple(self.selected_files()),
            )
            self.update_activity_entry(
                activity,
                f"Autopilot step: {result.status.value}",
                error=result.status is AutopilotStepStatus.ROLLBACK_ERROR,
            )
            self.append_model_entry(
                f"Autopilot · {result.status.value}",
                self.format_autopilot_result(result),
            )
            if result.applied:
                self.controls = self.projects.load_control_documents(self.root)
                files = self.projects.selectable_files(self.root)
                self.replace_file_controls(self.root, files)
                self.context_preview.value = self.assemble_context()
            self.refresh_autopilot_status()
            if result.status is AutopilotStepStatus.ROLLBACK_ERROR:
                self.message("CRITICAL: Autopilot rollback failed; working state is uncertain.")
            else:
                self.message(f"Autopilot step finished: {result.status.value}")
            self.request_transcript_scroll()
        except (ConfigurationError, KeyError, OSError, ValueError) as exc:
            self.log_workflow_failure("autopilot", exc)
            text = "AI Workbench: Autopilot step could not start. Check configuration and credentials."
            self.update_activity_entry(activity, text, error=True)
            self.message(text)
        except Exception as exc:
            self.log_workflow_failure("autopilot", exc)
            text = "AI Workbench: Autopilot step failed unexpectedly. Check the local log."
            self.update_activity_entry(activity, text, error=True)
            self.message(text)
        finally:
            self.set_busy(False)

    @staticmethod
    def format_autopilot_result(result: AutopilotStepResult) -> str:
        lines = [
            f"**Step:** {result.step or '(unavailable)'}",
            f"**Transaction applied:** {'yes' if result.applied else 'no'}",
        ]
        if result.changed_paths:
            lines.append("**Changed files:**\n" + "\n".join(
                f"- `{path.as_posix()}`" for path in result.changed_paths
            ))
        if result.snapshot_path is not None:
            lines.append(f"**Snapshot:** `{result.snapshot_path}`")
        if result.changes_kept is True:
            lines.append("**Changes kept:** yes — verification passed.")
        elif result.rollback_attempted and result.rollback_succeeded:
            lines.append(
                "**Changes rolled back:** yes\n\n"
                "Working state restored to the complete pre-step snapshot."
            )
        elif result.rollback_attempted:
            lines.append(
                "**Changes rolled back:** no\n\n"
                "**CRITICAL:** Rollback failed; working state is uncertain."
            )
        if result.verification is not None:
            lines.append(f"**Verification:** {result.verification.status.value}")
            for command in result.verification.commands:
                lines.append("**Command:** `" + " ".join(command.command) + "`")
                if command.stdout:
                    lines.append("**stdout**\n```text\n" + command.stdout.rstrip() + "\n```")
                if command.stderr:
                    lines.append("**stderr**\n```text\n" + command.stderr.rstrip() + "\n```")
        if result.problems:
            lines.append("**Readiness problems:**\n" + "\n".join(
                f"- {problem.message}" for problem in result.problems
            ))
        if result.message:
            lines.append(f"**Detail:** {result.message}")
        return "\n\n".join(lines)

    async def run_prompt_workflow(self, mode: str) -> None:
        if self.busy:
            return
        activity = "Reading next step..." if mode == "continue" else f"Sending {mode.upper()} request..."
        task = (self.prompt.value or "").strip()
        user_entry: ft.Container | None = None
        if mode != "continue" and task:
            user_entry = self.append_user_entry(mode, task)
        activity_entry = self.append_activity_entry(activity)
        self.set_busy(True, activity)
        try:
            if self.root is None:
                text = "AI Workbench: Open a project first."
                self.update_activity_entry(activity_entry, text, error=True)
                self.message("Open a project first.")
                return
            if mode == "continue":
                try:
                    task = self.projects.current_next_step(self.root) or ""
                except OSError as exc:
                    self.log_workflow_failure(mode, exc)
                    self.update_activity_entry(
                        activity_entry,
                        "AI Workbench: Could not read .ai/next-steps.md.",
                        error=True,
                    )
                    self.message("Could not read .ai/next-steps.md.")
                    return
                if not task:
                    self.update_activity_entry(
                        activity_entry,
                        "AI Workbench: No actionable next step was found in .ai/next-steps.md.",
                        error=True,
                    )
                    self.message("No actionable next step was found in .ai/next-steps.md.")
                    return
                user_entry = self.append_user_entry(
                    "continue",
                    f"Continue requested. Executing exactly one next step:\n\n{task}",
                )
                self.update_activity_entry(activity_entry, "Preparing CHANGE request...")
            elif not task:
                self.update_activity_entry(
                    activity_entry,
                    "AI Workbench: Enter a prompt before using ASK or CHANGE.",
                    error=True,
                )
                self.message("Enter a prompt before using ASK or CHANGE.")
                return

            context = self.assemble_context()
            name, adapter = self.selected_adapter()
            display_name = self.model_display_name(name)
            request_mode = "ASK" if mode == "ask" else "CHANGE"
            thinking = f"{display_name} — thinking ({request_mode})..."
            self.update_activity_entry(activity_entry, thinking)
            self.set_busy(True, thinking)
            if mode == "ask":
                result = await asyncio.to_thread(self.workflow.ask, adapter, context, task)
                self.proposal = None
                self.update_activity_entry(activity_entry, f"{display_name} answered.")
                self.append_model_entry(display_name, result.response)
                self.message(f"Answer received from {name}. No files were changed.")
            else:
                result = await asyncio.to_thread(
                    self.workflow.change, adapter, self.root, context, task
                )
                self.proposal = result.proposal
                self.update_activity_entry(activity_entry, f"{display_name} proposed changes.")
                self.display_proposal(result.proposal, self.root)
        except ProposalError as exc:
            self.log_workflow_failure(mode, exc)
            self.proposal = None
            text = f"AI Workbench: Model response was not a valid file proposal: {exc}"
            self.update_activity_entry(activity_entry, text, error=True)
            self.message(text)
        except (ConfigurationError, KeyError, ModelRequestError, OSError, ValueError) as exc:
            self.log_workflow_failure(mode, exc)
            text = f"AI Workbench: Request failed: {exc}"
            self.update_activity_entry(activity_entry, text, error=True)
            self.message(text)
        except Exception as exc:
            self.log_workflow_failure(mode, exc)
            text = "AI Workbench: Request failed. Check the local application log for details."
            self.update_activity_entry(activity_entry, text, error=True)
            self.message(text)
        finally:
            self.set_busy(False)

    @staticmethod
    def log_workflow_failure(mode: str, exc: Exception) -> None:
        # Log the mode and exception class, but not exception text: third-party
        # errors can echo request headers or other credential-bearing details.
        logger.error("%s workflow failed (%s)", mode.upper(), type(exc).__name__)

    def display_proposal(self, proposal: Proposal | None, root: Path | None = None) -> ProposalTransaction:
        transaction_root = root or self.root
        if transaction_root is None or proposal is None:
            raise ValueError("There is no proposal to display.")
        file_contents: dict[Path, str] = {}
        diffs: dict[Path, str] = {}
        file_rows: list[ft.Control] = []
        for proposed in proposal.files:
            target = transaction_root / proposed.relative_path
            old = target.read_text(encoding="utf-8") if target.is_file() else ""
            diff = "".join(difflib.unified_diff(
                old.splitlines(keepends=True), proposed.content.splitlines(keepends=True),
                fromfile=f"a/{proposed.relative_path.as_posix()}",
                tofile=f"b/{proposed.relative_path.as_posix()}",
            ))
            file_contents[proposed.relative_path] = proposed.content
            diffs[proposed.relative_path] = diff or "(no changes)"
            file_rows.append(ft.Row([
                ft.Text(proposed.relative_path.as_posix(), expand=True),
                ft.OutlinedButton(
                    "View File",
                    on_click=lambda _event, p=proposed.relative_path: self.show_proposal_content(
                        f"Proposed file · {p.as_posix()}", file_contents[p]
                    ),
                ),
                ft.OutlinedButton(
                    "View Diff",
                    on_click=lambda _event, p=proposed.relative_path: self.show_proposal_content(
                        f"Unified diff · {p.as_posix()}", diffs[p]
                    ),
                ),
            ]))
        proposal_status = ft.Text("Awaiting review", italic=True)
        accept_button = ft.Button("Accept Changes")
        reject_button = ft.OutlinedButton("Reject")
        content = ft.Column([
            ft.Text(
                f"Proposed changes · {len(proposal.files)} file(s)",
                weight=ft.FontWeight.BOLD,
            ),
            ft.Text(f"Project: {transaction_root}"),
            *file_rows,
            ft.Row([accept_button, reject_button, proposal_status]),
        ], spacing=8)
        container = ft.Container(
            content=content,
            padding=12,
            border=ft.Border.all(1, ft.Colors.OUTLINE),
            border_radius=8,
            data={"kind": "proposal"},
        )
        transaction = ProposalTransaction(
            root=transaction_root,
            proposal=proposal,
            container=container,
            status=proposal_status,
            accept_button=accept_button,
            reject_button=reject_button,
            file_contents=file_contents,
            diffs=diffs,
        )
        accept_button.on_click = lambda event, tx=transaction: self.accept_transaction(tx, event)
        reject_button.on_click = lambda event, tx=transaction: self.reject_transaction(tx, event)
        self.transactions.append(transaction)
        self.transcript.controls.append(container)
        self.request_transcript_scroll()
        self.message(f"Proposal contains {len(proposal.files)} file(s). Review before accepting.")
        return transaction

    def show_proposal_content(self, title: str, content: str) -> None:
        dialog = ft.AlertDialog(
            modal=True,
            title=title,
            content=ft.TextField(
                value=content,
                multiline=True,
                min_lines=18,
                max_lines=24,
                read_only=True,
            ),
            actions=[ft.TextButton("Close", on_click=lambda _event: self.page.pop_dialog())],
        )
        self.page.show_dialog(dialog)

    def accept_transaction(self, transaction: ProposalTransaction, _event: ft.ControlEvent) -> None:
        if transaction.resolved:
            return
        try:
            count = len(transaction.proposal.files)
            self.workspace.accept(transaction.root, transaction.proposal)
            transaction.resolved = True
            transaction.status.value = "Accepted"
            transaction.accept_button.disabled = True
            transaction.reject_button.disabled = True
            if self.proposal is transaction.proposal:
                self.proposal = None
            if (
                self.root == transaction.root
                and any(item.relative_path in CONTROL_PATHS for item in transaction.proposal.files)
            ):
                try:
                    self.controls = self.projects.load_control_documents(transaction.root)
                    self.context_preview.value = self.assemble_context()
                except OSError as exc:
                    self.log_workflow_failure("refresh-context", exc)
                    self.message(
                        f"Accepted and wrote {count} file(s), but could not refresh project context."
                    )
                    return
            if self.root == transaction.root:
                self.refresh_autopilot_status()
            self.message(f"Accepted and wrote {count} file(s).")
        except OSError as exc:
            self.log_workflow_failure("accept", exc)
            transaction.status.value = "Accept failed; no transaction was completed."
            self.message("Could not accept proposal. Check the local application log for details.")

    def reject_transaction(self, transaction: ProposalTransaction, _event: ft.ControlEvent) -> None:
        if transaction.resolved:
            return
        transaction.resolved = True
        transaction.status.value = "Rejected"
        transaction.accept_button.disabled = True
        transaction.reject_button.disabled = True
        if self.proposal is transaction.proposal:
            self.proposal = None
        self.message("Proposal rejected; no files were changed.")

    def accept(self, _event: ft.ControlEvent) -> None:
        transaction = next((item for item in reversed(self.transactions) if not item.resolved), None)
        if transaction is None:
            self.message("There is no reviewed proposal to accept.")
            return
        self.accept_transaction(transaction, _event)

    def reject(self, _event: ft.ControlEvent) -> None:
        transaction = next((item for item in reversed(self.transactions) if not item.resolved), None)
        if transaction is None:
            self.message("There is no reviewed proposal to reject.")
            return
        self.reject_transaction(transaction, _event)


def app(page: ft.Page) -> None:
    WorkbenchUI(page).build()


def main() -> None:
    ft.run(app)


if __name__ == "__main__":
    main()
