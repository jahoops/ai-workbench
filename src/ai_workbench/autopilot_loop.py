from __future__ import annotations

import threading
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from .autopilot_context import AutopilotContext
from .autopilot_runner import (
    AutopilotRunner,
    AutopilotStepProgress,
    AutopilotStepResult,
    AutopilotStepStatus,
)
from .models import ModelAdapter
from .project import ProjectService
from .project_config import ProjectConfigurationError, ProjectConfigurationService

logger = logging.getLogger(__name__)


class AutopilotRunStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class AutopilotStopReason(str, Enum):
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    USER_STOPPED = "USER_STOPPED"
    MAX_STEPS = "MAX_STEPS"
    MAX_CONSECUTIVE_FAILURES = "MAX_CONSECUTIVE_FAILURES"
    NOT_READY = "NOT_READY"
    ROLLBACK_ERROR = "ROLLBACK_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    STEP_ERROR = "STEP_ERROR"


@dataclass(frozen=True, slots=True)
class AutopilotLoopEvent:
    kind: str
    attempt: int
    step: str | None
    result: AutopilotStepResult | None = None
    paths: tuple[Path, ...] = ()
    context: AutopilotContext | None = None
    plan: str | None = None
    has_retry_context: bool = False


@dataclass(frozen=True, slots=True)
class AutopilotRunResult:
    status: AutopilotRunStatus
    stop_reason: AutopilotStopReason
    total_attempts: int
    successful_steps: int
    consecutive_failures: int
    current_step: str | None
    last_step_result: AutopilotStepResult | None
    paused: bool = False
    stopped_by_user: bool = False


@dataclass(slots=True)
class _RunState:
    root: Path
    total_attempts: int = 0
    successful_steps: int = 0
    consecutive_failures: int = 0
    current_step: str | None = None
    last_step_result: AutopilotStepResult | None = None
    retry_context: str | None = None


class AutopilotLoopRunner:
    """Thin bounded orchestration around the safe single-step transaction primitive."""

    def __init__(
        self,
        runner: AutopilotRunner | None = None,
        configuration: ProjectConfigurationService | None = None,
        projects: ProjectService | None = None,
    ) -> None:
        self.runner = runner or AutopilotRunner()
        self.configuration = configuration or ProjectConfigurationService()
        self.projects = projects or ProjectService()
        self._pause_requested = threading.Event()
        self._stop_requested = threading.Event()
        self._state: _RunState | None = None
        self.last_result: AutopilotRunResult | None = None

    def request_pause(self) -> None:
        self._pause_requested.set()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def run(
        self,
        root: Path,
        adapter: ModelAdapter,
        selected_files: tuple[Path, ...] | list[Path] = (),
        *,
        on_event: Callable[[AutopilotLoopEvent], None] | None = None,
        resume: bool = False,
    ) -> AutopilotRunResult:
        if resume:
            if self._state is None or self.last_result is None or not self.last_result.paused:
                return self._finish(AutopilotRunStatus.FAILED, AutopilotStopReason.NOT_READY)
            if self._state.root.resolve() != root.resolve():
                return self._finish(AutopilotRunStatus.FAILED, AutopilotStopReason.NOT_READY)
        else:
            self._state = _RunState(root.resolve())
        self._pause_requested.clear()
        self._stop_requested.clear()
        state = self._state
        assert state is not None

        try:
            configuration = self.configuration.load(root).autopilot
        except ProjectConfigurationError:
            return self._finish(AutopilotRunStatus.FAILED, AutopilotStopReason.CONFIG_ERROR)
        if configuration.max_steps is None or configuration.max_consecutive_failures is None:
            return self._finish(AutopilotRunStatus.FAILED, AutopilotStopReason.CONFIG_ERROR)

        while True:
            boundary = self._boundary_result(configuration.max_steps, configuration.max_consecutive_failures)
            if boundary is not None:
                return boundary
            try:
                step = self.projects.current_next_step(root)
            except OSError:
                return self._finish(AutopilotRunStatus.FAILED, AutopilotStopReason.NOT_READY)
            if not step:
                explicitly_complete = self.projects.next_step_is_complete(root)
                reason = (
                    AutopilotStopReason.COMPLETED
                    if state.successful_steps > 0 or explicitly_complete
                    else AutopilotStopReason.NOT_READY
                )
                status = (
                    AutopilotRunStatus.COMPLETED
                    if reason is AutopilotStopReason.COMPLETED
                    else AutopilotRunStatus.FAILED
                )
                state.current_step = None
                return self._finish(status, reason)

            state.current_step = step
            attempt = state.total_attempts + 1
            try:
                plan = (root / ".ai" / "next-steps.md").read_text(encoding="utf-8").strip()
            except OSError:
                plan = step
            self._emit(on_event, AutopilotLoopEvent(
                "attempt_started",
                attempt,
                step,
                plan=plan,
                has_retry_context=state.retry_context is not None,
            ))
            try:
                result = self.runner.run(
                    root,
                    adapter,
                    selected_files,
                    retry_context=state.retry_context,
                    on_progress=lambda progress: self._emit_progress(
                        on_event, attempt, step, progress
                    ),
                )
            except Exception:
                logger.exception("Unexpected failure in an Autopilot single-step transaction")
                return self._finish(AutopilotRunStatus.FAILED, AutopilotStopReason.STEP_ERROR)
            state.total_attempts += 1
            state.last_step_result = result
            self._emit(on_event, AutopilotLoopEvent("attempt_finished", attempt, step, result))

            if result.status is AutopilotStepStatus.SUCCESS:
                state.successful_steps += 1
                state.consecutive_failures = 0
                state.retry_context = None
            elif result.status in {
                AutopilotStepStatus.VERIFICATION_FAILED,
                AutopilotStepStatus.VERIFICATION_TIMEOUT,
            } and result.rollback_succeeded:
                state.consecutive_failures += 1
                state.retry_context = self.failure_retry_context(result)
                self._emit(on_event, AutopilotLoopEvent("retry_scheduled", attempt, step, result))
            elif result.status is AutopilotStepStatus.ROLLBACK_ERROR:
                return self._finish(AutopilotRunStatus.FAILED, AutopilotStopReason.ROLLBACK_ERROR)
            elif result.status is AutopilotStepStatus.CONFIGURATION_ERROR:
                return self._finish(AutopilotRunStatus.FAILED, AutopilotStopReason.CONFIG_ERROR)
            elif result.status is AutopilotStepStatus.NOT_READY:
                return self._finish(AutopilotRunStatus.FAILED, AutopilotStopReason.NOT_READY)
            else:
                return self._finish(AutopilotRunStatus.FAILED, AutopilotStopReason.STEP_ERROR)

    def resume(
        self,
        root: Path,
        adapter: ModelAdapter,
        selected_files: tuple[Path, ...] | list[Path] = (),
        *,
        on_event: Callable[[AutopilotLoopEvent], None] | None = None,
    ) -> AutopilotRunResult:
        return self.run(root, adapter, selected_files, on_event=on_event, resume=True)

    def _boundary_result(self, max_steps: int, max_failures: int) -> AutopilotRunResult | None:
        state = self._state
        assert state is not None
        if self._stop_requested.is_set():
            return self._finish(
                AutopilotRunStatus.STOPPED,
                AutopilotStopReason.USER_STOPPED,
                stopped_by_user=True,
            )
        if self._pause_requested.is_set():
            return self._finish(
                AutopilotRunStatus.PAUSED,
                AutopilotStopReason.PAUSED,
                paused=True,
            )
        if state.total_attempts >= max_steps:
            return self._finish(AutopilotRunStatus.STOPPED, AutopilotStopReason.MAX_STEPS)
        if state.consecutive_failures >= max_failures:
            return self._finish(
                AutopilotRunStatus.STOPPED,
                AutopilotStopReason.MAX_CONSECUTIVE_FAILURES,
            )
        return None

    def _finish(
        self,
        status: AutopilotRunStatus,
        reason: AutopilotStopReason,
        *,
        paused: bool = False,
        stopped_by_user: bool = False,
    ) -> AutopilotRunResult:
        state = self._state
        result = AutopilotRunResult(
            status=status,
            stop_reason=reason,
            total_attempts=state.total_attempts if state else 0,
            successful_steps=state.successful_steps if state else 0,
            consecutive_failures=state.consecutive_failures if state else 0,
            current_step=state.current_step if state else None,
            last_step_result=state.last_step_result if state else None,
            paused=paused,
            stopped_by_user=stopped_by_user,
        )
        self.last_result = result
        return result

    @staticmethod
    def failure_retry_context(result: AutopilotStepResult) -> str:
        lines = [
            "PREVIOUS AUTOPILOT ATTEMPT FAILED",
            "The previous attempt was rolled back.",
            "You are working from the restored pre-attempt state.",
            "Do not advance to a later project step.",
            "Repair the same CURRENT logical step.",
        ]
        verification = result.verification
        if verification is not None:
            for command in verification.commands:
                lines.extend([
                    "Verification command: " + " ".join(command.command),
                    "Exit code: " + (str(command.exit_code) if command.exit_code is not None else "TIMEOUT"),
                    "stdout:\n" + (command.stdout or "(empty)"),
                    "stderr:\n" + (command.stderr or "(empty)"),
                ])
        return "\n\n".join(lines)

    @staticmethod
    def _emit(
        callback: Callable[[AutopilotLoopEvent], None] | None,
        event: AutopilotLoopEvent,
    ) -> None:
        if callback is not None:
            callback(event)

    @classmethod
    def _emit_progress(
        cls,
        callback: Callable[[AutopilotLoopEvent], None] | None,
        attempt: int,
        step: str,
        progress: AutopilotStepProgress,
    ) -> None:
        cls._emit(
            callback,
            AutopilotLoopEvent(
                progress.kind,
                attempt,
                step,
                paths=progress.paths,
                context=progress.context,
            ),
        )
