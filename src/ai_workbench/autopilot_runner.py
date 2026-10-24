from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from .autopilot import AutopilotReadinessService, ProtectedPathPolicy, ReadinessProblem
from .autopilot_context import AutopilotContext, AutopilotContextError, AutopilotContextService
from .context import ContextBuilder
from .models import ModelAdapter, ModelRequestError
from .project import ProjectService
from .project_config import ProjectConfigurationError, ProjectConfigurationService
from .proposals import ProposalError
from .verification import VerificationResult, VerificationService, VerificationStatus
from .workflows import PromptWorkflow, autopilot_step_task
from .workspace import WorkspaceService


class AutopilotStepStatus(str, Enum):
    SUCCESS = "SUCCESS"
    MODEL_ERROR = "MODEL_ERROR"
    INVALID_PROPOSAL = "INVALID_PROPOSAL"
    PROTECTED_PATH_VIOLATION = "PROTECTED_PATH_VIOLATION"
    APPLY_ERROR = "APPLY_ERROR"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    VERIFICATION_TIMEOUT = "VERIFICATION_TIMEOUT"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    NOT_READY = "NOT_READY"
    ROLLBACK_ERROR = "ROLLBACK_ERROR"


@dataclass(frozen=True, slots=True)
class AutopilotStepResult:
    status: AutopilotStepStatus
    step: str | None = None
    model_response: str | None = None
    changed_paths: tuple[Path, ...] = ()
    verification: VerificationResult | None = None
    applied: bool = False
    snapshot_path: Path | None = None
    rollback_attempted: bool = False
    rollback_succeeded: bool | None = None
    changes_kept: bool | None = False
    problems: tuple[ReadinessProblem, ...] = ()
    message: str | None = None


@dataclass(frozen=True, slots=True)
class AutopilotStepProgress:
    kind: str
    paths: tuple[Path, ...] = ()
    context: AutopilotContext | None = None


class AutopilotRunner:
    """Executes exactly one supervised Autopilot transaction and never loops."""

    def __init__(
        self,
        *,
        projects: ProjectService | None = None,
        configuration: ProjectConfigurationService | None = None,
        readiness: AutopilotReadinessService | None = None,
        context_builder: ContextBuilder | None = None,
        workflow: PromptWorkflow | None = None,
        workspace: WorkspaceService | None = None,
        verification: VerificationService | None = None,
    ) -> None:
        self.projects = projects or ProjectService()
        self.configuration = configuration or ProjectConfigurationService()
        self.readiness = readiness or AutopilotReadinessService(
            configuration=self.configuration, projects=self.projects
        )
        self.context_builder = context_builder or ContextBuilder()
        self.autopilot_context = AutopilotContextService(self.context_builder)
        self.workflow = workflow or PromptWorkflow()
        self.workspace = workspace or WorkspaceService()
        self.verification = verification or VerificationService(self.configuration)

    def run(
        self,
        root: Path,
        adapter: ModelAdapter,
        selected_files: tuple[Path, ...] | list[Path] = (),
        retry_context: str | None = None,
        on_progress: Callable[[AutopilotStepProgress], None] | None = None,
    ) -> AutopilotStepResult:
        readiness = self.readiness.inspect(root)
        if not readiness.ready:
            status = (
                AutopilotStepStatus.CONFIGURATION_ERROR
                if any(problem.code == "configuration_error" for problem in readiness.problems)
                else AutopilotStepStatus.NOT_READY
            )
            return AutopilotStepResult(status=status, problems=readiness.problems)

        try:
            configuration = self.configuration.load(root)
            step = self.projects.current_next_step(root)
        except ProjectConfigurationError as exc:
            return AutopilotStepResult(
                AutopilotStepStatus.CONFIGURATION_ERROR,
                message=str(exc),
            )
        except OSError:
            return AutopilotStepResult(
                AutopilotStepStatus.NOT_READY,
                message="The current next step could not be read.",
            )
        if not step:
            return AutopilotStepResult(AutopilotStepStatus.NOT_READY)

        policy = ProtectedPathPolicy((
            *configuration.autopilot.protected_paths,
            *configuration.autopilot.verification_contract_paths,
            ".ai/project.yml",
        ))
        try:
            controls = self.projects.load_control_documents(root)
            context_description = self.autopilot_context.build(
                root,
                controls,
                configuration.autopilot.context_paths,
                selected_files,
                policy.is_protected,
            )
            context = context_description.content
            self._emit(
                on_progress,
                AutopilotStepProgress("context_ready", context=context_description),
            )
        except (OSError, UnicodeDecodeError, ValueError, AutopilotContextError) as exc:
            return AutopilotStepResult(
                AutopilotStepStatus.NOT_READY,
                step=step,
                message=str(exc) or "The project context could not be assembled.",
            )

        try:
            task = autopilot_step_task(step)
            if retry_context:
                task += "\n\n" + retry_context
            self._emit(on_progress, AutopilotStepProgress("waiting_for_model"))
            workflow_result = self.workflow.change(
                adapter,
                root,
                context,
                task,
                on_response=lambda: self._emit(
                    on_progress, AutopilotStepProgress("model_response_received")
                ),
            )
        except ProposalError as exc:
            return AutopilotStepResult(
                AutopilotStepStatus.INVALID_PROPOSAL,
                step=step,
                message=str(exc),
            )
        except ModelRequestError:
            return AutopilotStepResult(
                AutopilotStepStatus.MODEL_ERROR,
                step=step,
                message="The model request failed.",
            )
        except Exception:
            return AutopilotStepResult(
                AutopilotStepStatus.MODEL_ERROR,
                step=step,
                message="The model request failed unexpectedly.",
            )

        proposal = workflow_result.proposal
        if proposal is None:
            return AutopilotStepResult(
                AutopilotStepStatus.INVALID_PROPOSAL,
                step=step,
                model_response=workflow_result.response,
            )
        changed_paths = tuple(item.relative_path for item in proposal.files)
        self._emit(on_progress, AutopilotStepProgress("proposal_ready", changed_paths))
        self._emit(on_progress, AutopilotStepProgress("validating_paths", changed_paths))
        protected = tuple(path for path in changed_paths if policy.is_protected(path))
        if protected:
            return AutopilotStepResult(
                AutopilotStepStatus.PROTECTED_PATH_VIOLATION,
                step=step,
                model_response=workflow_result.response,
                changed_paths=changed_paths,
                message="Protected proposal paths: " + ", ".join(path.as_posix() for path in protected),
            )

        try:
            self._emit(on_progress, AutopilotStepProgress("applying_transaction", changed_paths))
            snapshot = self.workspace.accept(root, proposal)
        except (OSError, ProposalError):
            return AutopilotStepResult(
                AutopilotStepStatus.APPLY_ERROR,
                step=step,
                model_response=workflow_result.response,
                changed_paths=changed_paths,
                message="The validated transaction could not be applied.",
            )

        self._emit(on_progress, AutopilotStepProgress("verifying", changed_paths))
        verification = self.verification.run_configuration(root, configuration.verify)
        self._emit(
            on_progress,
            AutopilotStepProgress(f"verification_{verification.status.value.lower()}", changed_paths),
        )
        status = {
            VerificationStatus.PASS: AutopilotStepStatus.SUCCESS,
            VerificationStatus.FAIL: AutopilotStepStatus.VERIFICATION_FAILED,
            VerificationStatus.TIMEOUT: AutopilotStepStatus.VERIFICATION_TIMEOUT,
            VerificationStatus.CONFIGURATION_ERROR: AutopilotStepStatus.CONFIGURATION_ERROR,
        }[verification.status]
        if verification.status is not VerificationStatus.PASS:
            try:
                self._emit(on_progress, AutopilotStepProgress("restoring_snapshot", changed_paths))
                self.workspace.restore(snapshot)
                self._emit(on_progress, AutopilotStepProgress("rollback_complete", changed_paths))
            except Exception:
                return AutopilotStepResult(
                    status=AutopilotStepStatus.ROLLBACK_ERROR,
                    step=step,
                    model_response=workflow_result.response,
                    changed_paths=changed_paths,
                    verification=verification,
                    applied=True,
                    snapshot_path=snapshot,
                    rollback_attempted=True,
                    rollback_succeeded=False,
                    changes_kept=None,
                    message=(
                        "Verification did not pass and rollback failed. "
                        "The filesystem state is uncertain; stop immediately."
                    ),
                )
            return AutopilotStepResult(
                status=status,
                step=step,
                model_response=workflow_result.response,
                changed_paths=changed_paths,
                verification=verification,
                applied=True,
                snapshot_path=snapshot,
                rollback_attempted=True,
                rollback_succeeded=True,
                changes_kept=False,
                message=verification.problem,
            )
        return AutopilotStepResult(
            status=status,
            step=step,
            model_response=workflow_result.response,
            changed_paths=changed_paths,
            verification=verification,
            applied=True,
            snapshot_path=snapshot,
            rollback_attempted=False,
            rollback_succeeded=None,
            changes_kept=True,
            message=verification.problem,
        )

    @staticmethod
    def _emit(
        callback: Callable[[AutopilotStepProgress], None] | None,
        progress: AutopilotStepProgress,
    ) -> None:
        if callback is not None:
            callback(progress)
