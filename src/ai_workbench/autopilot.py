from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .project import ProjectService
from .project_config import ProjectConfigurationError, ProjectConfigurationService


class ProtectedPathPolicy:
    def __init__(self, patterns: tuple[str, ...] | list[str]) -> None:
        self.patterns = tuple(pattern.strip() for pattern in patterns if pattern.strip())

    def is_protected(self, path: Path | str) -> bool:
        relative = PurePosixPath(path.as_posix() if isinstance(path, Path) else path).as_posix()
        return any(
            relative == pattern or fnmatch.fnmatchcase(relative, pattern)
            for pattern in self.patterns
        )


@dataclass(frozen=True, slots=True)
class ReadinessProblem:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AutopilotReadiness:
    problems: tuple[ReadinessProblem, ...] = ()
    warnings: tuple[ReadinessProblem, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.problems


class AutopilotReadinessService:
    def __init__(
        self,
        configuration: ProjectConfigurationService | None = None,
        projects: ProjectService | None = None,
    ) -> None:
        self.configuration = configuration or ProjectConfigurationService()
        self.projects = projects or ProjectService()

    def inspect(self, root: Path) -> AutopilotReadiness:
        problems: list[ReadinessProblem] = []
        warnings: list[ReadinessProblem] = []
        goal = root / ".ai" / "goal.md"
        try:
            if not goal.is_file() or not goal.read_text(encoding="utf-8").strip():
                problems.append(ReadinessProblem("goal_missing", ".ai/goal.md is missing or empty."))
        except OSError:
            problems.append(ReadinessProblem("goal_unreadable", ".ai/goal.md could not be read."))

        try:
            next_step = self.projects.current_next_step(root)
        except OSError:
            next_step = None
        if not next_step:
            problems.append(ReadinessProblem(
                "next_step_missing", ".ai/next-steps.md has no actionable next step."
            ))

        try:
            configuration = self.configuration.load(root)
        except ProjectConfigurationError as exc:
            problems.append(ReadinessProblem("configuration_error", str(exc)))
            return AutopilotReadiness(tuple(problems), tuple(warnings))
        if not configuration.verify.commands:
            problems.append(ReadinessProblem(
                "verification_missing", "No verification commands are configured."
            ))
        if not configuration.autopilot.protected_paths:
            problems.append(ReadinessProblem(
                "protected_paths_missing", "No Autopilot protected paths are configured."
            ))
        elif not ProtectedPathPolicy(configuration.autopilot.protected_paths).is_protected(
            ".ai/goal.md"
        ):
            problems.append(ReadinessProblem(
                "goal_unprotected", ".ai/goal.md is not protected by the Autopilot policy."
            ))
        if configuration.autopilot.max_steps is None:
            problems.append(ReadinessProblem(
                "max_steps_missing", "Autopilot max_steps is not configured."
            ))
        if configuration.autopilot.max_consecutive_failures is None:
            problems.append(ReadinessProblem(
                "failure_limit_missing", "Autopilot max_consecutive_failures is not configured."
            ))
        if not configuration.autopilot.verification_contract_paths:
            problems.append(ReadinessProblem(
                "verification_contract_missing",
                "No authoritative verification-contract paths are configured.",
            ))
        if not configuration.autopilot.context_paths:
            warnings.append(ReadinessProblem(
                "project_context_empty",
                "No Autopilot project context paths are configured; only managed .ai context is visible.",
            ))
        return AutopilotReadiness(tuple(problems), tuple(warnings))
