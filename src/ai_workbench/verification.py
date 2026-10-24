from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .project_config import (
    ProjectConfigurationError,
    ProjectConfigurationService,
    VerificationConfiguration,
)


class VerificationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    TIMEOUT = "TIMEOUT"
    CONFIGURATION_ERROR = "CONFIGURATION ERROR"


@dataclass(frozen=True, slots=True)
class VerificationCommandResult:
    command: tuple[str, ...]
    status: VerificationStatus
    exit_code: int | None
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    status: VerificationStatus
    commands: tuple[VerificationCommandResult, ...] = ()
    problem: str | None = None


class VerificationService:
    def __init__(self, configuration: ProjectConfigurationService | None = None) -> None:
        self.configuration = configuration or ProjectConfigurationService()

    def run(self, root: Path) -> VerificationResult:
        try:
            configuration = self.configuration.load(root)
        except ProjectConfigurationError as exc:
            return VerificationResult(VerificationStatus.CONFIGURATION_ERROR, problem=str(exc))
        return self.run_configuration(root, configuration.verify)

    def run_configuration(
        self,
        root: Path,
        configuration: VerificationConfiguration,
    ) -> VerificationResult:
        """Run an already validated snapshot; never reload commands mid-transaction."""
        if not configuration.commands:
            return VerificationResult(
                VerificationStatus.CONFIGURATION_ERROR,
                problem="No verification commands are configured.",
            )

        results: list[VerificationCommandResult] = []
        for command in configuration.commands:
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=configuration.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                result = VerificationCommandResult(
                    command,
                    VerificationStatus.TIMEOUT,
                    None,
                    self._text(exc.stdout),
                    self._text(exc.stderr),
                )
                results.append(result)
                return VerificationResult(VerificationStatus.TIMEOUT, tuple(results))
            except OSError as exc:
                return VerificationResult(
                    VerificationStatus.CONFIGURATION_ERROR,
                    tuple(results),
                    f"Could not execute verification command {command[0]!r}: {exc}",
                )

            status = VerificationStatus.PASS if completed.returncode == 0 else VerificationStatus.FAIL
            results.append(VerificationCommandResult(
                command,
                status,
                completed.returncode,
                completed.stdout,
                completed.stderr,
            ))
            if status is VerificationStatus.FAIL:
                return VerificationResult(VerificationStatus.FAIL, tuple(results))
        return VerificationResult(VerificationStatus.PASS, tuple(results))

    @staticmethod
    def _text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        return value.decode(errors="replace") if isinstance(value, bytes) else value
