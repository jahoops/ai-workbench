from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

import yaml
import os
import tempfile


class ProjectConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VerificationConfiguration:
    timeout_seconds: float = 120
    commands: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class AutopilotConfiguration:
    max_steps: int | None = None
    max_consecutive_failures: int | None = None
    protected_paths: tuple[str, ...] = ()
    verification_contract_paths: tuple[str, ...] = ()
    context_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectConfiguration:
    verify: VerificationConfiguration
    autopilot: AutopilotConfiguration


class ProjectConfigurationService:
    def load(self, root: Path) -> ProjectConfiguration:
        path = root / ".ai" / "project.yml"
        if not path.is_file():
            raise ProjectConfigurationError(f"Project configuration not found: {path}")
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ProjectConfigurationError(f"Could not parse project configuration: {path}") from exc
        if not isinstance(document, dict):
            raise ProjectConfigurationError(".ai/project.yml must contain a mapping.")
        return ProjectConfiguration(
            verify=self._parse_verify(document.get("verify", {})),
            autopilot=self._parse_autopilot(document.get("autopilot", {})),
        )

    @staticmethod
    def _parse_verify(raw: object) -> VerificationConfiguration:
        if not isinstance(raw, dict):
            raise ProjectConfigurationError("project.yml verify must be a mapping.")
        timeout = raw.get("timeout_seconds", 120)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ProjectConfigurationError("verify.timeout_seconds must be a positive number.")
        commands = raw.get("commands", [])
        if not isinstance(commands, list):
            raise ProjectConfigurationError("verify.commands must be a list of argument arrays.")
        parsed: list[tuple[str, ...]] = []
        for command in commands:
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(argument, str) or not argument for argument in command)
            ):
                raise ProjectConfigurationError(
                    "Each verification command must be a non-empty array of non-empty strings."
                )
            parsed.append(tuple(command))
        return VerificationConfiguration(float(timeout), tuple(parsed))

    @staticmethod
    def _parse_autopilot(raw: object) -> AutopilotConfiguration:
        if not isinstance(raw, dict):
            raise ProjectConfigurationError("project.yml autopilot must be a mapping.")
        max_steps = raw.get("max_steps")
        max_failures = raw.get("max_consecutive_failures")
        for name, value in (("max_steps", max_steps), ("max_consecutive_failures", max_failures)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ProjectConfigurationError(f"autopilot.{name} must be a positive integer.")
        protected = raw.get("protected_paths", [])
        if (
            not isinstance(protected, list)
            or any(not isinstance(pattern, str) or not pattern.strip() for pattern in protected)
        ):
            raise ProjectConfigurationError(
                "autopilot.protected_paths must be a list of non-empty path patterns."
            )
        contracts = raw.get("verification_contract_paths", [])
        if (
            not isinstance(contracts, list)
            or any(not isinstance(pattern, str) or not pattern.strip() for pattern in contracts)
        ):
            raise ProjectConfigurationError(
                "autopilot.verification_contract_paths must be a list of non-empty path patterns."
            )
        context_paths = raw.get("context_paths", [])
        if (
            not isinstance(context_paths, list)
            or any(not isinstance(pattern, str) or not pattern.strip() for pattern in context_paths)
        ):
            raise ProjectConfigurationError(
                "autopilot.context_paths must be a list of non-empty path patterns."
            )
        return AutopilotConfiguration(
            max_steps,
            max_failures,
            tuple(protected),
            tuple(contracts),
            tuple(context_paths),
        )

    def save_context_paths(self, root: Path, paths: tuple[str, ...] | list[str]) -> None:
        for path in paths:
            candidate = PurePosixPath(path)
            if not path.strip() or candidate.is_absolute() or ".." in candidate.parts:
                raise ProjectConfigurationError(f"Unsafe Autopilot context path: {path}")
        configuration_path = root / ".ai" / "project.yml"
        try:
            document = yaml.safe_load(configuration_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ProjectConfigurationError("Could not update .ai/project.yml.") from exc
        if not isinstance(document, dict):
            raise ProjectConfigurationError(".ai/project.yml must contain a mapping.")
        autopilot = document.setdefault("autopilot", {})
        if not isinstance(autopilot, dict):
            raise ProjectConfigurationError("project.yml autopilot must be a mapping.")
        autopilot["context_paths"] = list(dict.fromkeys(paths))
        descriptor, temporary_name = tempfile.mkstemp(prefix=".project.", dir=configuration_path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                yaml.safe_dump(document, handle, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, configuration_path)
        finally:
            temporary.unlink(missing_ok=True)
