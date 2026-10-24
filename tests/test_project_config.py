from pathlib import Path

import pytest

from ai_workbench.project_config import ProjectConfigurationError, ProjectConfigurationService


def test_valid_project_configuration_loads(tmp_path: Path) -> None:
    control = tmp_path / ".ai"
    control.mkdir()
    (control / "project.yml").write_text(
        "verify:\n"
        "  timeout_seconds: 30\n"
        "  commands:\n"
        "    - [uv, run, pytest, -q]\n"
        "autopilot:\n"
        "  max_steps: 10\n"
        "  max_consecutive_failures: 3\n"
        "  protected_paths: [.ai/goal.md, 'acceptance/**']\n"
        "  verification_contract_paths: ['acceptance/**']\n"
        "  context_paths: [src/main.py, 'acceptance/**']\n",
        encoding="utf-8",
    )

    configuration = ProjectConfigurationService().load(tmp_path)

    assert configuration.verify.timeout_seconds == 30
    assert configuration.verify.commands == (("uv", "run", "pytest", "-q"),)
    assert configuration.autopilot.max_steps == 10
    assert configuration.autopilot.max_consecutive_failures == 3
    assert configuration.autopilot.protected_paths == (".ai/goal.md", "acceptance/**")
    assert configuration.autopilot.context_paths == ("src/main.py", "acceptance/**")


def test_malformed_project_configuration_fails_cleanly(tmp_path: Path) -> None:
    control = tmp_path / ".ai"
    control.mkdir()
    (control / "project.yml").write_text(
        "verify:\n  commands:\n    - uv run pytest -q\n",
        encoding="utf-8",
    )

    with pytest.raises(ProjectConfigurationError, match="array"):
        ProjectConfigurationService().load(tmp_path)
