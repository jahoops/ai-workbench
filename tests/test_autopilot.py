from pathlib import Path

import yaml

from ai_workbench.autopilot import AutopilotReadinessService, ProtectedPathPolicy


def write_configuration(
    root: Path,
    *,
    commands: list[list[str]] | None = None,
    protected: list[str] | None = None,
) -> None:
    control = root / ".ai"
    control.mkdir(exist_ok=True)
    (control / "project.yml").write_text(
        yaml.safe_dump({
            "verify": {"commands": commands or []},
            "autopilot": {
                "max_steps": 25,
                "max_consecutive_failures": 4,
                "protected_paths": protected or [],
                "verification_contract_paths": ["tests/**"],
            },
        }),
        encoding="utf-8",
    )


def complete_project(tmp_path: Path) -> Path:
    control = tmp_path / ".ai"
    control.mkdir()
    (control / "goal.md").write_text("Deliver a tested CLI.\n", encoding="utf-8")
    (control / "next-steps.md").write_text("# Next step\n\nImplement the CLI.\n", encoding="utf-8")
    write_configuration(
        tmp_path,
        commands=[["uv", "run", "pytest", "-q"]],
        protected=[".ai/goal.md", "acceptance/**"],
    )
    return tmp_path


def test_protected_path_policy_exact_and_glob() -> None:
    policy = ProtectedPathPolicy([".ai/goal.md", "acceptance/**"])

    assert policy.is_protected(".ai/goal.md")
    assert policy.is_protected("acceptance/cases/hidden.json")
    assert not policy.is_protected("src/main.py")
    assert not policy.is_protected(".ai/next-steps.md")


def test_readiness_reports_missing_goal(tmp_path: Path) -> None:
    control = tmp_path / ".ai"
    control.mkdir()
    (control / "next-steps.md").write_text("Implement it.\n", encoding="utf-8")
    write_configuration(
        tmp_path,
        commands=[["pytest"]],
        protected=[".ai/goal.md"],
    )

    readiness = AutopilotReadinessService().inspect(tmp_path)

    assert readiness.ready is False
    assert "goal_missing" in {problem.code for problem in readiness.problems}


def test_readiness_reports_missing_verification(tmp_path: Path) -> None:
    complete_project(tmp_path)
    write_configuration(tmp_path, protected=[".ai/goal.md"])

    readiness = AutopilotReadinessService().inspect(tmp_path)

    assert readiness.ready is False
    assert "verification_missing" in {problem.code for problem in readiness.problems}


def test_readiness_succeeds_for_complete_minimal_project(tmp_path: Path) -> None:
    complete_project(tmp_path)

    readiness = AutopilotReadinessService().inspect(tmp_path)

    assert readiness.ready is True
    assert readiness.problems == ()
    assert readiness.warnings[0].code == "project_context_empty"
