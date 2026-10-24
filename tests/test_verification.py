import sys
from pathlib import Path

import yaml

from ai_workbench.verification import VerificationService, VerificationStatus


def configure(tmp_path: Path, commands: list[list[str]], timeout: float = 2) -> None:
    control = tmp_path / ".ai"
    control.mkdir(exist_ok=True)
    (control / "project.yml").write_text(
        yaml.safe_dump({
            "verify": {"timeout_seconds": timeout, "commands": commands},
            "autopilot": {"protected_paths": [".ai/goal.md"]},
        }),
        encoding="utf-8",
    )


def test_verification_success_captures_output(tmp_path: Path) -> None:
    configure(tmp_path, [[sys.executable, "-c", "print('verification passed')"]])

    result = VerificationService().run(tmp_path)

    assert result.status is VerificationStatus.PASS
    assert result.commands[0].exit_code == 0
    assert result.commands[0].stdout == "verification passed\n"
    assert result.commands[0].stderr == ""


def test_verification_failure_captures_output_and_stops(tmp_path: Path) -> None:
    configure(tmp_path, [
        [sys.executable, "-c", "import sys; print('bad out'); print('bad err', file=sys.stderr); sys.exit(7)"],
        [sys.executable, "-c", "print('must not run')"],
    ])

    result = VerificationService().run(tmp_path)

    assert result.status is VerificationStatus.FAIL
    assert len(result.commands) == 1
    assert result.commands[0].exit_code == 7
    assert result.commands[0].stdout == "bad out\n"
    assert result.commands[0].stderr == "bad err\n"


def test_verification_timeout(tmp_path: Path) -> None:
    configure(
        tmp_path,
        [[sys.executable, "-c", "import time; time.sleep(1)"]],
        timeout=0.05,
    )

    result = VerificationService().run(tmp_path)

    assert result.status is VerificationStatus.TIMEOUT
    assert result.commands[0].status is VerificationStatus.TIMEOUT
    assert result.commands[0].exit_code is None
