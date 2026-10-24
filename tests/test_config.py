from pathlib import Path

import pytest

from ai_workbench.config import ConfigurationError, ConfigurationService, UserSettings


def test_bootstraps_user_config_and_deepseek_models(tmp_path: Path) -> None:
    service = ConfigurationService(tmp_path / "user-config")
    service.ensure_user_files()

    models = service.load_models()

    assert set(models) == {"deepseek-v4-pro", "deepseek-v4-flash"}
    assert models["deepseek-v4-pro"].base_url == "https://api.deepseek.com"
    assert models["deepseek-v4-pro"].request_options() == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }
    assert service.secrets_path.stat().st_mode & 0o077 == 0
    assert service.settings_path.is_file()
    assert service.load_settings() == UserSettings()


def test_environment_credential_precedes_secrets_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = ConfigurationService(tmp_path)
    service.ensure_user_files()
    service.secrets_path.write_text("DEEPSEEK_API_KEY=file-secret\n", encoding="utf-8")
    configuration = service.load_models()["deepseek-v4-pro"]
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-secret")

    assert service.credential(configuration) == "environment-secret"


def test_missing_credential_message_never_contains_a_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = ConfigurationService(tmp_path)
    service.ensure_user_files()
    configuration = service.load_models()["deepseek-v4-pro"]
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert service.credential(configuration) is None
    message = service.credential_setup_message(configuration)
    assert "DEEPSEEK_API_KEY" in message
    assert str(service.secrets_path) in message


def test_rejects_invalid_reasoning_effort(tmp_path: Path) -> None:
    service = ConfigurationService(tmp_path)
    service.models_path.write_text(
        "models:\n  bad:\n    adapter: openai-compatible\n    base_url: https://example.test\n"
        "    model: example\n    api_key_env: KEY\n    reasoning_effort: low\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        service.load_models()


def test_user_settings_round_trip_without_secrets(tmp_path: Path) -> None:
    service = ConfigurationService(tmp_path)
    settings = UserSettings(
        last_project="/projects/current",
        last_model="deepseek-v4-pro",
        recent_projects=("/projects/current", "/projects/older"),
    )

    service.save_settings(settings)

    assert service.load_settings() == settings
    contents = service.settings_path.read_text(encoding="utf-8")
    assert "last_project: /projects/current" in contents
    assert "last_model: deepseek-v4-pro" in contents
    assert "API_KEY" not in contents


def test_user_settings_deduplicate_recent_projects_when_loaded(tmp_path: Path) -> None:
    service = ConfigurationService(tmp_path)
    service.settings_path.write_text(
        "last_project: /one\nlast_model: fake\nrecent_projects:\n  - /one\n  - /two\n  - /one\n",
        encoding="utf-8",
    )

    assert service.load_settings().recent_projects == ("/one", "/two")
