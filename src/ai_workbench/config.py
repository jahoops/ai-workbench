from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "ai-workbench"
DEFAULT_MODELS = """models:
  deepseek-v4-pro:
    adapter: openai-compatible
    base_url: https://api.deepseek.com
    model: deepseek-v4-pro
    api_key_env: DEEPSEEK_API_KEY
    thinking: true
    reasoning_effort: max
  deepseek-v4-flash:
    adapter: openai-compatible
    base_url: https://api.deepseek.com
    model: deepseek-v4-flash
    api_key_env: DEEPSEEK_API_KEY
    thinking: false
    reasoning_effort: high
"""


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    name: str
    adapter: str
    base_url: str
    model: str
    api_key_env: str
    thinking: bool | None = None
    reasoning_effort: str | None = None

    def request_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.thinking is not None:
            options["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
        if self.reasoning_effort is not None:
            options["reasoning_effort"] = self.reasoning_effort
        return options


@dataclass(frozen=True, slots=True)
class UserSettings:
    last_project: str | None = None
    last_model: str | None = None
    recent_projects: tuple[str, ...] = ()


class ConfigurationService:
    def __init__(self, config_dir: Path = DEFAULT_CONFIG_DIR) -> None:
        self.config_dir = config_dir.expanduser()
        self.models_path = self.config_dir / "models.yml"
        self.secrets_path = self.config_dir / "secrets.env"
        self.settings_path = self.config_dir / "settings.yml"

    def ensure_user_files(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if not self.models_path.exists():
            self.models_path.write_text(DEFAULT_MODELS, encoding="utf-8")
        if not self.secrets_path.exists():
            self.secrets_path.write_text(
                "# Store credentials here. Never add this file to a project.\n"
                "# DEEPSEEK_API_KEY=replace-with-your-key\n",
                encoding="utf-8",
            )
            try:
                self.secrets_path.chmod(0o600)
            except OSError:
                pass
        if not self.settings_path.exists():
            self.save_settings(UserSettings())

    def load_models(self) -> dict[str, ModelConfiguration]:
        if not self.models_path.is_file():
            raise ConfigurationError(f"Model configuration not found: {self.models_path}")
        try:
            document = yaml.safe_load(self.models_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"Could not read model configuration: {self.models_path}") from exc
        raw_models = document.get("models")
        if not isinstance(raw_models, dict) or not raw_models:
            raise ConfigurationError("models.yml must contain a non-empty 'models' mapping.")
        return {name: self._parse_model(name, raw) for name, raw in raw_models.items()}

    def credential(self, configuration: ModelConfiguration) -> str | None:
        value = os.environ.get(configuration.api_key_env)
        if value:
            return value
        return self._load_secrets().get(configuration.api_key_env)

    def credential_setup_message(self, configuration: ModelConfiguration) -> str:
        return (
            f"Credential unavailable. Set {configuration.api_key_env} in the environment "
            f"or in {self.secrets_path}."
        )

    def load_settings(self) -> UserSettings:
        if not self.settings_path.is_file():
            return UserSettings()
        try:
            document = yaml.safe_load(self.settings_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"Could not read user settings: {self.settings_path}") from exc
        if not isinstance(document, dict):
            raise ConfigurationError("settings.yml must contain a mapping.")
        last_project = document.get("last_project")
        last_model = document.get("last_model")
        recent = document.get("recent_projects", [])
        if last_project is not None and not isinstance(last_project, str):
            raise ConfigurationError("settings.yml last_project must be a string or null.")
        if last_model is not None and not isinstance(last_model, str):
            raise ConfigurationError("settings.yml last_model must be a string or null.")
        if not isinstance(recent, list) or any(not isinstance(path, str) for path in recent):
            raise ConfigurationError("settings.yml recent_projects must be a list of paths.")
        return UserSettings(last_project, last_model, tuple(dict.fromkeys(recent)))

    def save_settings(self, settings: UserSettings) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        document = {
            "last_project": settings.last_project,
            "last_model": settings.last_model,
            "recent_projects": list(settings.recent_projects),
        }
        descriptor, temporary_name = tempfile.mkstemp(prefix=".settings.", dir=self.config_dir)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                yaml.safe_dump(document, handle, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.settings_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_secrets(self) -> dict[str, str]:
        if not self.secrets_path.is_file():
            return {}
        secrets: dict[str, str] = {}
        try:
            lines = self.secrets_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ConfigurationError(f"Could not read secrets file: {self.secrets_path}") from exc
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            if "=" not in stripped:
                raise ConfigurationError(f"Invalid secrets.env entry on line {number}.")
            name, value = stripped.split("=", 1)
            name = name.strip()
            if not name.isidentifier():
                raise ConfigurationError(f"Invalid variable name in secrets.env on line {number}.")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                secrets[name] = value
        return secrets

    @staticmethod
    def _parse_model(name: object, raw: object) -> ModelConfiguration:
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise ConfigurationError("Each model configuration must be a named mapping.")
        required = ("adapter", "base_url", "model", "api_key_env")
        if any(not isinstance(raw.get(field), str) or not raw[field].strip() for field in required):
            raise ConfigurationError(f"Model configuration '{name}' is missing a required string field.")
        thinking = raw.get("thinking")
        if thinking is not None and not isinstance(thinking, bool):
            raise ConfigurationError(f"Model configuration '{name}' has a non-boolean thinking value.")
        effort = raw.get("reasoning_effort")
        if effort is not None and effort not in {"high", "max"}:
            raise ConfigurationError(f"Model configuration '{name}' reasoning_effort must be high or max.")
        return ModelConfiguration(
            name=name,
            adapter=raw["adapter"],
            base_url=raw["base_url"].rstrip("/"),
            model=raw["model"],
            api_key_env=raw["api_key_env"],
            thinking=thinking,
            reasoning_effort=effort,
        )
