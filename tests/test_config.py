from pathlib import Path

import pytest
from pydantic import ValidationError

from urbanstock3d.config import Settings


def use_directory_without_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path(__file__).parent)


def test_settings_have_portable_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_directory_without_dotenv(monkeypatch)

    settings = Settings()

    assert settings.output_dir == Path("outputs")
    assert not settings.output_dir.is_absolute()
    assert settings.keep_temporary is False
    assert settings.http_connect_timeout_seconds == 10.0
    assert settings.http_read_timeout_seconds == 30.0


def test_settings_can_be_overridden_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_directory_without_dotenv(monkeypatch)
    monkeypatch.setenv("URBANSTOCK_OUTPUT_DIR", "custom-outputs")
    monkeypatch.setenv("URBANSTOCK_KEEP_TEMPORARY", "true")

    settings = Settings()

    assert settings.output_dir == Path("custom-outputs")
    assert settings.keep_temporary is True


def test_settings_reject_non_positive_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_directory_without_dotenv(monkeypatch)

    with pytest.raises(ValidationError):
        Settings(http_connect_timeout_seconds=0)
