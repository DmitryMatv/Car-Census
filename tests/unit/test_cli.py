from typer.testing import CliRunner

from car_census_cli import app


def test_analyze_help_exposes_transactional_overwrite_option() -> None:
    result = CliRunner().invoke(app, ["analyze", "--help"])

    assert result.exit_code == 0
    assert "--run-dir" in result.stdout
    assert "--overwrite" in result.stdout


def test_analyze_rejects_overwrite_without_explicit_run_directory() -> None:
    result = CliRunner().invoke(app, ["analyze", "input.mp4", "--overwrite"])

    assert result.exit_code == 2
    assert "--overwrite requires --run-dir" in result.stderr
