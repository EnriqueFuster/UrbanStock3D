import pytest
import typer
from typer.testing import CliRunner

from urbanstock3d.cli import app, build_query
from urbanstock3d.domain.cadastre import CoordinateQuery, RefcatQuery

runner = CliRunner()


def test_cli_displays_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "resolve" in result.stdout
    assert "reconstruct" in result.stdout


def test_reconstruction_plan_validates_and_serializes_request() -> None:
    result = runner.invoke(
        app,
        [
            "reconstruct",
            "plan",
            "--refcat",
            "4531917YJ2743B",
            "--lod",
            "2.2",
            "--backend",
            "roofer",
            "--policy",
            "strict",
        ],
    )

    assert result.exit_code == 0
    assert '"status": "request_validated"' in result.stdout
    assert '"lod": "2.2"' in result.stdout
    assert '"backend": "roofer"' in result.stdout


def test_reconstruction_plan_rejects_implicit_experimental_backend() -> None:
    result = runner.invoke(
        app,
        [
            "reconstruct",
            "plan",
            "--refcat",
            "4531917YJ2743B",
            "--policy",
            "force_experimental",
        ],
    )

    assert result.exit_code == 1
    assert "requires an explicit backend" in result.stderr


def test_build_query_accepts_refcat() -> None:
    query = build_query(
        refcat="4531917yj2743b",
        longitude=None,
        latitude=None,
    )

    assert isinstance(query, RefcatQuery)
    assert query.cadastral_root_id == "4531917YJ2743B"


def test_build_query_accepts_coordinates() -> None:
    query = build_query(
        refcat=None,
        longitude=-0.3911611691988909,
        latitude=39.47695517618966,
    )

    assert isinstance(query, CoordinateQuery)


@pytest.mark.parametrize(
    ("refcat", "longitude", "latitude"),
    [
        (None, None, None),
        (None, -0.391, None),
        (None, None, 39.476),
        ("4531917YJ2743B", -0.391, 39.476),
    ],
)
def test_build_query_rejects_ambiguous_or_incomplete_input(
    refcat: str | None,
    longitude: float | None,
    latitude: float | None,
) -> None:
    with pytest.raises(typer.BadParameter):
        build_query(
            refcat=refcat,
            longitude=longitude,
            latitude=latitude,
        )
