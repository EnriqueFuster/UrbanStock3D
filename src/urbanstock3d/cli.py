"""Command-line interface for UrbanStock3D."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from urbanstock3d.config import Settings
from urbanstock3d.domain.cadastre import CoordinateQuery, RefcatQuery
from urbanstock3d.errors import UrbanStockError
from urbanstock3d.outputs.cadastre import write_cadastral_artifacts
from urbanstock3d.providers.catastro import CatastroProvider, create_http_client
from urbanstock3d.providers.catastro_parser import CatastroParseError

app = typer.Typer(
    name="urbanstock",
    help="Open-data semantic building intelligence for Spain.",
    no_args_is_help=True,
)

BuildingQuery = CoordinateQuery | RefcatQuery


@app.callback()
def main() -> None:
    """Run UrbanStock3D commands."""


@app.command()
def resolve(
    refcat: Annotated[str | None, typer.Option(help="Cadastral reference.")] = None,
    longitude: Annotated[float | None, typer.Option("--lon", help="WGS84 longitude.")] = None,
    latitude: Annotated[float | None, typer.Option("--lat", help="WGS84 latitude.")] = None,
    srs: Annotated[str, typer.Option(help="Projected CRS for building geometry.")] = "EPSG:25830",
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Root directory for generated artifacts."),
    ] = None,
) -> None:
    """Resolve a cadastral building and print or write its current record."""
    try:
        query = build_query(refcat=refcat, longitude=longitude, latitude=latitude)
        settings = Settings()
        with create_http_client(settings) as client:
            building = CatastroProvider(client, settings).resolve_building(
                query,
                srs_name=srs,
            )
    except (ValidationError, UrbanStockError, CatastroParseError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    if output_dir is None:
        typer.echo(building.model_dump_json(indent=2))
        return

    building_dir = write_cadastral_artifacts(building, output_dir)
    typer.echo(f"Wrote cadastral artifacts to {building_dir}")


def build_query(
    *,
    refcat: str | None,
    longitude: float | None,
    latitude: float | None,
) -> BuildingQuery:
    """Validate the mutually exclusive CLI input forms."""
    has_any_coordinate = longitude is not None or latitude is not None

    if refcat is not None and has_any_coordinate:
        raise typer.BadParameter("Use either --refcat or --lon/--lat, not both.")
    if refcat is not None:
        return RefcatQuery(value=refcat)
    if longitude is None or latitude is None:
        raise typer.BadParameter("Provide --refcat or both --lon and --lat.")
    return CoordinateQuery(longitude=longitude, latitude=latitude)
