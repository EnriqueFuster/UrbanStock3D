"""Command-line interface for UrbanStock3D."""

import json
from dataclasses import asdict
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
from urbanstock3d.reconstruction import (
    BackendName,
    LodRequest,
    ReconstructionPolicy,
    ReconstructionPriority,
    ReconstructionRequest,
)

app = typer.Typer(
    name="urbanstock",
    help="Open-data semantic building intelligence for Spain.",
    no_args_is_help=True,
)
reconstruct_app = typer.Typer(
    help="Assess and plan adaptive 3D reconstruction.",
    no_args_is_help=True,
)
app.add_typer(reconstruct_app, name="reconstruct")

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


@reconstruct_app.command("plan")
def reconstruction_plan(
    refcat: Annotated[str, typer.Option(help="Cadastral reference.")],
    lod: Annotated[LodRequest, typer.Option(help="Requested level of detail.")] = LodRequest.AUTO,
    backend: Annotated[
        BackendName, typer.Option(help="Requested reconstruction backend.")
    ] = BackendName.AUTO,
    policy: Annotated[
        ReconstructionPolicy,
        typer.Option(help="Failure and fallback policy."),
    ] = ReconstructionPolicy.FALLBACK,
    priority: Annotated[
        ReconstructionPriority,
        typer.Option(help="Execution priority."),
    ] = ReconstructionPriority.BALANCED,
    use_dsm: Annotated[bool, typer.Option(help="Allow DSM evidence when available.")] = True,
    use_orthophoto: Annotated[
        bool,
        typer.Option(help="Allow orthophoto evidence when available."),
    ] = True,
) -> None:
    """Validate reconstruction intent; evidence-based planning arrives in R6."""
    try:
        cadastral_query = RefcatQuery(value=refcat)
        request = build_reconstruction_request(
            lod=lod,
            backend=backend,
            policy=policy,
            priority=priority,
            use_dsm=use_dsm,
            use_orthophoto=use_orthophoto,
        )
    except (ValidationError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    payload = {
        "status": "request_validated",
        "building_id": f"ES.SDGC.BU.{cadastral_query.cadastral_root_id}",
        "request": asdict(request),
        "notice": "Backend selection requires evidence assessment and is not implemented in R0.",
    }
    typer.echo(json.dumps(payload, indent=2))


def build_reconstruction_request(
    *,
    lod: LodRequest,
    backend: BackendName,
    policy: ReconstructionPolicy,
    priority: ReconstructionPriority,
    use_dsm: bool = True,
    use_orthophoto: bool = True,
) -> ReconstructionRequest:
    """Build and validate a reconstruction request from CLI values."""
    return ReconstructionRequest(
        lod=lod,
        backend=backend,
        policy=policy,
        priority=priority,
        use_dsm=use_dsm,
        use_orthophoto=use_orthophoto,
    )
