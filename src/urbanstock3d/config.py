"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from urbanstock3d.reconstruction.enums import (
    BackendName,
    LodRequest,
    ReconstructionPolicy,
    ReconstructionPriority,
)


class Settings(BaseSettings):
    """Runtime settings with safe, portable defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="URBANSTOCK_",
        extra="ignore",
    )

    output_dir: Path = Path("outputs")
    keep_temporary: bool = False
    roofer_executable: str = "roofer"
    http_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    http_read_timeout_seconds: float = Field(default=30.0, gt=0)
    catastro_coordinate_url: AnyHttpUrl = AnyHttpUrl(
        "https://ovc.catastro.meh.es/OVCServWeb/OVCWcfCallejero/"
        "COVCCoordenadas.svc/rest/Consulta_RCCOOR"
    )
    catastro_building_wfs_url: AnyHttpUrl = AnyHttpUrl(
        "https://ovc.catastro.meh.es/INSPIRE/wfsBU.aspx"
    )
    pnoa_wms_url: AnyHttpUrl = AnyHttpUrl("https://www.ign.es/wms-inspire/pnoa-ma")
    pnoa_wms_layer: str = "OI.OrthoimageCoverage"
    reconstruction_lod: LodRequest = LodRequest.AUTO
    reconstruction_backend: BackendName = BackendName.AUTO
    reconstruction_policy: ReconstructionPolicy = ReconstructionPolicy.FALLBACK
    reconstruction_priority: ReconstructionPriority = ReconstructionPriority.BALANCED
