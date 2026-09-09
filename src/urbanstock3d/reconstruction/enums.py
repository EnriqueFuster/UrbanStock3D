"""Stable vocabulary for reconstruction requests and results."""

from enum import StrEnum


class LodRequest(StrEnum):
    """Requested CityJSON level of detail."""

    AUTO = "auto"
    LOD12 = "1.2"
    LOD13 = "1.3"
    LOD22 = "2.2"


class BackendName(StrEnum):
    """Known reconstruction backends, including research-only candidates."""

    AUTO = "auto"
    ROOFER = "roofer"
    CITY3D = "city3d"
    ROOFDIFFUSION_CITY3D = "roofdiffusion_city3d"
    POINT2WSS = "point2wss"
    POINT2BUILDING = "point2building"
    BWFORMER = "bwformer"
    LOD2FORMER = "lod2former"
    QROOF = "qroof"


class ReconstructionPolicy(StrEnum):
    """Behavior when the requested reconstruction cannot be delivered."""

    STRICT = "strict"
    FALLBACK = "fallback"
    FORCE_EXPERIMENTAL = "force_experimental"


class ReconstructionPriority(StrEnum):
    """Execution priority selected by the caller."""

    SPEED = "speed"
    BALANCED = "balanced"
    ACCURACY = "accuracy"
    BENCHMARK = "benchmark"
    RESEARCH = "research"


class ReconstructionStatus(StrEnum):
    """Terminal state of a reconstruction request."""

    SUCCESS = "success"
    FAILED = "failed"
    ABSTAINED = "abstained"
