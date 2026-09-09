"""Explicit collection of configured reconstruction backends."""

from urbanstock3d.reconstruction.backends.base import ReconstructionBackend
from urbanstock3d.reconstruction.enums import BackendName


class BackendRegistry:
    """Register and retrieve backend adapters without embedding routing policy."""

    def __init__(self) -> None:
        self._backends: dict[BackendName, ReconstructionBackend] = {}

    def register(self, backend: ReconstructionBackend) -> None:
        name = backend.capabilities.name
        if name in self._backends:
            raise ValueError(f"Backend already registered: {name}")
        self._backends[name] = backend

    def get(self, name: BackendName) -> ReconstructionBackend:
        if name is BackendName.AUTO:
            raise ValueError("AUTO must be resolved by the planner")
        try:
            return self._backends[name]
        except KeyError as error:
            raise KeyError(f"Backend is not registered: {name}") from error

    def is_available(self, name: BackendName) -> bool:
        if name is BackendName.AUTO or name not in self._backends:
            return False
        return self._backends[name].is_available()

    def names(self) -> tuple[BackendName, ...]:
        """Return registered backend names in deterministic order."""
        return tuple(sorted(self._backends, key=str))
