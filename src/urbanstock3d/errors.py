"""Application-specific exception hierarchy."""


class UrbanStockError(Exception):
    """Base exception for expected UrbanStock3D failures."""


class ProviderError(UrbanStockError):
    """Base exception for external provider failures."""


class ProviderUnavailableError(ProviderError):
    """Raised when a provider cannot be reached."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider exceeds its configured timeout."""


class ProviderResponseError(ProviderError):
    """Raised when a provider returns an unsuccessful response."""


class RooferExecutionError(UrbanStockError):
    """Raised when the external Roofer reconstruction process fails."""
