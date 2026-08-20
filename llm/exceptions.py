class LLMError(Exception):
    """Base exception for LLM-related errors."""


class LLMProviderError(LLMError):
    """Raised when the provider returns an error or bad response."""


class LLMTimeoutError(LLMError):
    """Raised when a provider request times out."""
