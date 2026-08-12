class InvalidOperation(Exception):
    """Raised when an operation is applied to a WorldState it is not valid for."""
    pass


class GenerationError(Exception):
    """Raised when the generator cannot satisfy the requested constraints."""
    pass