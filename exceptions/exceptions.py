class BaseCL4NavException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseCL4NavException):
    """Raised when the choice of backbone Convnet is invalid."""


class InvalidDatasetSelection(BaseCL4NavException):
    """Raised when the choice of dataset is invalid."""
