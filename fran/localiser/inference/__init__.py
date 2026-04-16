"""Localiser inference utilities."""

from fran.localiser.inference.base import LocaliserInferer

__all__ = ["LocaliserInferer", "get_model"]


def get_model(*args, **kwargs):
    raise NotImplementedError(
        "Localiser inference model factory is not implemented in this module."
    )
