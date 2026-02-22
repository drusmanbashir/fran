"""Compatibility shim for legacy localiser imports."""


def get_model(*args, **kwargs):
    raise NotImplementedError("Localiser inference model factory is not implemented in this module.")
