"""Scratchpad for decorator experiments.

Kept intentionally import-safe so package-wide import sweeps don't execute ad-hoc code.
"""

import logging


def demo() -> None:
    logging.getLogger(__name__).info("decorators demo placeholder")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
