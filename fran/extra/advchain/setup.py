#!/usr/bin/env python

from setuptools import find_packages, setup


def main() -> None:
    setup(
        name="advchain",
        version="0.0.0",
        description="Adversarial data augmentation with chained transformation",
        author="Chen Chen",
        author_email="chen.chen15@imperial.ac.uk",
        url="https://github.com/cherise215/advchain",
        install_requires=["torch"],
        packages=find_packages(),
    )


if __name__ == "__main__":
    main()
