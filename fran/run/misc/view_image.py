#!/usr/bin/env python3
import sys

from matplotlib import pyplot as plt

from utilz.imageviewers import ImageMaskViewer


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (1, 2):
        raise SystemExit("usage: view_image.py image [labelimage]")

    image = argv[0]
    labelimage = argv[1] if len(argv) == 2 else image
    dtypes = "im" if len(argv) == 2 else "ii"
    ImageMaskViewer([image, labelimage], dtypes=dtypes)
    plt.show(block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
