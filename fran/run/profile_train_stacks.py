#!/usr/bin/env python3
from fran.profilers.torch_trace import build_parser, main


if __name__ == "__main__":
    main(build_parser().parse_args())
