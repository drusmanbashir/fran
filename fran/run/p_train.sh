#!/bin/bash
# python train.py -t lits32 -d [0] --bs 8 -f 2
python  train.py -t lits32 -d 2 --bs 8 -f 2 -c  # when debugging with -m ipdb done use ddp as it crashes cuda 
