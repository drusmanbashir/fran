#!/bin/bash
# python train.py -t lits32 -d [0] --bs 8 -f 2
python  train.py -t lits32 -d 2 --bs 8 -f 2    # when debugging with -m ipdb dont use ddp as it crashes cuda 
