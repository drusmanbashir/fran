#!/bin/bash
# python train.py -t lits32 -d [0] --bs 8 -f 2
# python  train.py -t litsmc -d 2 --bs 8 -f 3    # when debugging with -m ipdb dont use ddp as it crashes cuda 
python  train.py -t litsmc -r LITS-720 -e 800 --lr 11e-4
