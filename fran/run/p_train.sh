#!/bin/bash
# python train.py -t lits32 -d [0] --bs 8 -f 2
python  train.py -t litsmc -d 1 -r LITS-773 --bs 10 -f 3 -bsf
# python  train.py -t litsmc -r LITS-720 -e 800 --lr 11e-4 -bsf -b 8
