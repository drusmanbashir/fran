#!/bin/bash
# python train.py -t lits32 -d [0] --bs 8 -f 2
# python  train.py -t litsmc -d 1  --bs 10 -f 3 -bsf
# python  train.py -t litsmc -r LITS-787 -e 500 --lr 11e-4 -b 8
python  train.py -t lungs  -e 500 --lr 11e-3 -b 2
