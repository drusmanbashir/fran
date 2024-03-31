#!/bin/bash
# python train.py -t lits32 -d [0] --bs 8 -f 2
# python  train.py -t litsmc -d [0]  --bs 10 -f 1 -bsf  -e 500
# python  train.py -t litsmc -r LITS-811 -e 500 --lr 11e-4 -b 8
python  train.py -t lidc2 -r LITS-903 -e 500  -d [1]  -b 4
# python  train.py -t lungs  -e 500 --lr 11e-3 -b 2
