#!/bin/bash
# python train.py -t lits32 -d [0] --bs 8 -f 2
# python  train.py -t litsmc -d [0]  --bs 10 -f 1 -bsf  -e 500
# python  train.py -t litsmc -r LITS-811 -e 500 --lr 11e-4 -b 8
# python  train.py -t litsmc -r LITS-940 -e 500  -d [1]  
python train.py -t nodes -e 500 --plan 7 --fold 1  --epochs 500 --cache-rate 0.0 --run-name LITS-1405 --bs 2
# python  train.py -t lungs  -e 500 --lr 11e-3 -b 2
