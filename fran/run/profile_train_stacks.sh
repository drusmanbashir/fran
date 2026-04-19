#!/usr/bin/env bash
# python -m fran.run.profile_train_stacks -t lidc -p 1 --devices '[0]' --batch-size 1 --epochs 1 --n-samples 2 --val-every-n-epochs 1000 --skip-val true --num-workers 0 --cache-rate 0 --stack-depth 2 --profile-experimental-verbose false
# python -m ipdb -m fran.run.profile_train_stacks -t lidc -p 1 --devices '[0]' --batch-size 1 --epochs 1 --n-samples 2 --val-every-n-epochs 1000 --skip-val true --num-workers 0 --cache-rate 0 --stack-depth 2 --profile-experimental-verbose false
# python -m fran.run.profile_train_stacks -t kits2 -p 9 --devices '[0]' --batch-size 1 --epochs 1 --n-samples 2 --val-every-n-epochs 1000 --skip-val true --num-workers 0 --cache-rate 0 --stack-depth 2 --profile-experimental-verbose false
python -m fran.run.profile_train_stacks -t kits23 -p 2 --devices '[0]' --batch-size 1 --epochs 1 --n-samples 2 --val-every-n-epochs 1000 --skip-val true --num-workers 0 --cache-rate 0 --stack-depth 2 --profile-experimental-verbose false
