#!/bin/bash
# /home/ub/mambaforge/envs/dl/bin/python /home/ub/code/fran/fran/run/block_suspend.py /home/ub/code/fran/fran/run/analyze_resample.py -t litsmc -p 7 -n 8
# /home/ub/mambaforge/envs/dl/bin/python /home/ub/code/fran/fran/run/block_suspend.py /home/ub/code/fran/fran/run/analyze_resample.py -t lidc -p 1 -n 8
# /home/ub/mambaforge/envs/dl/bin/python /home/ub/code/fran/fran/run/block_suspend.py --allow-suspend /home/ub/code/fran/fran/run/analyze_resample.py -t totalseg -p 2 -n 6
exec /home/ub/mambaforge/envs/dl/bin/python /home/ub/code/fran/fran/run/block_suspend.py /home/ub/code/fran/fran/run/analyze_resample.py -t test -p 1 -n 6 -o
