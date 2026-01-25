A script analyzes entire project like so        

```
#!/bin/bash
#$ -wd $HOME/logs
#$ -j y
#$ -pe smp 4
#$ -l h_rt=0:30:0
#$ -l h_vmem=16G
#% -l tmp_free=50G

module load miniforge
conda activate dl

python /data/EECS-LITQ/fran_storage/code/fran/fran/run/analyze_resample.py -t nodes -p 0 -n 4
```


If you are running this interactive, ray will only work if you invoke the interactive session, 
```qlogin -pe smp 8 -l h_rt=1:0:0 -l h_vmem=11G -l gpu=1```
