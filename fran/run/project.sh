#!/bin/bash
# python -m ipdb project_init.py  -t lits32 -i  /s/datasets_bkp/lits_segs_improved/ /s/datasets_bkp/drli/ /s/datasets_bkp/litqsmall/  /s/xnat_shadow/litq/
# python project_init.py  -t short -i   /s/xnat_shadow/litq/
# python project_init.py  -t litstmp2 -m litsmall --multiprocess --datasources litsmall
# python  project_init.py -t lidc  -m lungs  --datasources lidc
# python  -m ipdb project_init.py -t nodes  --mnemonic nodes  --datasources nodes nodesthick
# python  -m ipdb project_init.py -t totalseg  --mnemonic totalseg  --datasources totalseg
# python   -m ipdb project_init.py -t bones  --mnemonic bones  --datasources uls23_bone
# python  analyze_resample.py -t bones -p 1
# python  analyze_resample.py -t totalseg -p 0
# python project_init.py  -t lungs -i  /s/datasets_bkp/Task06Lung/

python  analyze_resample.py -t colon -p 0
python   project_init.py -t colon  --mnemonic colon  --datasources colonmsd10
