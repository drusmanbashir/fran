#!/bin/bash
# python -m ipdb project_init.py  -t lits32 -i  /s/datasets_bkp/lits_segs_improved/ /s/datasets_bkp/drli/ /s/datasets_bkp/litqsmall/  /s/xnat_shadow/litq/
# python project_init.py  -t short -i   /s/xnat_shadow/litq/
# python project_init.py  -t litstmp2 -m litsmall --multiprocess --datasources litsmall
python  project_init.py -t lidc  -m lungs  --datasources lidc
# python project_init.py  -t lungs -i  /s/datasets_bkp/Task06Lung/
