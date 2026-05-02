#!/bin/bash
# /home/ub/code/fran/fran/run/misc/imageviewer.sh /tmp/image.pt
# /home/ub/code/fran/fran/run/misc/imageviewer.sh /tmp/image.nii.gz /tmp/label.nii.gz
exec /home/ub/mambaforge/envs/dl/bin/python /home/ub/code/fran/fran/run/misc/view_image.py "$@"
