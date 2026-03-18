import os

# set_autoreload()
from utilz.fileio import load_yaml

common_vars_filename = os.environ["FRAN_CONF"] + "/config.yaml"
PAD_VALUE = -123

COMMON_PATHS = load_yaml(common_vars_filename)
# WARN: DO NOT ADD Project or COnfig imports!!
