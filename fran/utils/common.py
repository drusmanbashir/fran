import os
from utilz.helpers import set_autoreload
# set_autoreload()
from utilz.fileio import load_yaml
import os
from pprint import pprint as pp
common_vars_filename = os.environ["FRAN_COMMON_PATHS"]+"/config.yaml"


COMMON_PATHS = load_yaml(common_vars_filename)
#WARN: DO NOT ADD Project or COnfig imports!!
