import os
from fran.utils.helpers import set_autoreload
set_autoreload()
from fran.utils.fileio import load_yaml
import os
common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
COMMON_PATHS = load_yaml(common_vars_filename)
