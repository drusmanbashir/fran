from fran.utils.helpers import *
import os
import itertools as il
common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
from fran.utils.config_parsers import ConfigMaker
from fran.utils.config_parsers2 import ConfigMaker2
from fran.managers.project import Project
if "get_ipython" in globals():
    print("setting autoreload")
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

