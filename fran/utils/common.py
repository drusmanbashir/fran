from fran.utils.helpers import *
import os
import itertools as il

from fran.utils.helpers import set_autoreload
common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
from fran.utils.config_parsers import ConfigMaker
from fran.managers.project import Project
set_autoreload()

