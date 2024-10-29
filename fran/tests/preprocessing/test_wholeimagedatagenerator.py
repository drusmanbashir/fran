import pytest
from fran.preprocessing import WholeImageDataGenerator
from fran.utils.string import ast_literal_eval
from fran.utils.common import *

from fran.utils.config_parsers import ConfigMaker
from fran.managers import Project


def test_wholeimagedatagenerator():
    P = Project(project_title="totalseg")
    spacing = [0.8, 0.8, 1.5]
    P.maybe_store_projectwide_properties()

# %%
#SECTION:--------------------SETUP --------------------------------------------------------------------------------------

if __name__ == '__main__':
# %%
    P = Project(project_title="totalseg")
    spacing = [0.8, 0.8, 1.5]
    P.maybe_store_projectwide_properties()
