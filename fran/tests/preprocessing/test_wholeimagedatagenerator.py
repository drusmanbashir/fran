
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
