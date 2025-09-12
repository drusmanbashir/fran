# %%
import argparse
from fran.managers.project import Project
from fran.managers.datasource import DS,MNEMONICS


def main(args):
    P = Project(project_title=args.title)

    if not P.db.exists():
        P.create(mnemonic=args.mnemonic)
    if args.datasources:
        datas = [DS[name] for name in args.datasources]
        P.add_data(datasources=datas, test=args.test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage FRAN projects")
    parser.add_argument("-t", "--title", help="Project title")
    parser.add_argument("-m", "--mnemonic", help="Mnemonic, must be in MNEMONICS: {}".format(MNEMONICS))
    parser.add_argument("--datasources", nargs="*", default=[], help="Datasources to add, i.e., {}".format(DS.__repr__()))
    parser.add_argument("--test", action="store_true", help="Mark datasources as test")
# %%
    args = parser.parse_known_args()[0]
    args.title = 'tmpja'
    args.mnemonic = 'liver'
    args.datasources = ['drli_short']

# %%
    main(args)
# %%

