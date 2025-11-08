# %%
import argparse
from utilz.string import headline
from fran.managers.project import Project
from fran.data.dataregistry import DS
from fran.configs.parser import MNEMONICS

# very top of project_init.py (above any 3rd-party imports)
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
os.environ.setdefault("MPLBACKEND","Agg")           # matplotlib headless
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")# belt & braces
os.environ.setdefault("OPENCV_LOG_LEVEL","ERROR")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS","1")
# optional noise reducers:
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","") 
def main(args):
    headline("Arguments:")
    print(f"args.title: {args.title}")
    print(f"args.mnemonic: {args.mnemonic}")
    print(f"args.datasources: {args.datasources}")
    print(f"args.test: {args.test}")
    print(f"args.num_processes: {args.num_processes}")
    multiprocess = False if args.num_processes == 1 else True
    
    P = Project(project_title=args.title)

    if not P.db.exists():
        P.create(mnemonic=args.mnemonic)
    if args.datasources:
        datas = [DS[name] for name in args.datasources]
        P.add_data(datasources=datas, test=args.test, multiprocess=multiprocess)
    P.maybe_store_projectwide_properties(overwrite=False,multiprocess=multiprocess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage FRAN projects")
    parser.add_argument("-t", "--title", help="Project title")
    parser.add_argument("-m", "--mnemonic", help="Mnemonic, must be in MNEMONICS: {}".format(MNEMONICS))
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        help="number of parallel processes. If 1 (default), a single process is used",
        default=1,
    )

    # parser.add_argument("-n", "--num")
    parser.add_argument("--datasources", nargs="*", default=[], help="Datasources to add, i.e., {}".format(DS.__repr__()))
    parser.add_argument("--test", action="store_true", help="Mark datasources as test")
# %%
    args = parser.parse_known_args()[0]
    # args.multiprocess=False
    # args.title = 'tmp2'
    # args.mnemonic = 'litsmall'
    # args.datasources = ['litsmall']


# %%
    main(args)
# %%

