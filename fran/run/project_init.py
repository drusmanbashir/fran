# %%
import ipdb
tr = ipdb.set_trace

from fastcore.script import argparse


def main(args):
    project_title = args.t
    input_folders = args.input_folders
    P = Project(project_title=project_title); 
    print("Project: {0}".format(project_title))
    if not args.delete==True:
        P.create_project(args.input_folders)
        P.add_raw_data_sources(input_folders)
        P.populate_raw_data_folder()
        P.create_train_valid_folds()
        pp(P.proj_summary)
        P.save_summary()
        P.raw_data_imgs
    else:
        P.delete()


# %%
if __name__ == "__main__":

    from fran.utils.common import *
    parser = argparse.ArgumentParser(description="Create new project or manage existing ones")
    parser.add_argument("-t", help="project title")
    parser.add_argument("-i","--input-folders" , help="Dataset parent folder containing subfolders 'images' and 'masks'",nargs='+')
    parser.add_argument("-d" ,"--delete" ,action='store_true')

# %%
    args = parser.parse_known_args()[0]
    args.t = "litsssass"
    args.delete=True
    args.i = "/s/datasets/drli_short/"
    main(args)
# %%

