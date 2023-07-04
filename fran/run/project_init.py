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
        if args.input_folders:
            P.add_datasources(input_folders)
            P.populate_raw_data_folder()
        P.raw_data_imgs
        if args.update_folds==True:
            P.update_folds()
        else:
            P.create_train_valid_folds()
    else:
        P.delete()


# %%
if __name__ == "__main__":

    from fran.utils.common import *
    parser = argparse.ArgumentParser(description="Create new project or manage existing ones")
    parser.add_argument("-t", help="project title")
    parser.add_argument("-i","--input-folders" , help="Dataset parent folder containing subfolders 'images' and 'masks'",nargs='+')
    parser.add_argument("-d" ,"--delete" ,action='store_true')
    parser.add_argument("-u" ,"--update-folds" ,action='store_true')

# %%
    args = parser.parse_known_args()[0]
    # args.t = "l2"
    # args.delete=True
    # args.i = "/s/datasets/drli_short/"
    main(args)
# %%

