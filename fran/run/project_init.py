from fastcore.script import argparse


def main(args):
    project_title = args.t
    P = Project(project_title=project_title); proj_defaults= P.proj_summary
    print("Project: {0}".format(project_title))
    P.save_summary()
    P.create_project(args.input_folders)
    P.populate_raw_data_folder()
    P.raw_data_imgs


# %%
if __name__ == "__main__":

    from fran.utils.common import *
    parser = argparse.ArgumentParser(description="Create new project or manage existing ones")
    parser.add_argument("t", help="project title")
    parser.add_argument("-i","--input-folders" , help="Folders containing nifti files you wish to add to this project")
# %%

