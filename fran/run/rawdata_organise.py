
# %%
import shutil
from pathlib import Path
import argparse


from fastcore.all import store_attr
from utilz.fileio import save_list
from utilz.helpers import multiprocess_multiarg


# %%
def copy_src_dest(img_src_fn:Path , mask_src_fn:Path, img_dest_fn,mask_dest_fn):
    for src_fn , dest_fn in zip([img_src_fn,mask_src_fn],[img_dest_fn,mask_dest_fn]):
            shutil.copy(src_fn,dest_fn)

class PrepRawData():
    def __init__(self,dataset_name,imgs_folder:Path,masks_folder:Path,img_id,mask_id,output_folder:Path):
        store_attr(but='imgs_folder,masks_folder')
        self.img_src_files = self.img_folder.glob("*")
        self.mask_src_files = self.masks_folder.glob("*")

    def verify_img_mask_pairs(self):
        self.src_file_pairs= []
        for mask_fn in self.mask_src_files:
            corresp_img_fn = [img_fn for img_fn in self.img_src_files if img_fn.name.replace(self.mask_id,self.img_id)==mask_fn.name]
            missing_imgs= []
            assert len(corresp_img_fn)<2,"Multiple matching image files {}. Please reconcile and start again.".format(corresp_img_fn)
            if len (corresp_img_fn)==0:
                missing_imgs.append(mask_fn)
            else:
                self.src_file_pairs.append([corresp_img_fn[0],mask_fn])
        if len(missing_imgs)>0:
                error_fn = self.proj_defaults['log_folder']/("missing_files.txt")
                print("Some mask file names couldn't be matched. Names of those masks are stored {}/missing_files.txt".format(error_fn))
                save_list(error_fn)
        else:
            print("All files matched. Proceed with copying files to project folders")

    def create_output_filenames(self):
        self.dest_file_pairs=[]
        for fn_pair in self.src_file_pairs:
            dest_pair = [fn.str_replace(id,self.dataset_name) for fn,id in zip(fn_pair,[self.img_id,self.mask_id])]
            self.dest_file_pairs.append(dest_pair)

    def rename_and_copy(self):
        args = [[*src_pair,*dest_pair] for src_pair, dest_pair in zip(self.src_file_pairs,self.dest_file_pairs)]
        multiprocess_multiarg(copy_src_dest, args,self.num_processes,debug=self.debug,io=True)


class PrepRawDataLITS(PrepRawData):
    '''
    All img and mask files are same folder but different id
    This class moves them into separate images / masks folders inside the project raw_data_folder
    '''
    def __init__(self,args):
        dataset_name ='lits'
        self.mask_id = 'segmentation'
        self.img_id = 'volume'
        self.imgs_folder = self.masks_folder = Path(args.input_folder)
        self.output_folder = Path(args.output_folder)
        super().__init__(dataset_name , self.imgs_folder,self.masks_folder,self.img_id,self.mask_id,self.output_folder)

def main(args):
    if args.t =='lits':
        P = PrepRawDataLITS(args)
    P.verify_img_mask_pairs()
    P.create_output_filenames()
    P.rename_and_copy()
# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Organises downloaded data and renames it to fran-compatible folder and filenames. Currently supports LITS and KITS only.")
    parser.add_argument("-t", help="project title", dest="project_title")
    parser.add_argument("-i","input-folder", help="input folder")
    parser.add_argument("-o","--output-folder", help="output folder") 
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        help="number of parallel processes",
        default=24,
    )
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_known_args()[0]

    main(args)

