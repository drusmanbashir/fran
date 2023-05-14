# %%
from fastcore.basics import store_attr
import pandas as pd
import ast
from ray import tune
from openpyxl import load_workbook

from fran.utils.helpers import *

BOOL_ROWS='patch_based,one_cycles,heavy,deep_supervision,self_attention,fake_tumours,square_in_union,apply_activation'
def check_bool(row):
    if row['var_name'] in BOOL_ROWS.split(','):
        row['manual_value']= bool(row['manual_value'])
    return row

def make_patch_size(patch_dim0,patch_dim1):
    patch_size = [patch_dim0,]*2+[patch_dim1,]
    return patch_size

def out_channels_from_dict_or_cell(src_dest_labels):  
    if isinstance(src_dest_labels, pd.core.series.Series):
        src_dest_labels = ast.literal_eval(src_dest_labels.item())
    out_channels = max([src_dest[1] for src_dest in src_dest_labels])+1
    return out_channels


def get_imagelists_from_config(proj_defaults, fold, patch_based, dim0, dim1):
    json_fname = proj_defaults.validation_folds_filename
    if patch_based == False:
        folder_name = proj_defaults.stage1_folder / "{0}_{1}_{1}/images".format(
            dim0, dim1, dim1
        )
        train_list, valid_list, _ = get_train_valid_test_lists_from_json(
            project_title=proj_defaults.project_title,
            fold=fold,
            json_fname=json_fname,
            image_folder=folder_name,
            ext=".pt",
        )
        print("Retrieved whole image datasets from folder: {}".format(folder_name))
    else:
        train_list, valid_list, _ = get_train_valid_test_lists_from_json(
            project_title=proj_defaults.project_title,
            fold=fold,
            image_folder=proj_defaults.stage1_folder / "cropped/images_pt/images",
            json_fname=json_fname,
            ext=".pt",
        )

    return train_list, valid_list


def resolve_tune_fnc(tune_type: str):
    if "_" in tune_type:
        return getattr(tune, tune_type.split("_")[0])
    else:
        return getattr(tune, tune_type)


def load_config_from_worksheet(settingsfilename, sheet_name, raytune, engine="pd"):
    """
    Reads a sheet row-by-row
    """

    if engine == "pd":
        df = pd.read_excel(settingsfilename, sheet_name=sheet_name)
    elif engine == "openpyxl":  # THIS IS NOT PROPERLY IMPLEMENTED AND HAS BUGS
        wb = load_workbook(settingsfilename)
        sheet = wb[sheet_name]
        df = pd.DataFrame(sheet.values)

    else:
        raise NotImplementedError
    config = {}
    for row in df.iterrows():
        rr = row[1]
        rr = check_bool(rr)
        var_type = rr["tune_type"]
        key = rr["var_name"]
        if sheet_name == "after_item_intensity":
            if raytune == False or rr["tune"] == False:
                val = parse_excel_cell(rr["manual_value"])
                prob = rr["manual_p"]
            else:
                if rr["tune_type"] == "double_range":
                    val_lower, val_upper = parse_excel_cell(rr["tune_value"])
                    val_lower = tune.uniform(lower=val_lower[0], upper=val_lower[1])
                    val_upper = tune.uniform(lower=val_upper[0], upper=val_upper[1])
                    val = [val_lower, val_upper]

                    prob = parse_excel_cell(rr["tune_p"])
                    prob = tune.uniform(lower=prob[0], upper=prob[1])
            config.update({key: [val, prob]})
        elif sheet_name == "after_item_spatial":
            if raytune == False or rr["tune"] == False:
                prob = rr["manual_p"]
            else:
                prob = parse_excel_cell(rr["tune_p"])
                prob = tune.uniform(lower=prob[0], upper=prob[1])
            config.update({key: prob})
        else:
            if raytune == False or rr["tune"] == False:
                val = parse_excel_cell(rr["manual_value"])
                config.update({key: parse_excel_cell(val)})
            else:
                if "spec" in var_type:
                    pass
                else:
                    val = parse_excel_cell(rr["tune_value"])

                    if "_" in var_type:
                        tr()
                        vals = parse_excel_cell(rr["tune_value"])
                        var_type.split("_")
                        tune_fnc = resolve_tune_fnc(var_type)
                        val_sample = [tune_fnc(val[0], val[1]) for val in vals]
                    elif (
                        var_type == "randint"
                        or var_type == "loguniform"
                        or var_type == "uniform"
                    ):
                        tune_fnc = resolve_tune_fnc(var_type)
                        val_sample = tune_fnc(val[0], val[1])
                    elif rr["tune_type"] == "double_range":
                        val_lower, val_upper = parse_excel_cell(rr["tune_value"])
                        val_lower = tune.uniform(lower=val_lower[0], upper=val_lower[1])
                        val_upper = tune.uniform(lower=val_upper[0], upper=val_upper[1])
                        val_sample = [val_lower, val_upper]

                    elif rr["tune_type"] == "choice":
                        val_sample = tune.choice(val)
                    else:
                        tune_fnc = resolve_tune_fnc(var_type)
                        if tune_fnc.__name__[0] == "q":
                            val.append(rr["quant"])
                        val_sample = tune_fnc(*val)
                    config.update({key: val_sample})
    return config


class ConfigMaker:
    def __init__(self, proj_defaults, raytune):

        store_attr()

        self.config = load_config_from_workbook(proj_defaults.configuration_filename, raytune)
        if not "mom_low" in self.config["model_params"].keys() and raytune==True:

            config = {
                "mom_low": tune.sample_from(
                    lambda spec: np.random.uniform(0.6, 0.9100)
                ),
                "mom_high": tune.sample_from(
                    lambda spec: np.minimum(
                        0.99,
                        spec.config.model_params.mom_low
                        + np.random.uniform(low=0.05, high=0.35),
                    )
                ),
            }
            self.config["model_params"].update(config)
        self.add_further_keys()
      


    def add_further_keys(self):
        self.add_out_channels()
        self.add_patch_size()
        self.add_dataset_props()


    def add_dataset_props(self):
        global_properties = load_dict(self.proj_defaults.global_properties_filename)
        self.config['dataset_params']['clip_range']=global_properties["intensity_clip_range"]
        self.config['dataset_params']['mean_fg']=global_properties["mean_fg"]
        self.config['dataset_params']['std_fg']=global_properties["std_fg"]

    def add_out_channels(self):
        if not 'out_channels' in self.config["model_params"]:

            self.config['model_params']["out_channels"] = out_channels_from_dict_or_cell(
                self.config['metadata']["src_dest_labels"]
            )

    def add_patch_size(self):
        if not "patch_size" in self.config['dataset_params']:
                    self.config['dataset_params']["patch_size"] = make_patch_size(
                        self.config['dataset_params']["patch_dim0"], self.config['dataset_params']["patch_dim1"]
                    )
                    


def load_config_from_workbook(settingsfilename, raytune):
    wb = load_workbook(settingsfilename)
    sheets = wb.sheetnames
    # sheets.remove("metadata")
    configs_dict = {
        sheet: load_config_from_worksheet(settingsfilename, sheet, raytune)
        for sheet in sheets
    }
    return configs_dict


def parse_excel_cell(cell_val):
    if isinstance(cell_val, (int, float)):
        return cell_val
    if isinstance(cell_val, str):
        if cell_val in ["None"]:
            return ast.literal_eval(cell_val)

        if any(pattern in cell_val for pattern in ("[", "{", "(")):
            try:
                cell_val = ast.literal_eval(cell_val)
                return cell_val
            except:
                return cell_val
            return
        else:
            return cell_val
    else:
        return cell_val


def load_metadata(settingsfilename):
    df = pd.read_excel(settingsfilename, sheet_name="metadata", index_col=None)
    return df


def parse_neptune_dict(dic: dict):
    # fixes lists of int appearing in strings
    for kk, vv in dic.items():
        vv = parse_excel_cell(vv)
        dic.update({kk: vv})
    return dic


# %%

if __name__ == "__main__":

    from fran.utils.common import *
    P = Project(project_title="lits"); proj_defaults= P
    proj_defaults = proj_defaults.configuration_filename
    wb = load_workbook(proj_defaults)
    sheets = wb.sheetnames
    mode = "manual"
    meta = load_metadata(proj_defaults)
    sheet_name = "after_item_intensity"
    trans = load_config_from_worksheet(
        proj_defaults, "after_item_intensity", raytune=True
    )
    spat = load_config_from_worksheet(
        proj_defaults, "after_item_spatial", raytune=True
    )
    met = load_config_from_worksheet(proj_defaults, "metadata", raytune=True)



# %%



    config = ConfigMaker(proj_defaults.configuration_filename, raytune=False).config
# %%
    wb = load_workbook(proj_defaults)
    sheets = wb.sheetnames
    metadata = wb["metadata"]
    dat = metadata[2 : metadata.max_row]
    # sheets.remove("metadata")
    df = pd.DataFrame(metadata.values)
    df2 = pd.DataFrame(dat.values)
    raytune = False

    configs_dict = {
        sheet: load_config_from_worksheet(proj_defaults, sheet, raytune)
        for sheet in sheets
    }

    df = pd.read_excel(proj_defaults, sheet_name="metadata", dtype=str)
    df
    # %%

    config = {
        # A random function
        "alpha": tune.sample_from(lambda _: np.random.uniform(100)),
        # Use the `spec.config` namespace to access other hyperparameters
        "beta": tune.sample_from(lambda spec: spec.config.alpha * np.random.normal())
        # %%
    }
    config["mom_low"].sample()
    config["mom_high"].sample()
    # %%
    tune.sample_from(lambda _: np.random.uniform(100) ** 2).sample()
    # %%
    config = load_config_from_workbook(proj_defaults, raytune=True)
    if not "mom_low" in config["model_params"].keys():
        conds = {
            "mom_low": tune.sample_from(
                lambda spec: np.random_uniform(low=0.6, high=0.9)
            ),
            "mom_high": tune.sample_from(
                lambda spec: np.maximum(
                    0.99, spec.conds.mom_low + np.ranomd.uniform(low=0.5, high=0.35)
                )
            ),
        }
        config["model_params"].update(conds)
    # %%
    config["model_params"]["mom_low"].sample()
    config["model_params"]["mom_added"].sample()

# %%
