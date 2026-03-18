# %%
import hashlib
from typing import Union

import ipdb
from fran.utils.string_works import is_excel_None

tr = ipdb.set_trace

# read a packaged template
from importlib.resources import files
from typing import Any, Dict

import yaml
from utilz.stringz import ast_literal_eval, dec_to_str, headline, int_to_str


def short_code(input_str: Union[str, dict, None], length: int = 8) -> str | None:
    # Use SHA1 (or MD5 if you want shorter)
    if input_str is None:
        return None
    elif isinstance(input_str, dict):
        input_str = dict(sorted(input_str.items()))
        input_str = str(input_str)

    elif isinstance(input_str, list):
        input_str = str(input_str)

    input_str = input_str.replace(" ", "").lower()
    ha = hashlib.sha1(input_str.encode()).hexdigest()
    return ha[:length]


# --- helpers (order-preserving) ---
# ---------- helpers ----------
def _remove_spaces_recursive(obj: Any) -> Any:
    """Recursively remove spaces from all string keys and values in a nested structure."""
    if isinstance(obj, dict):
        return {
            (k.replace(" ", "") if isinstance(k, str) else k): _remove_spaces_recursive(
                v
            )
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_remove_spaces_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return obj.replace(" ", "")
    else:
        return obj


def load_registry() -> Dict[str, Any]:
    path = files("fran.utils").joinpath("suffix_registry.yaml")
    if not path.exists():
        raise FileNotFoundError
    raw_data = yaml.safe_load(path.read_text()) or {"codes": {}}
    return _remove_spaces_recursive(raw_data)


def nan_parser(func):
    def _inner(reg, key, val):
        if is_excel_None(val):
            return ""
        ans = func(reg, key, val)
        return ans

    return _inner


@nan_parser
def expand_by_conv(reg, key, val):
    try:
        val = int(val)
    except Exception:
        headline(val)
    key2 = reg.get(key)
    val2 = int_to_str(val, 3)
    return key2 + val2


@nan_parser
def datasources_conv(reg, key, val):
    dici = reg.get(key)
    key = "datasources"
    vals = val.replace(" ", "").split(",")
    vals = sorted(vals)
    val2 = []
    for v in vals:
        v2 = dici.get(v)
        if not v2:
            raise ValueError
        val2.append(v2)
    val3 = "".join(val2)
    return val3


@nan_parser
def remapping_conv(reg, key, val):
    val2 = reg.get(key)
    return val2.get(val)


def spacing_to_str(prefix, spacing):
    if is_excel_None(spacing):
        return ""
    spc = ast_literal_eval(spacing)
    output = [dec_to_str(val, trailing_zeros=3) for val in spc]
    spc_out = "_".join([prefix] + output)
    return spc_out


def maybe_join(vals_list):
    vals_list = [v for v in vals_list if v]
    vals_out = "_".join(vals_list)
    return vals_out


def whole_image_suffix(row):
    ps0 = row.get("patch_dim0")
    ps0 = int_to_str(ps0, 3)
    ps1 = row.get("patch_dim1")
    ps1 = int_to_str(ps1, 3)
    patch_str = "_".join(["sze", ps0, ps0, ps1])
    return patch_str


def folder_names_from_plan(project, plan: dict):
    list_to_str = lambda x: "".join(int_to_str(v, 3) for v in x)
    # Src_fodler: spacing
    # LBD_folder: src_folder,  expand_by, remapping

    reg = load_registry()
    spc = plan.get("spacing")
    src_prefix = spacing_to_str("spc", spc)

    # expand_by = expand_by_conv(reg,"expand_by",plan.get("expand_by"))
    expand_by = expand_by_conv(reg, "expand_by", plan.get("expand_by"))

    remapping_src = short_code(plan.get("remapping_source"))
    remapping_src_code = short_code(remapping_src)
    if remapping_src_code:
        remapping_src_code = "rsc" + remapping_src_code

    source_plan = plan.get("source_plan", None)
    if source_plan is not None:
        assert plan["mode"] in ["lbd", "pbd"], (
            "Folder names are not implemented with source_plan unless the mode is lbd or pbd"
        )
    source_plan_code = short_code(source_plan)

    remapping_lbd = plan.get("remapping_lbd")
    remapping_lbd_code = short_code(remapping_lbd)
    if remapping_lbd_code:
        remapping_lbd_code = "rlb" + remapping_lbd_code

    remapping_imported = plan.get("remapping_imported")
    remapping_imported_code = short_code(remapping_imported)
    if remapping_imported_code:
        remapping_imported_code = "ric" + remapping_imported_code

    source_folder_suff = maybe_join(
        [src_prefix, remapping_src_code]
    )  # note source_flder has no plan suffix
    source_folder = project.fixed_spacing_folder / source_folder_suff

    if source_plan_code:
        lbd_folder_suff = maybe_join(
            [
                src_prefix,
                remapping_lbd_code,
                remapping_imported_code,
                remapping_lbd_code,
                expand_by,
                source_plan_code,
            ]
        )

    else:
        lbd_folder_suff = maybe_join(
            [
                src_prefix,
                remapping_lbd_code,
                remapping_imported_code,
                remapping_lbd_code,
                expand_by,
            ]
        )

    lbd_folder = str(project.lbd_folder / lbd_folder_suff)
    patch_str = list_to_str(plan["patch_size"])
    if source_plan_code:
        patch_folder_suff = "_".join([source_folder_suff, patch_str, source_plan_code])
    else:
        patch_folder_suff = "_".join([source_folder_suff, patch_str])
    patch_folder = str(project.patches_folder / patch_folder_suff)
    whole_folder_suff = maybe_join([patch_str, remapping_src_code, source_plan_code])
    whole_folder = str(project.whole_images_folder / whole_folder_suff)

    folders = {
        "data_folder_source": source_folder,
        "data_folder_lbd": lbd_folder,
        "data_folder_whole": whole_folder,
        "data_folder_pbd": patch_folder,
        "data_folder_sourcepbd": source_folder,
    }
    return folders


# %%


# %%
# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
if __name__ == "__main__":
    from fran.configs.parser import ConfigMaker
    from fran.managers import Project
    from fran.utils.common import *

    P = Project("lidc")
    C = ConfigMaker(P)
    C.setup(6)
    plan = C.configs["plan_train"]
    # df = pd.read_excel("/home/ub/code/fran/configurations/experiment_configs_totalseg.xlsx", sheet_name="plans")
    # row = df.iloc[3]

    folders = folder_names_from_plan(P, plan)

    aj = plan["imported_folder"]
    pp(folders)
    dd = short_code(aj)

    # %%

    reg = load_registry()
    spc = plan.get("spacing")
    src_prefix = spacing_to_str("spc", spc)

    expand_by = expand_by_conv(reg, "expand_by", plan.get("expand_by"))
    expand_by = expand_by_conv(reg, "expand_by", plan.get("expand_by"))

    remapping_src = short_code(plan.get("remapping_source"))
    remapping_src_code = short_code(remapping_src)
    if remapping_src_code:
        remapping_src_code = "rsc" + remapping_src_code

    remapping_lbd = plan.get("remapping_lbd")
    remapping_lbd_code = short_code(remapping_lbd)
    if remapping_lbd_code:
        remapping_lbd_code = "rlb" + remapping_lbd_code

    remapping_imported = plan.get("remapping_imported")
    remapping_imported_code = short_code(remapping_imported)
    if remapping_imported_code:
        remapping_imported_code = "ric" + remapping_imported_code

    source_folder_suff = maybe_join(
        [src_prefix, remapping_src_code]
    )  # note source_flder has no plan suffix
    source_folder = project.fixed_spacing_folder / source_folder_suff

    lbd_folder_suff = maybe_join(
        [
            src_prefix,
            remapping_lbd_code,
            remapping_imported_code,
            remapping_lbd_code,
            expand_by,
        ]
    )
    lbd_folder = str(project.lbd_folder / lbd_folder_suff)

    list_to_str = lambda x: "".join(int_to_str(v, 3) for v in x)
    patch_str = list_to_str(plan["patch_size"])
    patch_folder_suff = "_".join([source_folder_suff, patch_str])
    patch_folder = str(project.patches_folder / patch_folder_suff)
    whole_folder_suff = maybe_join([patch_str, remapping_src_code])
    whole_folder = str(project.whole_images_folder / whole_folder_suff)

    folders = {
        "data_folder_source": source_folder,
        "data_folder_lbd": lbd_folder,
        "data_folder_whole": whole_folder,
        "data_folder_pbd": patch_folder,
    }

    pp(folders)
    # %%

    if isinstance(plan, dict):
        plan = {k: v for k, v in plan.items() if v is not None}
    else:
        plan = plan.dropna()
    reg = load_registry()

    project = P
# %%
# %%
