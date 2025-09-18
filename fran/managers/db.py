# %%
import ipdb
import pandas as pd
from utilz.helpers import folder_name_from_list
from utilz.string import headline

from fran.utils.config_parsers import ConfigMaker
from fran.utils.folder_names import folder_names_from_plan

tr = ipdb.set_trace
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = "plans.db"
TABLE = "master_plans"

# Fixed schema (created earlier)
# differences in these critical columns from one plan to another necessitate a new row and independent plan folder
COLUMNS_CRITICAL = [
    "datasources",
    "lm_groups",
    "spacing",
    "expand_by",
    "fg_indices_exclude",
    "mode",
    "imported_folder",
    "remapping_source_code",
    "remapping_lbd_code",
    "remapping_imported_code",
    "merge_imported_labels",
]

COLUMNS_TEXT = [
            "datasources",
            "lm_groups",
            "spacing",
            "expand_by",
            "fg_indices_exclude",
            "mode",
            "imported_folder",
            "remapping_source_code",
            "remapping_lbd_code",
            "remapping_imported_code",
            "data_folder_whole",
            "data_folder_patch",
            "data_folder_lbd",
            "data_folder_source",
        ]

COLUMNS_NONCRIT = [
    "data_folder_patch",
    "data_folder_source",
    "data_folder_lbd",
    "data_folder_whole",
]
COLUMNS_ALL = COLUMNS_CRITICAL + COLUMNS_NONCRIT
import json


def _normalize_for_db(v):
    """Convert Python values into stable TEXT/NULL for SQLite."""
    if v is None:
        return None
    if isinstance(v, (list, tuple, set)):
        if isinstance(v, set):
            v = sorted(v)
        else:
            v = list(v)
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if isinstance(v, float):
        # canonical float text to avoid 0.8 vs .8 mismatches
        return format(v, ".15g")
    if isinstance(v, Path):
        return str(v)
    return str(v)


def _init_db(db_path: str = DB_PATH):
    ddl_cols = ", ".join(
        f'"{c}" TEXT'
        for c in COLUMNS_ALL

    )
    sql = f"""
    CREATE TABLE IF NOT EXISTS "{TABLE}" (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        {ddl_cols}
    )"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(sql)
        conn.commit()


def _read_kv_excel(xlsx_path: str, sheet_name=0) -> dict:
    """Excel with 2 columns: first=key, second=value."""
    df = pd.read_excel(xlsx_path, header=None, sheet_name=sheet_name).iloc[:, :2]
    df = df.rename(columns={0: "key", 1: "value"})
    out = {}
    for _, r in df.iterrows():
        k = str(r["key"]).strip()
        if not k or k.lower() == "nan":
            continue
        v = None if pd.isna(r["value"]) else str(r["value"]).strip()
        if v == "":
            v = None
        out[k] = v
    return out


def find_matching_plan(db_path: str, plan: dict) -> dict | None:
    """Return row data if a row where all provided key->value pairs match (for known columns)."""
    plan = {k: plan.get(k) for k in COLUMNS_CRITICAL}  # align to fixed schema
    keys = [k for k in COLUMNS_CRITICAL if k in plan]
    if not keys:
        return {}
    conds, params = [], []
    for k in keys:
        v = _normalize_for_db(plan[k])
        if v is None:
            conds.append(f'"{k}" IS NULL')
        else:
            conds.append(f'"{k}" = ?')
            params.append(v)
    sql = (
        f'SELECT id, data_folder_source, data_folder_lbd, data_folder_whole, data_folder_patch FROM "{TABLE}" WHERE '
        + " AND ".join(conds)
        + " LIMIT 1"
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
    if row is None:
        print("row not found in db. Below is the sql query")
        headline(sql)
        headline (params)
        return {}

    row_out = {
        "id": row[0],
        "data_folder_source": row[1],
        "data_folder_lbd": row[2],
        "data_folder_whole": row[3],
        "data_folder_patch": row[4],
    }

    return row_out


def _insert_row(conn: sqlite3.Connection, data: dict, data_folder: str = None) -> int:
    now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    cols = ["created_at"] + COLUMNS_ALL
    vals = [now] + [_normalize_for_db(data.get(c)) for c in COLUMNS_ALL]

    placeholders = ", ".join("?" for _ in cols)
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    sql = f'INSERT INTO "{TABLE}" ({cols_sql}) VALUES ({placeholders})'
    cur = conn.execute(sql, vals)
    conn.commit()
    return cur.lastrowid

def add_plan_to_db(
    plan: dict,
    db_path: str = DB_PATH,
    data_folder_source: str = None,
    data_folder_lbd: str = None,
    data_folder_whole: str = None,
    data_folder_patch: str = None,
) -> int:

    # Assert that only one data_folder argument has a value
    data_folders = [
        data_folder_source,
        data_folder_lbd,
        data_folder_whole,
        data_folder_patch,
    ]
    non_none_count = sum(1 for folder in data_folders if folder is not None)
    assert (
        non_none_count == 1
    ), f"Exactly one data_folder argument must be provided, got {non_none_count}"

    # Combine plan with data folder information
    combined_data = plan.copy()
    combined_data.update(
        {
            "data_folder_source": data_folder_source,
            "data_folder_lbd": data_folder_lbd,
            "data_folder_whole": data_folder_whole,
            "data_folder_patch": data_folder_patch,
        }
    )

    headline("Adding plan to db: {0}".format(db_path))
    rid = find_matching_plan(db_path, plan)
    if rid is not None:
        print("row exists in db data_folder:", rid)
        return rid
    with sqlite3.connect(db_path) as conn:
        return _insert_row(conn, combined_data, None)



def as_dataframe(
    db_path: str,
    table: str = "master_plans",
    where: str | None = None,
    params: tuple = (),
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Load a SQLite table into a pandas DataFrame and print it.

    where: optional SQL WHERE clause without the 'WHERE' keyword, e.g. '"mode" = ?'
    params: parameters tuple for the WHERE clause placeholders
    limit: optional LIMIT
    """
    sql = f'SELECT * FROM "{table}"'
    if where:
        sql += " WHERE " + where
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    # print nicely and return
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
# %%
# SECTION:-------------------- setup-------------------------------------------------------------------------------------- <CR>
    from fran.utils.common import *

    P = Project("litstmp")
    # P._create_plans_table()
    P = Project("nodes")
    # P.create("lits")

    # P.add_data([DS.litq, DS.lits, DS.drli, DS.litqsmall])
    # P.create('litsmc')
    C = ConfigMaker(P, raytune=False, configuration_filename=None)
    C.setup(6)
    plan = C.configs['plan_train']

# %%
    conn = sqlite3.connect(P.db)
    cur = conn.cursor()
    ss= """DROP TABLE IF EXISTS master_plans"""
    cur.execute(ss)
    conn.commit()
    ss = """ALTER TABLE master_plans ADD COLUMN remapping_imported TEXT"""
    ss = """ALTER TABLE master_plans ADD COLUMN remapping_imported TEXT"""
    ss = """ALTER TABLE master_plans ADD COLUMN  imported_folder TEXT"""
    cur.execute(ss)
    ss = """ALTER TABLE master_plans ADD COLUMN  merge_imported_labels INTEGER DEFAULT 0"""
    cur.execute(ss)
    conn.commit()
# %%
    plan = C.conf['plan_train']
    folder_names_from_plan(P, plan)

# %%
#SECTION:-------------------- DELETE ROW BASED ON COL VALUE MATCH--------------------------------------------------------------------------------------
    conn = sqlite3.connect(P.db)
    sql = """DELETE FROM master_plans WHERE data_folder_source  ='/r/datasets/preprocessed/litstmp/fixed_spacing/spc_080_080_150_drllqlqslts'"""
    cur = conn.cursor()
    cur.execute(sql)


    pathdad = Path("/s/fran_storage/projects")
    db_paths = list(pathdad.rglob("*.db"))

    cols_text = [
        "datasources",
        "lm_groups",
        "spacing",
        "expand_by",
        "fg_indices_exclude",
        "mode",
        "imported_folder",
        "remapping_source",
        "remapping_lbd",
        "remapping_imported",
        "data_folder_whole",
        "data_folder_patch",
        "data_folder_lbd",
        "data_folder_source",
    ]
    cols_text = [c + " TEXT" for c in cols_text]

    cols_BOOL = ["merge_imported_labels"]
    cols_BOOL = [c + " INTEGER DEFAULT 0" for c in cols_BOOL]

    all_cols = ", ".join(cols_text + cols_BOOL)
    all_cols = cols_BOOL + cols_text
# %%
    for dbpath in db_paths:
        cur = sqlite3.connect(dbpath)
        print("connecting to {}".format(dbpath))

        sql = f"""
        CREATE TABLE IF NOT EXISTS master_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            {all_cols}
        )"""

        cur.execute(sql)

# %%
    for dbpath in db_paths:
        cur = sqlite3.connect(dbpath)
        for col in all_cols:
            try:
                con = cur.cursor()
                ss = f"""ALTER TABLE master_plans ADD COLUMN {col} """
                con.execute(ss)
            except Exception as e:
                print(e)

        try:
            con = cur.cursor()
            ss = (
                """ALTER TABLE master_plans RENAME COLUMN remapping to remapping_lbd """
            )
            con.execute(ss)
        except Exception as e:
            print(e)
        cur.close()

# %%
    sql = "DELETE FROM master_plans WHERE data_folder_source LIKE '/s/fran%'"
    sql = "PRAGMA table_info(master_plans)"
    db_paths
    dbpath = db_paths[4]

# %%
    conn = sqlite3.connect(dbpath, timeout=30)  # timeout gives it time if locked
    cur = conn.cursor()
    sql = "DELETE FROM master_plans WHERE data_folder_source LIKE '/s/fran%'"
    cur.execute(sql)
    conn.commit()
    conn.close()
# %%
    con = sqlite3.connect(dbpath)
    cur = con.cursor()
    cur.execute(sql)
    for row in cur.fetchall():
        print(row)
    con.close()
# %%
    con.commit()
# %%
    from fran.managers import Project
    from fran.utils.common import *

    P = Project("totalseg")
    # P._create_plans_table()
    # P.add_data([DS.totalseg])
    C = ConfigMaker(P, raytune=False, configuration_filename=None)
    C.setup(1, 1)
    C.plans
    C.set_active_plans(6)
    plan = C.configs["plan_train"]
    plan
    conf = C.configs
    print(conf["model_params"])

    plan = conf["plan_train"]
    print(plan)
    plan["mode"]

# %%
    parent_folder = P.fixed_spacing_folder
    output_folder = folder_name_from_list(
        prefix="spc",
        parent_folder=parent_folder,
        values_list=plan["spacing"],
    )
# %%
    plan_name = plan.get("plan_name")
    output_name = "_".join([output_folder.name, plan_name])
    output_folder = Path(output_folder.parent / output_name)

# SECTION:-------------------- FINE-TUNING RUN-------------------------------------------------------------------------------------- <CR>
    bs = 14  # is good if LBD with 2 samples per case
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    tags = []
    description = None

    # device_id = 1
# %%


# %%
    #
    #     """Return id of a row where all provided key->value pairs match (for known columns)."""
    #     plan = {k: plan.get(k) for k in COLUMNS_CRITICAL}  # align to fixed schema
    #     keys = [k for k in COLUMNS_CRITICAL if k in plan]
    #     if not keys:
    #         return None
    #     conds, params = [], []
# %%
    #     for k in keys:
    #         v = _normalize_for_db(plan[k])
    #         if v is None:
    #             conds.append(f'"{k}" IS NULL')
    #         else:
    #             conds.append(f'"{k}" = ?')
    #             params.append(v)
    #     sql = (
    #         f'SELECT data_folder_source, data_folder_lbd, data_folder_whole, data_folder_patch FROM "{TABLE}" WHERE '
    #         + " AND ".join(conds)
    #         + " LIMIT 1"
    #     )
# %%
    #     db_path = P.db
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
# %%

    """Return row data if a row where all provided key->value pairs match (for known columns)."""
    plan = {k: plan.get(k) for k in COLUMNS_CRITICAL}  # align to fixed schema
    keys = [k for k in COLUMNS_CRITICAL if k in plan]
    conds, params = [], []
    for k in keys:
        v_norm = _normalize_for_db(plan[k])  # keep your normalizer
        if v_norm is None:
            conds.append(f'"{k}" IS NULL')
        elif _is_zeroish(v_norm):  # must handle "0", 0, 0.0
            conds.append(f'("{k}" = ? OR "{k}" IS NULL)')
            params.append("0")     # TEXT schema → compare as string
        else:
            conds.append(f'"{k}" = ?')
            params.append(v_norm)

    sql = (
        f'SELECT id, data_folder_source, data_folder_lbd, data_folder_whole, data_folder_patch '
        f'FROM "{TABLE}" WHERE ' + " AND ".join(conds) + " LIMIT 1"
    )
# %%
    import sqlite3
    db_path = P.db
    sql = 'UPDATE master_plans SET expand_by = 0 WHERE data_folder_source="/r/datasets/preprocessed/nodes/fixed_spacing/spc_080_080_150_ndndt"'
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(sql)
    con.commit()
# %%

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
# %%


    aa = {
        "id": row[0],
        "data_folder_source": row[1],
        "data_folder_lbd": row[2],
        "data_folder_whole": row[3],
        "data_folder_patch": row[4],
    }

# %%
    data_folder_source = I.R.output_folder
    data_folder_lbd = None
    data_folder_whole = None
    data_folder_patch = None

    data_folders = [
        data_folder_source,
        data_folder_lbd,
        data_folder_whole,
        data_folder_patch,
    ]
    non_none_count = sum(1 for folder in data_folders if folder is not None)
    assert (
        non_none_count == 1
    ), f"Exactly one data_folder argument must be provided, got {non_none_count}"

# %%
# Determine which data folder field is being set
    data_folder_field = None
    data_folder_value = None
    if data_folder_source is not None:
        data_folder_field = "data_folder_source"
        data_folder_value = data_folder_source
    elif data_folder_lbd is not None:
        data_folder_field = "data_folder_lbd"
        data_folder_value = data_folder_lbd
    elif data_folder_whole is not None:
        data_folder_field = "data_folder_whole"
        data_folder_value = data_folder_whole
    elif data_folder_patch is not None:
        data_folder_field = "data_folder_patch"
        data_folder_value = data_folder_patch

    folder_names = folder_names_from_plan(P,plan)
    folder_names[data_folder_field] = data_folder_value
# %%

