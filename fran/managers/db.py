# %%
import pandas as pd
import ipdb
tr = ipdb.set_trace
from pathlib import Path

import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = "plans.db"
TABLE = "master_plans"

# Fixed schema (created earlier)
PLAN_COLUMNS = [
    "datasources",
    "lm_groups",
    "spacing",
    "expand_by",
    "fg_indices_exclude",
    "mode",
    "remapping",
    # "samples_per_file",
    # "src_dest_labels",
]

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
    ddl_cols = ", ".join(f'"{c}" TEXT' for c in PLAN_COLUMNS+["data_folder", "derived_plans"])
    sql = f'''
    CREATE TABLE IF NOT EXISTS "{TABLE}" (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        {ddl_cols}
    )'''
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

def find_matching_plan(db_path: str, plan: dict) -> int | None:
        """Return id of a row where all provided key->value pairs match (for known columns)."""
        plan = {k: plan.get(k) for k in PLAN_COLUMNS}  # align to fixed schema
        keys = [k for k in PLAN_COLUMNS if k in plan]
        if not keys:
            return None
        conds, params = [], []
        for k in keys:
            v = _normalize_for_db(plan[k])
            if v is None:
                conds.append(f'"{k}" IS NULL')
            else:
                conds.append(f'"{k}" = ?')
                params.append(v)
        sql = f'SELECT data_folder FROM "{TABLE}" WHERE ' + " AND ".join(conds) + " LIMIT 1"
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(sql, params).fetchone()
        return row[0] if row else None

def _insert_row(conn: sqlite3.Connection, data: dict,data_folder:str) -> int:
    now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    cols = ["created_at"] + PLAN_COLUMNS + ["data_folder"]
    vals = [now] + [_normalize_for_db(data.get(c)) for c in PLAN_COLUMNS]
    vals+= [str(data_folder)]
    placeholders = ", ".join("?" for _ in cols)
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    sql = f'INSERT INTO "{TABLE}" ({cols_sql}) VALUES ({placeholders})'
    cur = conn.execute(sql, vals)
    conn.commit()
    return cur.lastrowid

def add_plan_to_db(plan: dict,data_folder:str, db_path: str = DB_PATH) -> int:
    rid = find_matching_plan(db_path, plan)
    if rid is not None:
            print("row exists in db data_folder:", rid)
            return rid
    with sqlite3.connect(db_path) as conn:
        return _insert_row(conn, plan,data_folder)


def as_dataframe(db_path: str ,
                       table: str = "master_plans",
                       where: str | None = None,
                       params: tuple = (),
                       limit: int | None = None) -> pd.DataFrame:
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

