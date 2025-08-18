# %%

import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = "plans.db"
TABLE = "master_plans"

# ---- helpers --------------------------------------------------------------

def read_kv_excel(xlsx_path: str, sheet_name=0) -> dict[str, str | None]:
    """Excel with 2 columns: first=key, second=value."""
    df = pd.read_excel(xlsx_path, header=None, sheet_name=sheet_name).iloc[:, :2]
    df = df.rename(columns={0: "key", 1: "value"})
    kv = {}
    for _, r in df.iterrows():
        k = str(r["key"]).strip()
        if not k or k.lower() == "nan":
            continue
        v = None if pd.isna(r["value"]) else str(r["value"]).strip()
        kv[k] = v
    return kv

def get_columns(conn) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{TABLE}")').fetchall()]

def ensure_table_and_columns(conn: sqlite3.Connection, keys: list[str]):
    # create table (id + created_at only; data columns added below)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "{TABLE}" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL
        )
        """
    )
    # add missing columns as TEXT
    existing = set(get_columns(conn))
    for k in keys:
        if k not in existing:
            conn.execute(f'ALTER TABLE "{TABLE}" ADD COLUMN "{k}" TEXT')
    conn.commit()

def find_matching_row(conn: sqlite3.Connection, data: dict) -> int | None:
    if not data:
        return None
    conds, params = [], []
    for k, v in data.items():
        if v is None:
            conds.append(f'"{k}" IS NULL')
        else:
            conds.append(f'"{k}" = ?')
            params.append(v)
    sql = f'SELECT id FROM "{TABLE}" WHERE ' + " AND ".join(conds) + " LIMIT 1"
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None

def quote_ident(name: str) -> str:
    # SQLite identifier quoting with double quotes
    return '"' + name.replace('"', '""') + '"'

def insert_row(conn: sqlite3.Connection, data: dict) -> int:
    cols = [c for c in get_columns(conn) if c != "id"]
    now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")

    values = []
    for c in cols:
        if c == "created_at":
            values.append(now)
        else:
            values.append(data.get(c))  # None -> NULL

    cols_sql = ", ".join(quote_ident(c) for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    sql = f'INSERT INTO {quote_ident(TABLE)} ({cols_sql}) VALUES ({placeholders})'

    cur = conn.execute(sql, values)
    conn.commit()
    return cur.lastrowid
def upsert_plan_from_excel(xlsx_path: str, sheet_name=0) -> int:
    data = read_kv_excel(xlsx_path, sheet_name)
    with sqlite3.connect(DB_PATH) as conn:
        ensure_table_and_columns(conn, list(data.keys()))
        # If a row exists where ALL provided key→value pairs match, reuse it.
        match_id = find_matching_row(conn, data)
        if match_id is not None:
            return match_id
        # Otherwise, insert a new row (missing keys become NULL).
        return insert_row(conn, data)
# %%
# ---- example without Excel (your provided pairs) --------------------------

if __name__ == "__main__":
    DB_PATH= "plans.db"
    sample = {
        "var_name": "manual_value",
        "datasources": "lits,drli,litq,litqsmall",
        "lm_groups": None,
        "spacing": "0.8,.8,1.5",
        "expand_by": "40",
        "fg_indices_exclude": "1",
        "mode": "lbd",
    }
    with sqlite3.connect(DB_PATH) as conn:
        ensure_table_and_columns(conn, list(sample.keys()))
        row_id = find_matching_row(conn, sample)
        if row_id is None:
            row_id = insert_row(conn, sample)
    print("row_id:", row_id)

# %%

# %%
