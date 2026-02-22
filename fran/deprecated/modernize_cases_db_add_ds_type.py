#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Modernize a FRAN project cases.db by ensuring datasources.ds_type exists "
            "and filling all rows with a user-provided value."
        )
    )
    p.add_argument(
        "--db",
        required=True,
        help="Path to project cases.db (for example: /s/fran_storage/projects/<project>/cases.db)",
    )
    p.add_argument(
        "--ds-type",
        required=True,
        choices=["full", "patch"],
        help="Value to write into datasources.ds_type for all rows.",
    )
    return p.parse_args()


def has_column(cur, table, column):
    cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return any(c[1] == column for c in cols)


def main():
    args = parse_args()
    db = Path(args.db)
    if not db.exists():
        raise FileNotFoundError(f"cases.db not found: {db}")

    con = sqlite3.connect(str(db))
    try:
        cur = con.cursor()
        tbl_exists = cur.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name='datasources'"
        ).fetchone()
        if not tbl_exists:
            raise RuntimeError("Table 'datasources' not found in DB.")

        if not has_column(cur, "datasources", "ds_type"):
            cur.execute("ALTER TABLE datasources ADD COLUMN ds_type TEXT")

        cur.execute("UPDATE datasources SET ds_type = ?", (args.ds_type,))
        con.commit()

        total = cur.execute("SELECT COUNT(*) FROM datasources").fetchone()[0]
        print(f"Updated {total} rows in {db} with ds_type='{args.ds_type}'.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
