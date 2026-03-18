#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from pathlib import Path

import h5py


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Ensure datasources.labels/nnz columns exist in cases.db and populate them "
            "from datasource fg_voxels.h5 attrs['labels'] per case_id."
        )
    )
    p.add_argument(
        "--db",
        help=(
            "Path to project cases.db (for example: "
            "/s/fran_storage/projects/<project>/cases.db)."
        ),
    )
    p.add_argument(
        "--project",
        help="Project title. If set, DB resolves to <projects-folder>/<project>/cases.db.",
    )
    p.add_argument(
        "--projects-folder",
        default=None,
        help=(
            "Projects root folder override. If omitted, script tries FRAN_CONF/config.yaml "
            "for projects_folder, then falls back to /s/fran_storage/projects."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show intended updates without writing to DB.",
    )
    args = p.parse_args()
    if not args.db and not args.project:
        p.error("Provide one of --db or --project.")
    return args


def load_projects_folder(cli_value):
    if cli_value:
        return Path(cli_value)

    fran_conf = os.environ.get("FRAN_CONF")
    if fran_conf:
        cfg = Path(fran_conf) / "config.yaml"
        if cfg.exists():
            try:
                import yaml

                data = yaml.safe_load(cfg.read_text()) or {}
                pf = data.get("projects_folder")
                if pf:
                    return Path(pf)
            except Exception:
                pass

    return Path("/s/fran_storage/projects")


def resolve_db_path(args):
    if args.db:
        return Path(args.db)
    projects_folder = load_projects_folder(args.projects_folder)
    return projects_folder / args.project / "cases.db"


def table_exists(cur, table):
    row = cur.execute(
        "SELECT name FROM sqlite_schema WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def table_columns(cur, table):
    cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return [c[1] for c in cols]


def ensure_column(cur, table, col_name, col_type):
    cols = table_columns(cur, table)
    if col_name in cols:
        return False
    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
    return True


def load_ds_h5_map(project_folder):
    gp = project_folder / "global_properties.json"
    if not gp.exists():
        raise FileNotFoundError(
            f"global_properties.json not found at {gp}. Cannot resolve datasource h5 files."
        )

    data = json.loads(gp.read_text())
    datasources = data.get("datasources") or []
    if not datasources:
        raise RuntimeError("No datasources found in global_properties.json")

    ds_to_h5 = {}
    for ds_info in datasources:
        ds_name = ds_info.get("ds")
        if not ds_name:
            continue

        h5 = ds_info.get("h5_fname")
        if h5:
            h5_path = Path(h5)
        else:
            folder = ds_info.get("folder")
            if not folder:
                continue
            h5_path = Path(folder) / "fg_voxels.h5"

        ds_to_h5[ds_name] = h5_path

    if not ds_to_h5:
        raise RuntimeError("Could not build ds->h5 mapping from global_properties.json")

    return ds_to_h5


def labels_from_h5(h5_file, case_id):
    if case_id not in h5_file:
        raise KeyError(f"Case '{case_id}' not found in H5: {h5_file.filename}")

    case = h5_file[case_id]
    if "labels" not in case.attrs:
        raise KeyError(
            f"Case '{case_id}' has no 'labels' attr in H5: {h5_file.filename}"
        )

    raw_labels = list(case.attrs["labels"])
    labels = [int(x) for x in raw_labels]

    # Mirrors project.py logic: nnz is True when no foreground labels were found.
    nnz = int(len(labels) == 0)
    return str(labels), nnz


def main():
    args = parse_args()
    db = resolve_db_path(args)
    if not db.exists():
        raise FileNotFoundError(f"cases.db not found: {db}")

    project_folder = db.parent
    ds_to_h5 = load_ds_h5_map(project_folder)

    con = sqlite3.connect(str(db))
    h5_handles = {}

    try:
        cur = con.cursor()
        if not table_exists(cur, "datasources"):
            raise RuntimeError("Table 'datasources' not found in DB.")

        before_cols = table_columns(cur, "datasources")
        added_labels = False
        added_nnz = False

        if not args.dry_run:
            added_labels = ensure_column(cur, "datasources", "labels", "TEXT")
            added_nnz = ensure_column(cur, "datasources", "nnz", "INTEGER")

        rows = cur.execute("SELECT rowid, ds, case_id FROM datasources").fetchall()

        updated = 0
        unresolved_ds = 0
        missing_h5 = 0
        missing_case = 0
        missing_labels_attr = 0

        for rowid, ds, case_id in rows:
            h5_path = ds_to_h5.get(ds)
            if h5_path is None:
                unresolved_ds += 1
                continue

            if not h5_path.exists():
                missing_h5 += 1
                continue

            if ds not in h5_handles:
                h5_handles[ds] = h5py.File(h5_path, "r")

            try:
                labels, nnz = labels_from_h5(h5_handles[ds], case_id)
            except KeyError as e:
                msg = str(e)
                if "not found in H5" in msg:
                    missing_case += 1
                else:
                    missing_labels_attr += 1
                continue

            if not args.dry_run:
                cur.execute(
                    "UPDATE datasources SET labels = ?, nnz = ? WHERE rowid = ?",
                    (labels, nnz, rowid),
                )
            updated += 1

        if not args.dry_run:
            con.commit()

        after_cols = table_columns(cur, "datasources")

        print(f"DB: {db}")
        print(f"Rows in datasources: {len(rows)}")
        print(f"Columns before: {before_cols}")
        print(f"Columns after:  {after_cols}")
        print(f"Added labels column: {added_labels}")
        print(f"Added nnz column: {added_nnz}")
        print(f"Rows updated from H5: {updated}")
        print(f"Unresolved datasource mapping: {unresolved_ds}")
        print(f"Missing H5 files: {missing_h5}")
        print(f"Missing case_id in H5: {missing_case}")
        print(f"Missing labels attr in H5: {missing_labels_attr}")
        if args.dry_run:
            print("Dry-run mode: no database writes applied.")

    finally:
        for h5f in h5_handles.values():
            h5f.close()
        con.close()


if __name__ == "__main__":
    main()
