#!/usr/bin/env python3
import os
import sys
from shutil import which
from pathlib import Path


PYTHON_EXE = "/home/ub/mambaforge/envs/dl/bin/python"


def main(args):
    target_script = args.target_script
    script_args = args.script_args

    target_cmd = [PYTHON_EXE, target_script, *script_args]
    if args.allow_suspend:
        os.execvp(target_cmd[0], target_cmd)

    systemd_inhibit_cmd = [
        "systemd-inhibit",
        "--what=sleep",
        "--why=Running long fran job",
        *target_cmd,
    ]

    # GNOME's idle suspend path may bypass the expected sleep blockers.
    # When in a graphical GNOME session, add gnome-session-inhibit too.
    should_try_gnome_inhibit = bool(
        os.environ.get("DISPLAY")
        and os.environ.get("DBUS_SESSION_BUS_ADDRESS")
        and which("gnome-session-inhibit")
    )
    if should_try_gnome_inhibit:
        gnome_inhibit_cmd = [
            "gnome-session-inhibit",
            "--inhibit",
            "suspend:idle",
            "--reason",
            "Running long fran job",
            *systemd_inhibit_cmd,
        ]
        try:
            print(
                "[block_suspend] using gnome-session-inhibit + systemd-inhibit",
                flush=True,
            )
            os.execvp(gnome_inhibit_cmd[0], gnome_inhibit_cmd)
        except OSError as exc:
            print(
                f"[block_suspend] gnome-session-inhibit failed ({exc}); "
                "falling back to systemd-inhibit",
                file=sys.stderr,
                flush=True,
            )

    print("[block_suspend] using systemd-inhibit", flush=True)
    os.execvp(systemd_inhibit_cmd[0], systemd_inhibit_cmd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch a Python script with system suspend blocked by default."
    )
    parser.add_argument(
        "--allow-suspend",
        action="store_true",
        help="Run target script without systemd-inhibit.",
    )
    parser.add_argument(
        "target_script",
        type=str,
        help="Path to target Python script.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the target script.",
    )
    args = parser.parse_args()

    if args.script_args and args.script_args[0] == "--":
        args.script_args = args.script_args[1:]

    if not Path(args.target_script).exists():
        raise FileNotFoundError(f"Target script not found: {args.target_script}")

    main(args)
