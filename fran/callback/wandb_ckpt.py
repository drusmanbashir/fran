# fran/callbacks/wandb_ckpt_path.py
from __future__ import annotations
from lightning.pytorch.callbacks import Callback
# fran/storage/hpc_fetch.py

from pathlib import Path
from typing import Optional

import paramiko


def ssh_download_file(
    *,
    host: str,
    username: str,
    remote_path: str,
    local_path: str | Path,
    key_filename: Optional[str] = None,
    port: int = 22,
) -> Path:
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=username, key_filename=key_filename)

    try:
        sftp = client.open_sftp()
        try:
            sftp.get(remote_path, str(local_path))
        finally:
            sftp.close()
    finally:
        client.close()

    return local_path

