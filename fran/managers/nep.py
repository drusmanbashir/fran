import os
from typing import Optional

import torch._dynamo
from paramiko import SSHClient


torch._dynamo.config.suppress_errors = True
from typing import Any

import neptune as nt
import torch
# from fastcore.basics import GenttAttr
from lightning.pytorch.loggers.neptune import NeptuneLogger

from fran.evaluation.losses import *
from fran.utils.common import *
from utilz.fileio import *
from utilz.helpers import *
from utilz.imageviewers import *
from utilz.helpers import *


from lightning.pytorch.loggers.neptune import NeptuneLogger


from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None


def _to_plain(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    if np is not None and isinstance(x, np.generic):
        return x.item()

    if isinstance(x, Path):
        return str(x)

    if isinstance(x, tuple):
        return [_to_plain(v) for v in x]

    if isinstance(x, list):
        return [_to_plain(v) for v in x]

    if isinstance(x, set):
        return [_to_plain(v) for v in sorted(x, key=lambda z: str(z))]

    if isinstance(x, dict):
        return {str(k): _to_plain(v) for k, v in x.items()}

    return str(x)

def get_neptune_checkpoint(project, run_id):
    nl = NeptuneManager(
        project=project,
        run_id=run_id,  # "LIT-46",
        nep_mode="read-only",
        log_model_checkpoints=False,  # Update to True to log model checkpoints
    )
    ckpt = nl.model_checkpoint
    nl.experiment.stop()
    return ckpt


def download_neptune_checkpoint(project, run_id):
    nl = NeptuneManager(
        project=project,
        run_id=run_id,  # "LIT-46",
        log_model_checkpoints=False,  # Update to True to log model checkpoints
    )
    nl.download_checkpoints()
    ckpt = nl.model_checkpoint
    nl.experiment.stop()
    return ckpt


def get_neptune_project(project, mode):
    """
    Returns project instance based on project title
    """

    project_name, api_token = get_neptune_config()
    return nt.init_project(project=project_name, api_token=api_token, mode=mode)


def get_neptune_config():
    """
    Returns particular project workspace
    """
    commons = load_yaml(common_vars_filename)
    project_name = commons["neptune_project"]
    api_token = commons["neptune_api_token"]
    return project_name, api_token



class NeptuneManager(NeptuneLogger):
    def __init__(
        self,
        *,
        project,
        nep_mode="async",
        run_id: Optional[str] = None,
        log_model_checkpoints: Optional[bool] = False,
        prefix: str = "training",
        **neptune_run_kwargs: Any
    ):
        store_attr("project")
        project_nep, api_token = get_neptune_config()
        os.environ["NEPTUNE_API_TOKEN"] = api_token
        os.environ["NEPTUNE_PROJECT"] = project_nep
        self.df = self.fetch_project_df()
        # if run_id is given cannot give neptune parameters !
        if run_id:
            name = None
            nep_run = self.load_run(run_id, nep_mode)
            project_nep, api_token = None, None
            neptune_run_kwargs = {}
        else:
            name = project.project_title
            nep_run = None

        NeptuneLogger.__init__(
            self,
            api_key=api_token,
            project=project_nep,
            run=nep_run,
            name = name,
            log_model_checkpoints=log_model_checkpoints,
            prefix=prefix,
            **neptune_run_kwargs
        )

    def log_hyperparams(self, params):
        # called by Lightning in _log_hyperparams(self)
        try:
            key = self._construct_path_with_prefix(self.PARAMETERS_KEY)  # usually "training/parameters"
            self.run[key] = _to_plain(params)
        except Exception as e:
            print(f"[Neptune] log_hyperparams skipped: {e}")

    @property
    def nep_run(self):
        return self.experiment

    @property
    def model_checkpoint(self):
        try:
            ckpt = self.experiment["training/best_model_path"].fetch()
            return ckpt
        except:
            print("No checkpoints in this run")

    @model_checkpoint.setter
    def model_checkpoint(self, value):
        self.experiment["training/best_model_path"] = value
        self.experiment.wait()

    def fetch_project_df(self, columns=None):
        print("Downloading runs history as dataframe")
        project_tmp = get_neptune_project(self.project, "read-only")
        df = project_tmp.fetch_runs_table(columns=columns).to_pandas()
        return df
    #
    # def on_fit_start(self):
    #     tr()
    #     headline("Logging to Neptune")
    #     self.experiment["sys/name"] = self.project.project_title
    #     self.experiment.wait()

    def load_run(
        self,
        run_name,
        nep_mode="async",
    ):
        """

        :param run_name:
            If a legit name is passed it will be loaded.
            If an illegal run-name is passed, throws an exception
            If most_recent is passed, most recent run  is loaded.

        :param update_nep_run_from_config: This is a dictionary which can be uploaded on Neptune to alter the parameters of the existing model and track new parameters
        """
        run_id, msg = self.get_run_id(run_name)
        print("{}. Loading".format(msg))
        nep_run = nt.init_run(
            with_id=run_id,
            mode=nep_mode,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=True,
            capture_hardware_metrics=True,
        )
        return nep_run

    def get_run_id(self, run_id):
        if run_id == "most_recent":
            run_id = self.id_most_recent()
            msg = "Most recent run"
        elif run_id is any(["", None]):
            raise Exception(
                "Illegal run name: {}. No ids exist with this name".format(run_id)
            )

        else:
            self.id_exists(run_id)
            msg = "Run id matching {}".format(run_id)
        return run_id, msg

    def id_exists(self, run_id):
        row = self.df.loc[self.df["sys/id"] == run_id]
        try:
            print("Existing Run found. Run id {}".format(row["sys/id"].item()))
            return row["sys/id"].item()
        except Exception as e:
            print("No run with that name exists .. {}".format(e))

    def id_most_recent(self):
        self.df = self.df.sort_values(by="sys/creation_time", ascending=False)
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            if self._has_checkpoints(row):
                print("Loading most recent run. Run id {}".format(row["sys/id"]))
                return row["sys/id"], row["metadata/run_name"]

    def download_checkpoints(self):
        remote_dir = str(Path(self.model_checkpoint).parent)
        latest_ckpt = self.shadow_remote_ckpts(remote_dir)
        if latest_ckpt:
            self.nep_run["training"]["model"]["best_model_path"] = latest_ckpt
            self.nep_run.wait()

    def shadow_remote_ckpts(self, remote_dir):
        hpc_settings = load_yaml(os.environ["HPC_SETTINGS"])
        local_dir = (
            self.project.checkpoints_parent_folder
            / self.run_id
            / ("checkpoints")
        )
        print("\nSSH to remote folder {}".format(remote_dir))
        client = SSHClient()
        client.load_system_host_keys()

        # client.connect(
        #     "login.hpc.qmul.ac.uk",
        #     username=hpc_settings["username"],
        #     password=hpc_settings["password"],
        # )
        client.connect(
            hpc_settings["host"],
            username=hpc_settings["username"],
            password=hpc_settings["password"],
        )
        ftp_client = client.open_sftp()
        tr()
        try:
            fnames = []
            for f in sorted(
                ftp_client.listdir_attr(remote_dir),
                key=lambda k: k.st_mtime,
                reverse=True,
            ):
                fnames.append(f.filename)
        except FileNotFoundError:
            print(
                "\n------------------------------------------------------------------"
            )
            print(
                "Error:Could not find {}.\nIs this a remote folder and exists?\n".format(
                    remote_dir
                )
            )
            return
        remote_fnames = [os.path.join(remote_dir, f) for f in fnames]
        local_fnames = [os.path.join(local_dir, f) for f in fnames]
        maybe_makedirs(local_dir)
        downloaded_files = []
        for rem, loc in zip(remote_fnames, local_fnames):

            if Path(loc).exists():
                print("Local file {} exists already.".format(loc))
                downloaded_files.append(loc)
            else:
                print("Copying file {0} to local folder {1}".format(rem, local_dir))
                ftp_client.get(rem, loc)
                downloaded_files.append(loc)
        # Get the latest file by modification time from downloaded files
        latest_ckpt = max(downloaded_files, key=lambda f: Path(f).stat().st_mtime)
        return latest_ckpt

    def stop(self):
        self.experiment.stop()

    @property
    def run_id(self):
        return self.experiment["sys/id"].fetch()

    @property
    def save_dir(self) -> Optional[str]:
        sd = self.project.checkpoints_parent_folder
        return str(sd)


# %%
# %%
#SECTION:-------------------- SETTING
if __name__ == '__main__':
    from fran.managers.project import Project
    P = Project(project_title="nodes")
    run_name = "LITS-1416"
    download_neptune_checkpoint(P,run_name)
# %%
    nl = NeptuneManager(
        project=P,
        run_id=run_name,  # "LIT-46",
        log_model_checkpoints=False,  # Update to True to log model checkpoints
    )
    nl.download_checkpoints()
# %%
    ckpt = nl.model_checkpoint
    nl.experiment.stop()

    hpc_settings = load_yaml(os.environ["HPC_SETTINGS"])

    client = SSHClient()
    client.load_system_host_keys()

    client.connect(
            "login.hpc.qmul.ac.uk",
            username=hpc_settings["username"],
            password=hpc_settings["password"],
    )

    remote_dir = str(Path(nl.model_checkpoint).parent)

#p %%
# %%


    # def download_checkpoints(self):
    remote_dir = str(Path(self.model_checkpoint).parent)
    latest_ckpt = self.shadow_remote_ckpts(remote_dir)
    if latest_ckpt:
            self.nep_run["training"]["model"]["best_model_path"] = latest_ckpt
#SECTION:-------------------- TRBOUE--------------------------------------------------------------------------------------

