from pathlib import Path
from typing import Optional

from fran.managers.data.main import DataManagerDual
from utilz.cprint import cprint


class DataManagerRT(DataManagerDual):
    def __init__(
        self,
        project_title,
        configs: dict,
        batch_size: int,
        manager_class,
        cache_rate=0.0,
        device="cuda",
        ds_type=None,
        data_folder: Optional[str | Path] = None,
        train_indices=None,
        debug=False,
        batch_tfms: bool = False,
    ):
        self.manager_class = manager_class
        super().__init__(
            project_title=project_title,
            configs=configs,
            batch_size=batch_size,
            cache_rate=cache_rate,
            device=device,
            ds_type=ds_type,
            save_hyperparameters=True,
            data_folder=data_folder,
            manager_class_train=manager_class,
            manager_class_valid=None,
            train_indices=train_indices,
            val_indices=None,
            val_sampling=1.0,
            debug=debug,
            batch_tfms=batch_tfms,
        )

    def _build_managers(self):
        self.train_manager = self.manager_class(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="all",
            device=self.device,
            ds_type=self.ds_type,
            data_folder=self.data_folder,
            debug=self.debug,
        )

    def _iter_managers(self):
        return (self.train_manager,)

    def prepare_data(self):
        self._build_managers()
        self._call_prepare_data()
        if self.train_indices is not None:
            cprint(
                f"Limiting training dataset size to{self.train_indices}", color="yellow"
            )
            self.train_manager.select_cases_from_inds(self.train_indices)
            self.train_manager.data = self.train_manager.create_staged_data_dicts(
                self.train_manager.cases
            )

    def val_dataloader(self):
        self._validation_crash()

    def state_dict(self) -> dict:
        return {
            "batch_size": int(self._batch_size),
            "train_indices": self.train_indices,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not state_dict:
            return
        if "batch_size" in state_dict:
            self._batch_size = int(state_dict["batch_size"])
        if "train_indices" in state_dict:
            self.train_indices = state_dict["train_indices"]

    @property
    def valid_ds(self):
        self._validation_crash()

    @property
    def valid_manager(self):
        self._validation_crash()

    @DataManagerDual.batch_size.setter
    def batch_size(self, v: int) -> None:
        v = int(v)
        if v == self._batch_size:
            return
        self._batch_size = v
        if hasattr(self, "train_manager"):
            self.train_manager.batch_size = v
            self.train_manager.set_effective_batch_size()
            self.train_manager.create_train_dataloader()

    def _validation_crash(self):
        raise RuntimeError("DataManagerRT has no validation path")
