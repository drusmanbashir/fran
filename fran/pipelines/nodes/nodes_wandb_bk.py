import os

from fran.configs.parser import ConfigMaker
from fran.managers import Project
from fran.trainers.trainer_bk import TrainerBK


if __name__ == "__main__":
    P = Project("nodes")
    C = ConfigMaker(P)
    C.setup(6)
    conf = C.configs

    devices = [0]
    bs = 2
    compiled = False
    profiler = False
    batch_finder = False
    use_wandb = True
    override_dm = False
    tags = ["wandb-bk"]
    description = "nodes wandb smoke up to fit"
    run_name = os.environ.get("RUN_ID") or None
    target_epochs = int(os.environ.get("EPOCHS", "3"))
    lr = 1e-2

    conf["dataset_params"]["cache_rate"] = 0.0
    conf["dataset_params"]["fold"] = 0

    Tm = TrainerBK(P.project_title, conf, run_name)
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=devices,
        cbs=[],
        epochs=target_epochs if not profiler else 1,
        batchsize_finder=batch_finder,
        profiler=profiler,
        wandb=use_wandb,
        wandb_grid_epoch_freq=1,
        tags=tags,
        description=description,
        lr=lr,
        override_dm_checkpoint=override_dm,
    )
    Tm.N.compiled = compiled
    Tm.fit()
