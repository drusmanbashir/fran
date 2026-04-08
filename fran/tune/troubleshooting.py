# #python  analyze_resample.py -t nodes -p 6 -n 4 -o


if __name__ == "__main__":
    # %%
    #     conf["dataset_params"]["src_dims"] = make_patch_size(conf["dataset_params"]["src_dim0"], conf["dataset_params"]["src_dim1"])
    #     conf["dataset_params"]["src_dims"]
    #     conf["plan_train"]["patch_size"]= make_patch_size(conf["plan_train"]["patch_dim0"], conf["plan_train"]["patch_dim1"])
    #     conf["plan_train"]
    #
    # %%
    #     patch_dim0 = conf["dataset_params"]["src_dim0"]
    #     patch_dim1 = conf["dataset_params"]["src_dim1"]
    #
    #     patch_size = [
    #         patch_dim0,
    #     ] + [
    #         patch_dim1,
    #     ] * 2
    # %%
    # conf["dataset_params"]["src_dims"]
    # %%

    project_title = P.project_title  # noqa: F821

    Tm = RayTrainer(project_title, conf, None)  # noqa: F821

    lr = conf["model_params"]["lr"]  # noqa: F821
    bs = 4
    devices = 1
    headline(f"Training with conf: {conf}")  # noqa: F821
    lr = conf["model_params"]["lr"]  # noqa: F821
    conf["dataset_params"]["src_dims"] = make_patch_size(  # noqa: F821
        conf["dataset_params"]["src_dim0"], conf["dataset_params"]["src_dim1"]  # noqa: F821
    )
    conf["plan_train"]["patch_size"] = make_patch_size(  # noqa: F821
        conf["plan_train"]["patch_dim0"], conf["plan_train"]["patch_dim1"]  # noqa: F821
    )
    headline(conf["dataset_params"]["src_dims"])  # noqa: F821
    headline(conf["plan_train"]["patch_size"])  # noqa: F821

    compiled = False
    neptune = False
    tags = None
    description = ""
    override_dm = False
    # %%

    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=devices,
        epochs=num_epochs,  # noqa: F821
        batchsize_finder=False,
        profiler=False,
        neptune=neptune,
        tags=tags,
        description=description,
        lr=lr,
        override_dm_checkpoint=override_dm,
    )
    # %%
    # %%
    # SECTION:-------------------- TS-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
    conf["dataset_params"]["src_dim1"]  # noqa: F821
    conf2 = conf.copy()  # noqa: F821
    conf2["dataset_params"]["src_dim0"] = conf["dataset_params"]["src_dim0"].sample()  # noqa: F821
    conf2["dataset_params"]["src_dim1"] = conf["dataset_params"]["src_dim1"].sample()  # noqa: F821
    conf2["plan_train"]["patch_dim0"] = conf["plan_train"]["patch_dim0"].sample()  # noqa: F821
    conf2["plan_train"]["patch_dim1"] = conf["plan_train"]["patch_dim1"].sample()  # noqa: F821
    conf2["plan_train"]["expand_by"] = conf["plan_train"]["expand_by"].sample()  # noqa: F821
    # %%
    conf2["plan_train"]["patch_size"] = make_patch_size(  # noqa: F821
        conf2["plan_train"]["patch_dim0"], conf2["plan_train"]["patch_dim1"]
    )
    print(conf2["plan_train"]["patch_size"])
    conf2["dataset_params"]["src_dims"] = make_patch_size(  # noqa: F821
        conf2["dataset_params"]["src_dim0"], conf2["dataset_params"]["src_dim1"]
    )
    # %%
    print(conf2["dataset_params"]["src_dims"])
    # %%

    plan = conf2["plan_train"]
    plan["expand_by"]
    conf2["plan_valid"]
    conf2["plan_train"]["patch_size"]
    statuses = confirm_plan_analyzed(P, plan)  # noqa: F821

# %%
