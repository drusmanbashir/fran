    # #python  analyze_resample.py -t nodes -p 6 -n 4 -o


if __name__ == '__main__':
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

    project_title = P.project_title

    Tm = RayTrainer(project_title, conf, None)

    lr = conf["model_params"]["lr"]
    bs = 4
    devices = 1
    headline(f"Training with conf: {conf}")
    lr = conf["model_params"]["lr"]
    conf["dataset_params"]["src_dims"] = make_patch_size(
        conf["dataset_params"]["src_dim0"], conf["dataset_params"]["src_dim1"]
    )
    conf["plan_train"]["patch_size"] = make_patch_size(
        conf["plan_train"]["patch_dim0"], conf["plan_train"]["patch_dim1"]
    )
    headline(conf["dataset_params"]["src_dims"])
    headline(conf["plan_train"]["patch_size"])

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
        epochs=num_epochs,
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
    conf["dataset_params"]["src_dim1"]
    conf2 = conf.copy()
    conf2["dataset_params"]["src_dim0"] = conf["dataset_params"]["src_dim0"].sample()
    conf2["dataset_params"]["src_dim1"] = conf["dataset_params"]["src_dim1"].sample()
    conf2["plan_train"]["patch_dim0"] = conf["plan_train"]["patch_dim0"].sample()
    conf2["plan_train"]["patch_dim1"] = conf["plan_train"]["patch_dim1"].sample()
    conf2["plan_train"]["expand_by"] = conf["plan_train"]["expand_by"].sample()
# %%
    conf2["plan_train"]["patch_size"] = make_patch_size(
        conf2["plan_train"]["patch_dim0"], conf2["plan_train"]["patch_dim1"]
    )
    print(conf2["plan_train"]["patch_size"])
    conf2["dataset_params"]["src_dims"] = make_patch_size(
        conf2["dataset_params"]["src_dim0"], conf2["dataset_params"]["src_dim1"]
    )
# %%
    print(conf2["dataset_params"]["src_dims"])
# %%

    plan = conf2["plan_train"]
    plan["expand_by"]
    conf2["plan_valid"]
    conf2["plan_train"]["patch_size"]
    statuses = confirm_plan_analyzed(P, plan)

# %%
