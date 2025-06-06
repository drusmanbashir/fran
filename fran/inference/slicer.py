from contextlib import contextmanager

from torch.profiler import ProfilerActivity, profile

from fran.inference.cascade import CascadeInferer


class SlicerCascadeInferer(CascadeInferer):
    def __init__(
        self,
        run_name_w,
        runs_p,
        localiser_labels: list[str],  # these labels will be used to create bbox
        devices=[0],
        overwrite=True,
        safe_mode=False,
        profile=None,
        save_channels=False,
        save=True,
        save_localiser=True,
        k_largest=None,  # assign a number if there are organs involved
        **kwargs
    ):
        super().__init__(
            run_name_w=run_name_w,
            runs_p=runs_p,
            localiser_labels=localiser_labels,
            devices=devices,
            overwrite=overwrite,
            safe_mode=safe_mode,
            profile=profile,
            save_channels=save_channels,
            save_localiser=save_localiser,
            save=save,
            k_largest=k_largest,
            **kwargs
        )
        self.profile_enabled = profile

    def profile_decorator(self, func):
        @contextmanager()
        def profiler(*args, **kwargs):
            if self.profile_enabled:
                with profile(
                    activities=[ProfilerActivity.CPU], record_shapes=True
                ) as prof:
                    yield
                    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=2))
                    # prof.export_stacks("/home/ub/.tmp/profiler_stacks.txt", "self_cpu_time_total")
                print(
                    prof.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=100
                    ),
                    file=open("/tmp/profile2.txt", "a"),
                )
            else:
                yield

        def wrapper(*args, **kwargs):
            with profiler():
                return func(*args, **kwargs)

        return wrapper

    #
    # def transforms(self):
    #
    #     self.E = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")
    #     self.S = Spacingd(keys=["image"], pixdim=self.dataset_params['spacings'])
    #     self.N = NormaliseClipd(
    #         keys=["image"],
    #         clip_range=self.dataset_params["intensity_clip_range"],
    #         mean=self.dataset_params["mean_fg"],
    #         std=self.dataset_params["std_fg"],
    #     )
    #     self.O = Orientationd(keys=["image"], axcodes="RPS")  # nOTE RPS
    #
    #

    # @profile_decorator
    # def run(self, img):
    #     data = self.load_images(img)
    #     self.bboxes = self.extract_fg_bboxes(data)
    #     data = self.apply_bboxes(data, self.bboxes)
    #     pred_patches = self.patch_prediction(data)
    #     pred_patches = self.decollate_patches(pred_patches, self.bboxes)
    #     output = self.postprocess(pred_patches)
    #     return output
