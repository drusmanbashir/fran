#
# # %%
#     with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#         with record_function("model_inference"):
#             learn.fit_one_cycle(n_epoch=1,lr_max=1e-2)
# # %%
#             # pred = learn.model(a)
#     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#     prof.export_chrome_trace("trace.json")
#     a,b = learn.dls.train.one_batch()
#     learn = learn.to_fp32()
#     with profile(
#         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         with_stack=True,
#     ) as prof:
#         learn.model(a.to('cuda'))
#     print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
#     prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
#
#     fake_data = False
#     if fake_data == True:
#         valid_len = len(learn.dls.valid)
#         train_len = len(learn.dls.train)
#         a, b = learn.dls.train.one_batch()
#         for dl_len, train in zip([train_len, valid_len], [True, False]):
#             for i in tqdm.tqdm(range(dl_len)):
#                 x = torch.rand(a.shape, device="cuda")
#                 y = torch.randint(0, 1, a.shape, device="cuda")
#                 pred = learn.model(x)
#                 loss = learn.loss_func(pred, y)[0]
#                 if train == True:
#                     loss.backward()

