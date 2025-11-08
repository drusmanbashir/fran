# %%
import warnings
from fran.managers.data import source_collated
from fran.trainers.impsamp import DeepSupervisionLoss, Trainer, pool_op_kernels_nnunet
import ipdb

tr = ipdb.set_trace

import torch

from fran.extra.deepcore.deepcore.met.earlytrain import EarlyTrain
from fran.extra.deepcore.deepcore.met.met_utils import submodular_optimizer
from fran.extra.deepcore.deepcore.met.met_utils.euclidean import euclidean_dist_pair_np
from fran.extra.deepcore.deepcore.met.met_utils.submodular_function import (
    FacilityLocation,
)
from fran.extra.deepcore.deepcore.nets.nets_utils.parallel import MyDataParallel
import numpy as np


class CraigSeg(EarlyTrain):
    def __init__(
        self,
        dst_train,
        dataset_params,
        model_params,
        args,
        fraction=0.5,
        random_seed=None,
        epochs=200,
        specific_model=None,
        balance=True,
        greedy="LazyGreedy",
        **kwargs,
    ):
        super().__init__(
            dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs
        )

        if greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        self.dataset_params = dataset_params
        self.model_params = model_params
        self._greedy = greedy

        self.balance = balance
        self.loss_fnc = self.create_loss_fnc()

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def train(self, epoch, list_of_train_idx):
        pass

    def _common_step(self, batch, batch_idx):
        if not hasattr(self, "batch_size"):
            self.batch_size = batch["image"].shape[0]
        inputs, target = batch["image"], batch["lm"]
        pred = self.model.forward(
            inputs.to("cuda")
        )  # self.pred so that NeptuneImageGridCallback can use it

        loss = self.loss_fnc(pred, target.to("cuda"))
        loss_dict = self.loss_fnc.loss_dict
        return loss, pred

    def create_loss_fnc(self):
        num_pool = 5
        self.net_num_pool_op_kernel_sizes = pool_op_kernels_nnunet(
            self.dataset_params["patch_size"]
        )
        self.deep_supervision_scales = [[1, 1, 1]] + list(
            list(i)
            for i in 1
            / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)
        )[:-1]
        loss_func = DeepSupervisionLoss(
            levels=num_pool,
            deep_supervision_scales=self.deep_supervision_scales,
            fg_classes=self.model_params["out_channels"] - 1,
        )
        return loss_func

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print(
                "| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f"
                % (
                    epoch,
                    self.epochs,
                    batch_idx + 1,
                    (self.n_pretrain_size // batch_size) + 1,
                    loss.item(),
                )
            )

    def calc_gradient(self, index=None):
        self.model.eval()
        batch_loader = torch.utils.data.DataLoader(
            (
                self.dst_train
                if index is None
                else torch.utils.data.Subset(self.dst_train, index)
            ),
            batch_size=self.args.selection_batch,
            num_workers=self.args.workers,
        )
        # sample_num = len(self.dst_val.targets) if index is None else len(index)
        self.embedding_dim = self.model.get_last_layer().in_features

        gradients = []

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(
                outputs.requires_grad_(True), targets.to(self.args.device)
            ).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = self.model.embedding_recorder.embedding.view(
                    batch_num, 1, self.embedding_dim
                ).repeat(1, self.args.num_classes, 1) * bias_parameters_grads.view(
                    batch_num, self.args.num_classes, 1
                ).repeat(
                    1, 1, self.embedding_dim
                )
                gradients.append(
                    torch.cat(
                        [bias_parameters_grads, weight_parameters_grads.flatten(1)],
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )

        gradients = np.concatenate(gradients, axis=0)

        self.model.train()
        return euclidean_dist_pair_np(gradients)

    def calc_weights(self, matrix, result):
        min_sample = np.argmax(matrix[result], axis=0)
        weights = np.ones(np.sum(result) if result.dtype == bool else len(result))
        for i in min_sample:
            weights[i] = weights[i] + 1
        return weights

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

        self.model.no_grad = True
        with self.model.embedding_recorder:
            tr()
            if self.balance:
                # Do selection by class
                selection_result = np.array([], dtype=np.int32)
                weights = np.array([])
                for c in range(self.args.num_classes):
                    class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                    matrix = -1.0 * self.calc_gradient(class_index)
                    matrix -= np.min(matrix) - 1e-3
                    submod_function = FacilityLocation(
                        index=class_index, similarity_matrix=matrix
                    )
                    submod_optimizer = submodular_optimizer.__dict__[self._greedy](
                        args=self.args,
                        index=class_index,
                        budget=round(self.fraction * len(class_index)),
                    )
                    class_result = submod_optimizer.select(
                        gain_function=submod_function.calc_gain,
                        update_state=submod_function.update_state,
                    )
                    selection_result = np.append(selection_result, class_result)
                    weights = np.append(
                        weights,
                        self.calc_weights(matrix, np.isin(class_index, class_result)),
                    )
            else:
                matrix = np.zeros([self.n_train, self.n_train])
                all_index = np.arange(self.n_train)
                for c in range(self.args.num_classes):  # Sparse Matrix
                    class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                    matrix[np.ix_(class_index, class_index)] = (
                        -1.0 * self.calc_gradient(class_index)
                    )
                    matrix[np.ix_(class_index, class_index)] -= (
                        np.min(matrix[np.ix_(class_index, class_index)]) - 1e-3
                    )
                submod_function = FacilityLocation(
                    index=all_index, similarity_matrix=matrix
                )
                submod_optimizer = submodular_optimizer.__dict__[self._greedy](
                    args=self.args, index=all_index, budget=self.coreset_size
                )
                selection_result = submod_optimizer.select(
                    gain_function=submod_function.calc_gain_batch,
                    update_state=submod_function.update_state,
                    batch=self.args.selection_batch,
                )
                weights = self.calc_weights(matrix, selection_result)
        self.model.no_grad = False
        return {"indices": selection_result, "weights": weights}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result


# %%
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR>

if __name__ == "__main__":
    import torch

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")
    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

    project_title = "litsmc"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_config_wholeimage.xlsx"
    )
    configuration_filename = "/s/fran_storage/projects/litsmc/experiment_config.xlsx"
    configuration_filename = None

    conf = ConfigMaker(proj, ).config
    conf["dataset_params"]["batch_size"] = 2

    # conf['model_params']['lr']=1e-3

    # conf['dataset_params']['plan']=5
# %%
    from fran.extra.deepcore.deepcore.met.args import args

    device_id = 1
    # run_name = "LITS-1007"
    # device_id = 0
    run_totalseg = "LITS-1025"
    run_emp = None
    run_litsmc = "LITS-1018"
    bs = 2  # 5 is good if LBD with 2 samples per case
    # run_name ='LITS-1003'
    compiled = False
    profiler = False
    batch_finder = False
    neptune = False
    tags = []
    cbs = []
    description = f""
# %%
# SECTION:-------------------- IMPORTANCE SAMPLING-------------------------------------------------------------------------------------- <CR> <CR> <CR>
# %%
    Tm = Trainer(proj, conf, run_litsmc)
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=50 if profiler == False else 1,
        batchsize_finder=batch_finder,
        profiler=profiler,
        cbs=cbs,
        neptune=neptune,
        tags=tags,
        description=description,
    )
    Tm.N.model
    Tm.D.setup()
    print(Tm.N.model.embedding_recorder.record_embedding)
# %%
    args.batch = 2
    args.selection_batch = 2
    C = CraigSeg(
        dst_train=Tm.D.train_ds,
        specific_model=Tm.N.model,
        dataset_params=conf["dataset_params"],
        model_params=conf["model_params"],
        args=args,
        random_seed=42,
    )
    C.run()

# %%
# SECTION:-------------------- CALC_GRAD-------------------------------------------------------------------------------------- <CR>

    index = None
    C.model.eval()

# %%
    batch_loader = torch.utils.data.DataLoader(
        C.dst_train,
        batch_size=C.args.selection_batch,
        num_workers=C.args.workers,
        collate_fn=source_collated,
    )
# %%
    # sample_num = len(C.dst_val.targets) if index is None else len(index)
    with C.model.embedding_recorder:
        C.embedding_dim = C.model.get_last_layer().in_channels
        gradients = []
        batch = next(iter(batch_loader))
        # for i, batch in enumerate(batch_loader):
        C.model_optimizer.zero_grad()
        # input = batch['image']
        # targets = batch['lm']
        # outputs = C.model(input.to(C.args.device))[0] # only the main
        loss, outputs = C._common_step(batch, 0)
        outputs = outputs[0]

        C.model.embedding_recorder.record_embedding
        C.specific_model.embedding_recorder.record_embedding = True
        targets = batch["lm"]
        batch_num = targets.shape[0]
        with torch.no_grad():
            tr()
            bias_parameters_grads = torch.autograd.grad(loss, C.model.embedding_recorder.embedding)
            C.model.embedding_recorder.embedding.numel()
            weight_parameters_grads = C.model.embedding_recorder.embedding.view(
                batch_num, 1, C.embedding_dim
            ).repeat(1, C.args.num_classes, 1) * bias_parameters_grads.view(
                batch_num, C.args.num_classes, 1
            ).repeat(
                1, 1, C.embedding_dim
            )
            gradients.append(
                torch.cat(
                    [bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1
                )
                .cpu()
                .numpy()
            )

        gradients = np.concatenate(gradients, axis=0)

        # self.model.train()

# %%
# %%
