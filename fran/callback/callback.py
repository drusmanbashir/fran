
from lightning.pytorch.callbacks import BatchSizeFinder


class FranBatchSizeFinder(BatchSizeFinder):
    """
    Use Lightning's stock batch-size finder, but restore the temporary
    probe checkpoint with weights_only=False so PyTorch 2.6+ can unpickle
    the trusted local temp checkpoint it just wrote.
    """

    def scale_batch_size(self, trainer, pl_module) -> None:
        restore = trainer._checkpoint_connector.restore

        def restore_trusted_temp_checkpoint(checkpoint_path=None, weights_only=None):
            return restore(checkpoint_path, weights_only=False)

        trainer._checkpoint_connector.restore = restore_trusted_temp_checkpoint
        try:
            super().scale_batch_size(trainer, pl_module)
        finally:
            trainer._checkpoint_connector.restore = restore


