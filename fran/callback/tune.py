from fran.callback.base import *
from ray import tune
from pathlib import Path

from utilz.fileio import load_dict, save_dict

class TuneCheckpointCallback(TrackerCallback):
    order=51 # (Recorder callback order +1)
    def __init__(self, monitor='valid_loss', comp=None, min_delta=0.,  fname='model', freq= 6, at_end=True,
                 with_opt=True, reset_on_fit=False):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        store_attr('fname,freq,at_end,with_opt')
        # keep track of file path for loggers
        self.last_saved_path = None
    def after_create(self):
        self.checkpoint_dir = Path(tune.get_trial_dir())
        self.checkpoint_file = self.checkpoint_dir/("checkpoint.json")
        self.learn.model_dir = self.checkpoint_dir/ "model_checkpoints"
        self.fname_prefix = self.learn.model_dir/self.fname
        self._load_latest_checkpoint()
        try:
            self.state = load_dict(self.checkpoint_file)
        except:

            self.state = {'epoch': 0}

    def after_epoch(self):
        self.state['epoch']=self.epoch
        save_dict(self.state,self.checkpoint_file)
        if (self.epoch%self.freq) == 0:
            self._save(f'{self.fname_prefix}')
    def after_fit(self, **kwargs):
        if self.at_end: self._save(f'{self.fname_prefix}')
        # elif not self.every_epoch: self.learn.load_model(f'{self.fname_prefix}', with_opt=self.with_opt)

    def _save(self, name):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def _load_latest_checkpoint(self):
        checkpoints = list(self.learn.model_dir.glob("*"))
        if len(checkpoints) > 0:
            checkpoints.sort(key=os.path.getmtime, reverse=True)
            latest_chckpnt_file = checkpoints[-1].name[:-4]
            print("Loading checkpoint from filename: {}".format(latest_chckpnt_file))
            self.learn.load(latest_chckpnt_file, device='cuda', with_opt=self.with_opt)
        else:
            print("No checkpoints on Tune. Initializing..")

class TuneTrackerCallback(Callback):
    order = TuneCheckpointCallback.order+1
    def __init__(self,freq=2, **kwargs):
        store_attr()
        super().__init__(**kwargs)
    def before_epoch(self):
        self.running_losses_valid = []
    def after_batch(self):
        if not self.training and self.iter%self.freq==0:
            self.running_losses_valid.append(([v.item() for v in self.loss_dict.values()]))
    def after_epoch(self):
        mean_loss =torch.tensor(self.running_losses_valid).mean(0)
        reporting = {key:val for key,val in zip(self.loss_dict.keys(),mean_loss)}
        tune.report(**reporting)



