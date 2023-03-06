import re
import matplotlib.pyplot as plt
import seaborn as sns
from fran.callback.base import *
from fastai.callback.tracker import Callback
from fran.utils.helpers import get_case_id_from_filename
# %%

class CaseIDRecorder(Callback):
    def __init__(self,freq=25, local_folder='/tmp',tracked_var=None,dpi=300):
        '''

        :param freq:
        :param local_folder:
        :param tracked_var:  this variable is invoked at epoch 0 end to set internal var 'track_var'
        :param dpi:
        '''
        self.nep_field = "metrics/case_id_dices"
        self.set_figure_params(dpi)
        store_attr()


    def before_fit(self):
        self.case_fnames = [],[]
        self.losses_dice=[],[]
        self.df_titles = ['valid','train']
        self.dfs ={'valid':None, 'train':None}
    
    def before_batch(self):
        cases_this_batch={}
        for i, fn in enumerate (self.learn.yb[0]):
            cases_this_batch.update({f"case_file{i}":fn})
        self.case_fnames[self.training].append(cases_this_batch)
        self.learn.yb =  [self.learn.xb[1]]
        self.learn.xb =  [self.learn.xb[0]]
    def after_batch(self):
        dict_this_batch={}
        for k , v in self.loss_dict.items():
            if 'dice' in k:
                dict_this_batch.update({k:v.item()})
        self.losses_dice[self.training].append(dict_this_batch)
    def after_epoch(self):
        if self.epoch==0: self.track_var = self.tracked_var
        for i,label in enumerate(self.df_titles):
            df = map(pd.DataFrame,[self.case_fnames[i], self.losses_dice[i]])
            df = pd.concat(df,axis=1)
            self.append_to_running_df(label, df)
        if all([self.epoch>=self.freq , self.epoch%self.freq==0]):
            if not hasattr(self,'rows_per_plot'): self.compute_rows_per_plot()
            self.store_results()

    def set_figure_params(self,dpi):
        self.rcs = [ {'figure.dpi': dpi, 'figure.figsize':(15,10)},  # valid
                 {'figure.dpi': dpi, 'figure.figsize':(25,10)}]     # train

    def append_to_running_df(self,label,df):
        if isinstance(self.dfs[label],pd.DataFrame):
            self.dfs[label] = pd.concat([self.dfs[label],df],axis=0)
        else:
             self.dfs[label]: self.dfs[label] = df

    def store_results(self):
                for label in self.df_titles:
                    storage_string_plot = label+"_plot_epoch{}".format(self.epoch)
                    storage_string_df = label
                    small_df = self.create_limited_df(self.dfs[label])
                    figure = self.create_plot(small_df)
                    if hasattr(self.learn,'nep_run') :
                        self.nep_run[self.nep_field+"_dataframes/{}".format(storage_string_df)].upload(File.as_html(self.dfs[label]))
                        self.nep_run[self.nep_field+"_plots/{}".format(storage_string_plot)].upload(figure)
                    else:
                        fname_df = Path(self.local_folder)/("{}.csv".format(label))
                        fname_plot = fname_df.str_replace(".csv,.jpg")
                        self.dfs[label].to_csv(fname_df, index=False)
    def compute_rows_per_plot(self):
        self.rows_per_plot =[len(self.dfs[label]) for label in self.df_titles]

    def create_limited_df(self,dfd):
        cols = dfd.columns
        dfd = dfd[-self.rows_per_plot[self.training]::]

        x_cols = [c for c in cols if 'case_file' in c]
        y_cols = [c for c in cols if 'loss_dice' in c]

        df2 = pd.melt(dfd,id_vars = y_cols,value_vars = x_cols,value_name = 'case_file' ).drop('variable',axis=1)
        df2 = df2.dropna()
        df3 = df2.assign(case_id = case_id_from_series(df2.case_file))
        df3 = df3.assign(case_file_name= filename_from_series(df3.case_file))
        return df3

                
    def create_plot(self, dfd):
        sns.set(rc=self.rcs[self.training])
        plt.ioff()
        ax= sns.boxplot(x=dfd['case_id'],y=dfd[self.track_var] )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        figure = ax.figure
        figure.tight_layout()
        return figure

    def get_largest_label(self):
        labels = list(self.losses_dice[0][0].keys())
        ranking={}
        pat = r"\D*(\d*)"
        for a in labels:
            mm = re.match(pat,a)
            if len(mm.groups()[0])>0:
                gp = {mm.groups()[0]: mm.group()}
                ranking.update(gp)
        if len(ranking)==1:
            return list(ranking.values())[0]
        else:
            ranks = list(ranking.keys())
            ranks.sort()
            largest = ranks[-1]
            return ranking[largest]

    @property
    def track_var(self):
        return self._track_var

    @track_var.setter
    def track_var(self,value):
        if not value:
            self._track_var = self.get_largest_label()
        else:
            self._track_var = value

# %%


def case_id_from_series(series):
    output = [get_case_id_from_filename(None , Path(y)) for y in series]
    return output

def filename_from_series(series):
    output= [Path(y).name for y in series]
    return output

