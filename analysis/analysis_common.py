import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.transforms import Bbox


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

from tad4bj import DataStorage

ESTIMATOR_TO_USE = np.mean

VIOLIN_BW = 0.1


def plot_things(data, y_col, estimator=ESTIMATOR_TO_USE, ylim=None, hue_order=None):
    ax = sns.barplot(data=data, x="nodes", hue="mode", y=y_col, 
                estimator=estimator, hue_order=hue_order)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_title("All")

    if ylim is not None:
        ax.set_ylim(ylim)
    plt.show()

#     fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,10))
    
#     sns.barplot(data=data.query("dataclay == 0"), x="nodes", hue="mode", y="iteration_time", 
#                 palette="Set2", estimator=estimator, ax=ax1)

    sns.violinplot(data=data.query("dataclay == 0"), x="nodes", hue="mode", y=y_col, 
                   scale='width',
                   palette="Set2", inner="quartile", bw=VIOLIN_BW)

#     ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
#     ax2.get_legend().remove()
    plt.suptitle("COMPSs executions")
    plt.show()

    if "copy_fit_struct" in data:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,10))

        plotdata = data.query("(dataclay == 1) and (copy_fit_struct == 0)")
        sns.barplot(data=plotdata, x="nodes", hue="mode", y=y_col,
                    estimator=estimator, ax=ax1, palette="Set2")
        sns.violinplot(data=plotdata, x="nodes", hue="mode", y=y_col, 
                       scale='width',
                       split=True, ax=ax2, inner="quartile", bw=VIOLIN_BW, palette="Set2")
    
        ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax2.get_legend().remove()
        plt.suptitle("dataClay executions (no copy on fit struct)")
        plt.show()

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,10))

        plotdata = data.query("(dataclay == 1) and (copy_fit_struct == 1)")
        sns.barplot(data=plotdata, x="nodes", hue="mode", y=y_col,
                    estimator=estimator, ax=ax1, palette="Set2")
        sns.violinplot(data=plotdata, x="nodes", hue="mode", y=y_col, 
                       scale='width',
                       split=True, ax=ax2, inner="quartile", bw=VIOLIN_BW, palette="Set2")
    
        ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax2.get_legend().remove()
        plt.suptitle("dataClay executions (copy on fit struct")
        plt.show()

    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,10))

        sns.barplot(data=data.query("dataclay == 1"), x="nodes", hue="mode", y=y_col,
                    estimator=estimator, ax=ax1, palette="Set2")
        sns.violinplot(data=data.query("dataclay == 1"), x="nodes", hue="mode", y=y_col, 
                       scale='width',
                       split=True, ax=ax2, inner="quartile", bw=VIOLIN_BW, palette="Set2")
    
        ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax2.get_legend().remove()
        plt.suptitle("dataClay executions")
        plt.show()


def save_tweaks(filename, big=False):
    if big:
        size = (6, 4)
    else:
        size = (8, 4)
    plt.gcf().set_size_inches(size[0], size[1])
    b = plt.gcf().get_tightbbox(plt.gcf().canvas.get_renderer())
    bbox = Bbox.from_bounds(b.x0, b.y0, b.x1-b.x0+0.1, b.y1-b.y0)
    plt.savefig(filename, bbox_inches=bbox)


def prepare_df(db, list_of_jobid):
    df = db.to_dataframe().query("tracing == 0")

    df.loc[df['dataclay'] == 0, 'mode'] = 'COMPSs'
    df.loc[(df['dataclay'] == 1) & (df['use_split'] == 1), 'mode'] = 'dataClay+split'
    df.loc[(df['dataclay'] == 1) & (df['use_split'] == 0), 'mode'] = 'dataClay'

    if "copy_fit_struct" in df:
        # Set copy_fit_struct to 1 when not defined (old executions)
        df.loc[(df['dataclay'] == 1) & (df['copy_fit_struct'] != 0), 'copy_fit_struct'] = 1

        df.loc[(df['dataclay'] == 1) & (df['copy_fit_struct'] == 0), "mode"] += "[no copy]"


    return df[df.id.isin(list_of_jobid) == False]
