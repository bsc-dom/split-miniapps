from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import mstats
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

    # This is only for 
    compss_data = data.query("dataclay == 0 & dask != 1")
    if not compss_data.empty:
        sns.violinplot(data=compss_data, x="nodes", hue="mode", y=y_col, 
                       scale='width',
                       palette="Set2", inner="quartile", bw=VIOLIN_BW)

#     ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
#     ax2.get_legend().remove()
    plt.suptitle("COMPSs executions")
    plt.show()

    if "copy_fit_struct" in data or "use_active" in data:
        filter_by = "copy_fit_struct" if "copy_fit_struct" in data else "use_active"
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,10))

        plotdata = data.query("(dataclay == 1) and (%s == 0)" % filter_by)
        sns.barplot(data=plotdata, x="nodes", hue="mode", y=y_col,
                    estimator=estimator, ax=ax1, palette="Set2")
        sns.violinplot(data=plotdata, x="nodes", hue="mode", y=y_col, 
                       scale='width',
                       split=True, ax=ax2, inner="quartile", bw=VIOLIN_BW, palette="Set2")
    
        ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax2.get_legend().remove()
        plt.suptitle("dataClay executions (no %s)" % filter_by)
        plt.show()

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,10))

        plotdata = data.query("(dataclay == 1) and (%s == 1)" % filter_by)
        sns.barplot(data=plotdata, x="nodes", hue="mode", y=y_col,
                    estimator=estimator, ax=ax1, palette="Set2")
        sns.violinplot(data=plotdata, x="nodes", hue="mode", y=y_col, 
                       scale='width',
                       split=True, ax=ax2, inner="quartile", bw=VIOLIN_BW, palette="Set2")
    
        ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax2.get_legend().remove()
        plt.suptitle("dataClay executions (%s)" % filter_by)
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


def prepare_df(db, list_of_jobid=[], maintain_compss=False):
    df = db.to_dataframe().query("tracing != 1")

    df.loc[(df['dataclay'] == 0) & (df['dask'] != 1), 'mode'] = 'COMPSs'
    df.loc[(df['dask'] == 1) & (df['use_split'] == 1), 'mode'] = 'zdask+split'
    df.loc[(df['dask'] == 1) & (df['dask_rechunk'] > 0), 'mode'] = 'zdask+rechunk'
    df.loc[(df['dask'] == 1) & (df['use_split'] != 1) & ~(df['dask_rechunk'] > 0), 'mode'] = 'zdask'
    df.loc[(df['dataclay'] == 1) & (df['use_split'] == 1), 'mode'] = 'dataClay+split'
    df.loc[(df['dataclay'] == 1) & (df['use_split'] == 0), 'mode'] = 'dataClay'

    if "copy_fit_struct" in df:
        # Set copy_fit_struct to 1 when not defined (old executions)
        df.loc[(df['dataclay'] == 1) & (df['copy_fit_struct'] != 0), 'copy_fit_struct'] = 1

        # We no longer want to analyze the copy_fit_struct options, as we have learned
        # that it always makes sense to have it active
        df.drop(df[df["copy_fit_struct"] != 1].index, inplace=True)
        del df["copy_fit_struct"]

    #     df.loc[(df['dataclay'] == 1) & (df['copy_fit_struct'] == 0), "mode"] += "[no copy]"
    #     df.loc[(df['dask'] == 1) & (df['copy_fit_struct'] == 0), "mode"] += "[no copy]"

    if "use_active" in df:
        df.loc[(df['dataclay'] == 1) & (df['use_active'] == 1), "mode"] += "[active]"
        df.loc[(df['dataclay'] == 1) & (df['use_active'] == 0), "mode"] += "[non-active]"

    if not maintain_compss:
        df.drop(df[df["mode"] == "COMPSs"].index, inplace=True)

    return df[df.id.isin(list_of_jobid) == False]

def winsorize_edf(edf, value_to_winsorize, grouping_cols):
    # Prepare the intermediate structure that we will cartesian
    cartesian = list()
    for gcol in grouping_cols:
        cartesian.append(edf[gcol].unique())

    for unique_config in product(*cartesian):
        # print("Doing %s" % (unique_config,))
        working_df = edf[value_to_winsorize]
        mask = working_df.notnull()
        for gcol, value in zip(grouping_cols, unique_config):
            # print("Filtering %s with value %s" % (gcol, value))
            mask &= edf[gcol] == value

        # Less than five values -> skip
        if mask.sum() < 5:
            continue

        working_df[mask] = mstats.winsorize(working_df[mask], limits=[0, 0.1])
