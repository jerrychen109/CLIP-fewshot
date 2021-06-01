import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_run_accuracies(accuracies, xtick_interval=10):
    """ Plots the results of a fewshot classification experiment across different values of k
    and multiple trials per value of k.

    Input:
    - accuracies: a list (k, trial_accuracies) tuples where trial_accuracies is a list of accuracies
                  across all trials for k.
    """
    df = pd.DataFrame.from_records(np.array(accuracies), columns=['k', 'acc'])
    df = df.explode('acc')
    df['acc'] = df['acc'].astype(float)
    ax = sns.barplot(x='k', y='acc', data=df)
    ax.yaxis.set_major_locator(plt.MaxNLocator(20))
    ax.set(ylim=(0, 1))
    xticks = ax.xaxis.get_major_ticks()
    for i in range(len(xticks)):
        if i % xtick_interval != xtick_interval - 1:
            xticks[i].set_visible(False)
    plt.show()
