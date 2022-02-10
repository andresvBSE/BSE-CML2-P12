import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


from itertools import cycle

def missings_plot(df):
    missinvalues = df.isnull().sum(axis=0)/df.shape[0]
    missinvalues.sort_values(inplace=True, ascending=False)

    plt.figure().set_size_inches(12, 6)
    ax = sns.barplot(x=missinvalues.index, y=missinvalues)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90, ha='right')
    ax.axhline(0.2); ax.axhline(0.4); ax.axhline(0.6); ax.axhline(0.8)
    ax.set_title('Plot of missing values (%)')
    ax.set_xlabel('Column')
    ax.set_ylabel('%')
    plt.show()


def target_features_plot(df, target:str, features:list, kind:str):
    plt.figure(figsize=(np.ceil(len(features)/4) * 4,16))
    
    if kind == 'boxplot':
        for i,col in enumerate(features):
            plt.subplot(4,4,i+1)
            sns.boxplot(x=target, y=col, data=df)
    elif kind == 'scatter':      
        for i,col in enumerate(features):
            plt.subplot(4,4,i+1)
            sns.scatterplot(x=target, y=col, data=df)
    elif kind == 'density':      
        for i,col in enumerate(features):
            plt.subplot(4,4,i+1)
            sns.kdeplot(x=col, hue=target, data=df)


def roc_curve_plot(y_true, y_hat):
    roc_vs = roc_curve(y_true, y_hat)
    auc_vs = auc(roc_vs[0], roc_vs[1] )

    plt.figure()
    line_width = 2
    plt.plot(roc_vs[0], roc_vs[1], color='darkorange', lw=line_width,
             label=f"ROC based on training data (AUC = {auc_vs:0.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classification of Defaulters')
    plt.legend(loc='lower right')
    plt.show()

def col_import_plot(columns, coef):
    cols_impact = {'Colname':columns, 'Coef':coef}
    cols_impact_df = pd.DataFrame(cols_impact)
    cols_impact_df['Coef'] = cols_impact_df['Coef'].abs()
    cols_impact_df.sort_values('Coef', ascending=False,inplace=True)

    # Plot for the n vars with more impact
    plt.figure(figsize=(16,6))
    sns.barplot(x='Colname', y='Coef', data=cols_impact_df)
    plt.xticks(rotation=90)

def grid_serach_dt(scores, std_error, alphas, best_params):
    plt.figure().set_size_inches(8, 6)
    plt.semilogx(alphas, scores)

    # plot error lines showing +/- std. errors of the scores
    plt.semilogx(alphas, scores + std_error, 'b--')
    plt.semilogx(alphas, scores - std_error, 'b--')

    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

    plt.ylabel('CV score +/- std error')
    plt.xlabel('C')
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    ymin, ymax = plt.ylim()
    plt.vlines(best_params ,ymin, ymax, linestyle='dashed')
    plt.xlim([alphas[0], alphas[-1]])

# Compute ROC curve and ROC area for each class
def roc_auc_multiclass_plot(y, y_hat, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_hat[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_hat.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    lw = 2

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(9,6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "seagreen", "indigo", "darkgoldenrod", "dimgray"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i+1, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()

