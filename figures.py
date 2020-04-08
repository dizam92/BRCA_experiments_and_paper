import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import h5py
import scipy
from experiments.utilities import autolabel, load_data
from sklearn.model_selection import train_test_split

def barplot_figures(file_name, fig_title, saving_repo, train_metrics, test_metrics, std_train_metrics, std_test_metrics,
                    metrics_list=['Acc', 'F1', 'Prec', 'Rec']):
    """
    Args:

    Return:
        Fig, png with the train and test per experiment next to next
    """
    # Build the plot
    nbResults = len(train_metrics)
    f, ax = plt.subplots(nrows=1, ncols=1)
    barWidth = 0.35
    ax.set_title(f"{fig_title}")
    rects = ax.bar(range(nbResults), test_metrics, barWidth, color="r", yerr=std_test_metrics)
    rect2 = ax.bar(np.arange(nbResults) + barWidth, train_metrics, barWidth, color="0.7", yerr=std_train_metrics)
    autolabel(rects, ax)
    autolabel(rect2, ax)
    ax.legend((rects[0], rect2[0]), ('Test', 'Train'), loc='upper right', ncol=2, mode="expand", borderaxespad=0.)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xticks(np.arange(nbResults) + barWidth)
    ax.set_xticklabels(metrics_list)
    plt.tight_layout()
    f.savefig(f"{saving_repo}/{file_name}.png")
    plt.show()
    
def stars(p):
    """ Small function find on the internet to plot the importance of the p value """
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "-"
    
def boxplot_figures(data, return_views, saving_repo, features_cibles):
    """ 
    Args:  
        data, str, datapath
        return_views, str, the view
        features_cibles, list of features to built the boxplot
    Returns:
        figures
    """
    x, y, features_names, patients_names = load_data(data=data, return_views=return_views)
    index_patients_positifs = np.where(y == 1)[0]
    index_patients_negatifs = np.where(y == -1)[0]
    x_patients_positifs = x[index_patients_positifs]
    x_patients_negatifs = x[index_patients_negatifs]
    features_cibles_index = [(i, el) for feat_cibl in features_cibles for i, el in enumerate(features_names) if feat_cibl == el]
    for feat in features_cibles_index:  # feat est un tuple avec feat[0] contenant l'index et feat[1] contenant le nom
        data_positif = x_patients_positifs[:, feat[0]]
        data_negatif = x_patients_negatifs[:, feat[0]]

        z, p = scipy.stats.mannwhitneyu(data_positif, data_negatif)
        p_value = p * 2
        y_max = np.max(np.concatenate((data_positif, data_negatif)))
        y_min = np.min(np.concatenate((data_positif, data_negatif)))

        data = [data_positif, data_negatif]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(data, notch=0, sym='b+', vert=1, whis=1.5, positions=None, widths=0.6)

        ax.annotate("", xy=(1, y_max), xycoords='data',
                    xytext=(2, y_max), textcoords='data',
                    arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                                    connectionstyle="bar,fraction=0.2"))

        ax.text(1.5, y_max + abs(y_max - y_min) * 0.1, stars(p_value),
                horizontalalignment='center',
                verticalalignment='center')

        plt.xlabel(f'Positives                                               Negatives')
        plt.ylabel(f'{feat[1]}')
        # plt.boxplot(data=data)
        feat_name = feat[1].replace('.', '')
        feat_name = feat_name.replace('_', '')
        plt.tight_layout()
        fig.savefig(f"{saving_repo}/{feat_name}_BoxPlots.pdf")
        plt.close()

    
    
if __name__ == "__main__":
    boxplot_figures(data='/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5',
                    return_views='methyl_rna_iso_mirna', 
                    saving_repo='/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper', 
                    features_cibles=['uc002vwt.2_MLPH', 'hsa-mir-218-2'])
    exit()
    barplot_figures(file_name='DT_metrics_bar_plots', 
                    fig_title='metrics', 
                    saving_repo='./', 
                    train_metrics=[0.9877, 0.9636, 0.9431, 0.986], 
                    test_metrics=[0.965, 0.8898, 0.8722, 0.9121], 
                    std_train_metrics=[0.0117, 0.0341, 0.0483, 0.0238], 
                    std_test_metrics=[0.0115, 0.0412, 0.0682, 0.0434], 
                    metrics_list=['Acc', 'F1', 'Prec', 'Rec'])
    barplot_figures(file_name='SCM_metrics_bar_plots', 
                    fig_title='metrics', 
                    saving_repo='./', 
                    train_metrics=[0.9801, 0.9385, 0.9455, 0.9324], 
                    test_metrics=[0.972, 0.91, 0.9148, 0.9085], 
                    std_train_metrics=[0.0048, 0.0145, 0.025, 0.0261], 
                    std_test_metrics=[0.0106, 0.0351, 0.0633, 0.0383], 
                    metrics_list=['Acc', 'F1', 'Prec', 'Rec'])
    barplot_figures(file_name='RF_metrics_bar_plots', 
                    fig_title='metrics', 
                    saving_repo='./', 
                    train_metrics=[0.9945, 0.9834, 0.9696, 0.9977], 
                    test_metrics=[0.9834, 0.948, 0.9429, 0.9545], 
                    std_train_metrics=[0.0025, 0.0074, 0.0121, 0.0039], 
                    std_test_metrics=[0.0092, 0.0287, 0.0428, 0.0319], 
                    metrics_list=['Acc', 'F1', 'Prec', 'Rec'])