import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import h5py
import scipy
from experiments.experiments_utilities import load_data, histogram_repo, parcours_one_directory
from sklearn.model_selection import train_test_split

def plot_stats_on_biogrid_distribution(dictionnaire, fig_name='biogrid_inner_genes_distribution.png'):
    """
    Args:
        dictionnaire, str, path to the dictionnary
    Returns:
        an histogram picture of the distribution on the biogrid groups pattern
    """
    dico = pickle.load(open(dictionnaire, 'rb'))
    if type(dico) == list:
        y = [len(el) for el in dico]
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, 'bo')
        plt.xlabel('Groups(Pathways): collection of genes intereacting with each other')
        plt.ylabel('Number of elements(genes) in each pathways') 
        plt.savefig(fig_name)
        plt.close()
    if type(dico) == dict:
        y = [len(v) for k, v in dico.items()]
        x = np.arange(1, len(dico.keys()) + 1)
        plt.plot(x, y, 'bo')
        plt.xlabel('Features')
        plt.ylabel('Number of pathways a feature is linked to') 
        plt.savefig(fig_name)
        plt.close()
        

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

    
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, "%.2f" % height, ha='center', va='bottom')


def generate_histogram(file_name, fig_title, accuracy_train, accuracy_test, f1_score_train, f1_score_test, 
                       precision_train, precision_test, recall_train, recall_test):
    train_metrics = np.asarray([np.round(np.mean(accuracy_train), 4), np.round(np.mean(f1_score_train), 4),
                               np.round(np.mean(precision_train), 4), np.round(np.mean(recall_train), 4)])
    test_metrics = np.asarray([np.round(np.mean(accuracy_test), 4), np.round(np.mean(f1_score_test), 4),
                               np.round(np.mean(precision_test), 4), np.round(np.mean(recall_test), 4)])
    std_train_metrics = np.asarray([np.round(np.std(accuracy_train), 4), np.round(np.std(f1_score_train), 4),
                               np.round(np.std(precision_train), 4), np.round(np.std(recall_train), 4)])
    std_test_metrics = np.asarray([np.round(np.std(accuracy_test), 4), np.round(np.std(f1_score_test), 4),
                               np.round(np.std(precision_test), 4), np.round(np.std(recall_test), 4)])
    
    nbResults = len(train_metrics)
    # figKW = {"figsize": (nbResults, 8)}
    # f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
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
    ax.set_xticklabels(['Acc', 'F1', 'Prec', 'Rec'])
    plt.tight_layout()
    f.savefig(f"{histogram_repo}/{file_name}.png")
    plt.close()


# CLICOMMAND THAT SHIT TOO
def generate_figures_mean_results(directory, experiment, f='exp', type_of_update='inner', random_weights='False'):
    """
    Utility function to plot the results for the groups method using the MEAN (Should rethink this probably)
    Args:
        directory,
        experiment, str, experiment name
        f, str, activation function name
        type_of_update, str, 
        random_weights, str
    """
    x = np.round(np.linspace(0.1, 1, 10), 3)
    os.chdir(f"{directory}")
    list_of_directories = os.listdir('./')
    list_of_directories = [directory for directory in list_of_directories if directory.startswith(experiment)] 
    list_of_directories = [directory for directory in list_of_directories if directory.find(f'{type_of_update}') != -1]
    list_of_directories = [directory for directory in list_of_directories if directory.find(f'{random_weights}') != -1]
    list_of_directories = list(np.sort(list_of_directories)) # garantie que ca va de 0.1 à 1.0 ici (sinon tjrs de min a max value de c)
    train_metrics_list = []; test_metrics_list = []; std_train_metrics_list = []; std_test_metrics_list = []
    for directory in list_of_directories:
        train_metrics, test_metrics, std_train_metrics, std_test_metrics, _, _, _ = parcours_one_directory(directory=directory)
        train_metrics_list.append(train_metrics)
        test_metrics_list.append(test_metrics)
        std_train_metrics_list.append(std_train_metrics)
        std_test_metrics_list.append(std_test_metrics)
    train_metrics_list = np.asarray(train_metrics_list)
    test_metrics_list = np.asarray(test_metrics_list)
    std_train_metrics_list = np.asarray(std_train_metrics_list)
    std_test_metrics_list = np.asarray(std_test_metrics_list)
    # Plot the train fig
    fig_title_train = f'Train mean metrics: Update Function:{f} {type_of_update}_groups random_weights: {random_weights}'
    fig_name_train = f'{f}_train_mean_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    f_train, ax_train = plt.subplots(nrows=1, ncols=1)
    ax_train.set_title(f"{fig_title_train}")
    ax_train.set_xlabel('c values')
    ax_train.set_ylabel('Metrics values')
    # ax.set_ylim(-0.1, 1.2)
    ax_train.plot(x, train_metrics_list[:, 0], 'bo-', label='Acc', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 1], 'ro-', label='F1 ', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 2], 'go-', label='Prec', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 3], 'ko-', label='Rec', linewidth=2)
    ax_train.legend()
    plt.tight_layout()
    f_train.savefig(f"{saving_repository}/{fig_name_train}")
    plt.close()
    
    # Plot the Test fig
    fig_title_test = f'Test mean metrics: {type_of_update}_groups random_weights: {random_weights}'
    fig_name_test = f'{f}_test_mean_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    f_test, ax_test = plt.subplots(nrows=1, ncols=1)
    ax_test.set_title(f"{fig_title_test}")
    ax_test.set_xlabel('c values')
    ax_test.set_ylabel('Metrics values')
    # ax.set_ylim(-0.1, 1.2)
    ax_test.plot(x, test_metrics_list[:, 0], 'bo-', label='Acc', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 1], 'ro-', label='F1 ', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 2], 'go-', label='Prec', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 3], 'ko-', label='Rec', linewidth=2)
    ax_test.legend()
    plt.tight_layout()
    f_test.savefig(f"{saving_repository}/{fig_name_test}")
    plt.close()
    os.chdir(f'{saving_repository}')
    
    
def generate_figures_best_results(directory, experiment, f='exp', type_of_update='inner', random_weights='False'):
    """
    Utility function to plot the results for the groups method using the BEST scores return (Should rethink this probably)
    Args:
        directory,
        experiment, str, experiment name
        f, str, activation function name
        type_of_update, str, 
        random_weights, str
    """
    x = np.round(np.linspace(0.1, 1, 10), 3)
    os.chdir(f"{directory}")
    list_of_directories = os.listdir('./')
    list_of_directories = [directory for directory in list_of_directories if directory.startswith(experiment)] 
    list_of_directories = [directory for directory in list_of_directories if directory.find(f'{type_of_update}') != -1]
    list_of_directories = [directory for directory in list_of_directories if directory.find(f'{random_weights}') != -1]
    list_of_directories = list(np.sort(list_of_directories)) # garantie que ca va de 0.1 à 1.0 ici (sinon tjrs de min a max value de c)
    train_metrics_list = []; test_metrics_list = []
    for directory in list_of_directories:
        _, _, _, _, train_metrics_best_file, test_metrics_best_file, _ = parcours_one_directory(directory=directory)
        train_metrics_list.append(train_metrics_best_file)
        test_metrics_list.append(test_metrics_best_file)
    train_metrics_list = np.asarray(train_metrics_list)
    test_metrics_list = np.asarray(test_metrics_list)
    # Plot the train fig
    fig_title_train = f'Train best metrics: Update Function:{f} {type_of_update}_groups random_weights: {random_weights}'
    fig_name_train = f'{f}_train_best_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    f_train, ax_train = plt.subplots(nrows=1, ncols=1)
    ax_train.set_title(f"{fig_title_train}")
    ax_train.set_xlabel('c values')
    ax_train.set_ylabel('Metrics values')
    # ax.set_ylim(-0.1, 1.2)
    ax_train.plot(x, train_metrics_list[:, 0], 'bo-', label='Acc', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 1], 'ro-', label='F1 ', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 2], 'go-', label='Prec', linewidth=2)
    ax_train.plot(x, train_metrics_list[:, 3], 'ko-', label='Rec', linewidth=2)
    ax_train.legend()
    plt.tight_layout()
    f_train.savefig(f"{saving_repository}/{fig_name_train}")
    plt.close()
    
    # Plot the Test fig
    fig_title_test = f'Test best metrics: {type_of_update}_groups random_weights: {random_weights}'
    fig_name_test = f'{f}_test_best_metrics_c_values_of_{type_of_update}_groups_random_weights_{random_weights}.png'
    f_test, ax_test = plt.subplots(nrows=1, ncols=1)
    ax_test.set_title(f"{fig_title_test}")
    ax_test.set_xlabel('c values')
    ax_test.set_ylabel('Metrics values')
    # ax.set_ylim(-0.1, 1.2)
    ax_test.plot(x, test_metrics_list[:, 0], 'bo-', label='Acc', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 1], 'ro-', label='F1 ', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 2], 'go-', label='Prec', linewidth=2)
    ax_test.plot(x, test_metrics_list[:, 3], 'ko-', label='Rec', linewidth=2)
    ax_test.legend()
    plt.tight_layout()
    f_test.savefig(f"{saving_repository}/{fig_name_test}")
    plt.close()
    os.chdir(f'{saving_repository}')
 
    
    
if __name__ == "__main__":
    boxplot_figures(data='/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5',
                    return_views='methyl_rna_iso_mirna_snp_clinical', 
                    saving_repo='/Users/maoss2/PycharmProjects/BRCA_experiments_and_paper', 
                    features_cibles=['uc002vwt.2_MLPH', 'uc002hul.3_RARA', 'cg00347904_SCUBE3', 'uc003jqp.2_INEXISTANT', 'cg06784466_FPR3'])
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