import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import h5py
import scipy
import pickle
import seaborn as sns
import click
from experiments.experiments_utilities import load_data, load_prad_data, histogram_repo, brca_dictionnary_for_prior_rules, prad_dictionnary_for_prior_rules
from sklearn.model_selection import train_test_split

@click.group()
def cli():
    pass

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
        plt.savefig(f"{histogram_repo}/{fig_name}")
        plt.close()
    if type(dico) == dict:
        y = [len(v) for k, v in dico.items()]
        x = np.arange(1, len(dico.keys()) + 1)
        plt.plot(x, y, 'bo')
        plt.xlabel('Features')
        plt.ylabel('Number of pathways a feature is linked to') 
        plt.savefig(fig_name)
        plt.savefig(f"{histogram_repo}/{fig_name}")
        plt.close()
        

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
    

def boxplot_figures(data, return_views, cancer_name, algo_used, features_cibles, saving_repo):
    """ 
    Args:  
        data, str, datapath
        return_views, str, the view
        cancer_name, str, cancer name
        algo_used, str, name of the algo used (for the figure name) must be in [dt, scm, rf, group_scm]
        features_cibles, list of features to built the boxplot
    Returns:
        figures
    """
    if cancer_name == 'brca':
        x, y, features_names, patients_names = load_data(data=data, return_views=return_views)
    elif cancer_name == 'prad':
        x, y, features_names, patients_names = load_prad_data(data=data, return_views=return_views)
    else:
        raise ValueError(f'{cancer_name} is not recognize yet')
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
        sns.set_style("darkgrid")
        ax = fig.add_subplot(111)
        ax.boxplot(data, notch=0, sym='b+', vert=1, whis=1.5, positions=None, widths=0.6)
        ax.annotate("", xy=(1, y_max), xycoords='data', xytext=(2, y_max), textcoords='data', 
                    arrowprops=dict(arrowstyle="-", ec='#aaaaaa', connectionstyle="bar,fraction=0.2"))
        ax.text(1.5, y_max + abs(y_max - y_min) * 0.1, stars(p_value), horizontalalignment='center', verticalalignment='center')

        plt.xlabel(f'Positives                                               Negatives')
        # plt.xlabel(f'Positives\t\t\t\t\t\t\t\t\t\t\t\tNegatives')
        plt.ylabel(f'{feat[1]}')
        # plt.boxplot(data=data)
        feat_name = feat[1].replace('.', '')
        feat_name = feat_name.replace('_', '')
        fig_name = f'{feat_name}_{algo_used}_{cancer_name}'
        plt.tight_layout()
        fig.savefig(f"{saving_repo}/{fig_name}_BoxPlots.png")
        plt.close()


@cli.command(help="Run the analysis results")
@click.option('--data', type=str, default=None, help="""data path""")
@click.option('--sous-experiment-types', type=str, default='all', help="""name of experiment in results_views""")
@click.option('--cancer-name', type=str, default='brca', help="""cancer name""")
@click.option('--algo-used', type=str, default='scm', help="""name of the algo used""")
@click.option('--target-features', type=str, default='dt scm rf', help="""list of feature to bocplot figure""")
@click.option('--output', type=str, default=histogram_repo, help="""saving repo path""")
def run_box_plot_fig(data, sous_experiment_types, cancer_name, algo_used, output, target_features):
    boxplot_figures(data=data, return_views=sous_experiment_types, cancer_name=cancer_name, algo_used=algo_used, 
                    features_cibles=target_features.split(), saving_repo=output)
    if cancer_name == 'brca':
        plot_stats_on_biogrid_distribution(dictionnaire=brca_dictionnary_for_prior_rules, fig_name='brca_biogrid_inner_genes_distribution.png')
    else:
        plot_stats_on_biogrid_distribution(dictionnaire=prad_dictionnary_for_prior_rules, fig_name='prad_biogrid_inner_genes_distribution.png')
    

def add_boxplot_figures(data, return_views, cancer_name, algo_used, features_cibles, saving_repo):
    """ 
    This is Francois version, in order to take into account the impact of both of the features in the boxplot
    Args:  
        data, str, datapath
        return_views, str, the view
        cancer_name, str, cancer name
        algo_used, str, name of the algo used (for the figure name) must be in [dt, scm, rf, group_scm]
        features_cibles, list of features to built the boxplot
    Returns:
        figures
    """
    if cancer_name == 'brca':
        x, y, features_names, patients_names = load_data(data=data, return_views=return_views)
    elif cancer_name == 'prad':
        x, y, features_names, patients_names = load_prad_data(data=data, return_views=return_views)
    else:
        raise ValueError(f'{cancer_name} is not recognize yet')
    index_patients_positifs = np.where(y == 1)[0]
    index_patients_negatifs = np.where(y == -1)[0]
    x_patients_positifs = x[index_patients_positifs]
    x_patients_negatifs = x[index_patients_negatifs]
    features_cibles_index = [(i, el) for feat_cibl in features_cibles for i, el in enumerate(features_names) if feat_cibl == el]
    data_positif = x_patients_positifs[:, features_cibles_index[0][0]]
    data_negatif = x_patients_negatifs[:, features_cibles_index[0][0]]
    feat_name = features_cibles_index[0][1]
    for feat in features_cibles_index[1:]: # on va du 2eme element vu qu'on a déja recuperer le 1er plus haut
        data_positif += x_patients_positifs[:, feat[0]]
        data_negatif += x_patients_negatifs[:, feat[0]]
        feat_name += f'|{feat[1]}'
    z, p = scipy.stats.mannwhitneyu(data_positif, data_negatif)
    p_value = p * 2
    y_max = np.max(np.concatenate((data_positif, data_negatif)))
    y_min = np.min(np.concatenate((data_positif, data_negatif)))
    data = [data_positif, data_negatif]
    fig = plt.figure()
    sns.set_style("darkgrid")
    ax = fig.add_subplot(111)
    ax.boxplot(data, notch=0, sym='b+', vert=1, whis=1.5, positions=None, widths=0.6)
    ax.annotate("", xy=(1, y_max), xycoords='data', xytext=(2, y_max), textcoords='data', 
                arrowprops=dict(arrowstyle="-", ec='#aaaaaa', connectionstyle="bar,fraction=0.2"))
    ax.text(1.5, y_max + abs(y_max - y_min) * 0.1, stars(p_value), horizontalalignment='center', verticalalignment='center')

    plt.xlabel(f'Positives                                               Negatives')
    plt.ylabel(f'{feat_name}')
    feat_name = feat_name.replace('_', '')
    fig_name = f'{feat_name}_{algo_used}_{cancer_name}'
    plt.tight_layout()
    fig.savefig(f'{saving_repo}/{fig_name}_BoxPlots.png')
    plt.close()


@cli.command(help="Run additicf boxplots analysis results")
@click.option('--data', type=str, default=None, help="""data path""")
@click.option('--sous-experiment-types', type=str, default='all', help="""name of experiment in results_views""")
@click.option('--cancer-name', type=str, default='brca', help="""cancer name""")
@click.option('--algo-used', type=str, default='scm', help="""name of the algo used""")
@click.option('--target-features', type=str, default='dt scm rf', help="""list of feature to bocplot figure""")
@click.option('--output', type=str, default=histogram_repo, help="""saving repo path""")
def run_add_box_plot_fig(data, sous_experiment_types, cancer_name, algo_used, output, target_features):
    add_boxplot_figures(data=data, return_views=sous_experiment_types, cancer_name=cancer_name, algo_used=algo_used, 
                    features_cibles=target_features.split(), saving_repo=output)
    
    
if __name__ == "__main__":
    cli()

