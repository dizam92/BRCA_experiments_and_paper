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
    
if __name__ == "__main__":
    cli()
    
# python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/triple_neg_new_labels_unbalanced_cpg_rna_rna_iso_mirna.h5 --sous-experiment-types 'methyl_rna_iso_mirna_snp_clinical' --cancer-name 'brca' --algo-used group_scm --target-features 'cg00347904_SCUBE3 cg20261915_GLP2R uc002acg.3_KIAA1370 cg20556988_CCL1 cg14620221_OR8B8 uc001xqa.2_LTBP2'

# python experiments/figures.py run-box-plot-fig --data /home/maoss2/PycharmProjects/BRCA_experiments_and_paper/datasets/datasets_repository/prad_cancer_metastase_vs_non_metastase.h5 --sous-experiment-types 'all' --cancer-name 'prad' --algo-used group_scm --target-features 'mrna_LRIT1 mrna_SUN3 mrna_AGMO mrna_TMEM71'
   