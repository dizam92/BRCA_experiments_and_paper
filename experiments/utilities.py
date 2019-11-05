# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import random
import os
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from itertools import combinations
from glob import glob
from collections import Counter


def load_data(data, return_views='all'):
    """
    Load the triple neg datasets
    Args:
        data, str, path to the .h5py dataset
        return_views, str, the type of the x we want to return, default 'all'
                - 'methyl_rna_iso_mirna' for methyl + rna_iso + mirna,
                - 'methyl_rna_iso_mirna_snp_clinical' for methyl + rna_iso + mirna + snp + clinical,
                - 'methyl_rna_mirna' for methyl + rna + mirna,
                - 'methyl_rna_mirna_snp_clinical' for methyl + rna_iso + rna + snp + clinical,
                - 'all' for all
                - 'majority_vote' for the majority vote experiments
    Returns:
        x, y, features_names, patients_names
    """
    assert return_views in ['methyl_rna_iso_mirna', 'methyl_rna_iso_mirna_snp_clinical',
                            'methyl_rna_mirna', 'methyl_rna_mirna_snp_clinical', 'all', 'majority_vote']
    d = h5py.File(data, 'r')
    x = d['data'][()]
    y = d['target'][()]
    features_names = d['features_names'][()]
    features_names = np.asarray([el.decode("utf-8") for el in features_names])
    patients_names = d['patients_ids'][()]
    patients_names = np.asarray([el.decode("utf-8") for el in patients_names])
    random.seed(42)
    data_x_y_patients_names = list(zip(x, y, patients_names))
    random.shuffle(data_x_y_patients_names)
    x = [el[0] for el in data_x_y_patients_names]
    y = [el[1] for el in data_x_y_patients_names]
    patients_names = [el[2] for el in data_x_y_patients_names]
    x = np.asarray(x)
    y = np.asarray(y)
    patients_names = np.asarray(patients_names)
    x_methyl = x[:, 0:23381]
    features_names_methyl = features_names[0:23381]
    # RNA ISO
    x_rna_iso = x[:, 23381:96980]
    features_names_rna_iso = features_names[23381:96980]
    # MiRNA
    x_mirna = x[:, 96980:98026]
    features_names_mirna = features_names[96980:98026]
    # SNP
    x_snp = x[:, 98026:102218]
    features_names_snp = features_names[98026:102218]
    # Clinical
    x_clinical = x[:, 102218:102235]
    features_names_clinical = features_names[102218:102235]
    # RNA
    x_rna = x[:, 102235:122767]
    features_names_rna = features_names[102235:122767]
    # Normalization
    x_rna_iso = StandardScaler().fit_transform(x_rna_iso)
    x_mirna = StandardScaler().fit_transform(x_mirna)
    x_rna = StandardScaler().fit_transform(x_rna)
    if return_views == 'methyl_rna_iso_mirna':
        x_methyl_rna_iso_mirna = np.hstack((x_methyl, x_rna_iso, x_mirna))
        features_names_rna_iso_mirna = np.hstack((features_names_methyl, features_names_rna_iso, features_names_mirna))
        x = x_methyl_rna_iso_mirna
        features_names = features_names_rna_iso_mirna
        x = x.T
        data_x_names = list(zip(x, features_names))
        random.seed(42)
        random.shuffle(data_x_names)
        x = [el[0] for el in data_x_names]
        features_names = [el[1] for el in data_x_names]
        x = np.asarray(x)
        x = x.T
        features_names = np.asarray(features_names)
        return x, y, features_names, patients_names

    if return_views == 'methyl_rna_iso_mirna_snp_clinical':
        x_methyl_rna_iso_mirna_snp_clinical = np.hstack((x_methyl, x_rna_iso, x_mirna, x_snp, x_clinical))
        features_names_rna_iso_mirna_snp_clinical = np.hstack((features_names_methyl, features_names_rna_iso,
                                                               features_names_mirna, features_names_snp,
                                                               features_names_clinical))
        x = x_methyl_rna_iso_mirna_snp_clinical
        features_names = features_names_rna_iso_mirna_snp_clinical
        x = x.T
        data_x_names = list(zip(x, features_names))
        random.seed(42)
        random.shuffle(data_x_names)
        x = [el[0] for el in data_x_names]
        features_names = [el[1] for el in data_x_names]
        x = np.asarray(x)
        x = x.T
        features_names = np.asarray(features_names)
        return x, y, features_names, patients_names

    if return_views == 'methyl_rna_mirna':
        x_methyl_rna_mirna = np.hstack((x_methyl, x_rna, x_mirna))
        features_names_rna_mirna = np.hstack((features_names_methyl, features_names_rna, features_names_mirna))
        x = x_methyl_rna_mirna
        features_names = features_names_rna_mirna
        x = x.T
        data_x_names = list(zip(x, features_names))
        random.seed(42)
        random.shuffle(data_x_names)
        x = [el[0] for el in data_x_names]
        features_names = [el[1] for el in data_x_names]
        x = np.asarray(x)
        x = x.T
        features_names = np.asarray(features_names)
        return x, y, features_names, patients_names

    if return_views == 'methyl_rna_mirna_snp_clinical':
        x_methyl_rna_mirna_snp_clinical = np.hstack((x_methyl, x_rna, x_mirna, x_snp, x_clinical))
        features_names_rna_mirna_snp_clinical = np.hstack((features_names_methyl, features_names_rna,
                                                           features_names_mirna, features_names_snp,
                                                           features_names_clinical))
        x = x_methyl_rna_mirna_snp_clinical
        features_names = features_names_rna_mirna_snp_clinical
        x = x.T
        data_x_names = list(zip(x, features_names))
        random.seed(42)
        random.shuffle(data_x_names)
        x = [el[0] for el in data_x_names]
        features_names = [el[1] for el in data_x_names]
        x = np.asarray(x)
        x = x.T
        features_names = np.asarray(features_names)
        return x, y, features_names, patients_names

    if return_views == 'all':
        x_all = np.hstack((x_methyl, x_rna, x_rna_iso, x_mirna, x_snp, x_clinical))
        x = x_all
        x = x.T
        data_x_names = list(zip(x, features_names))
        random.seed(42)
        random.shuffle(data_x_names)
        x = [el[0] for el in data_x_names]
        features_names = [el[1] for el in data_x_names]
        x = np.asarray(x)
        x = x.T
        features_names = np.asarray(features_names)
        return x, y, features_names, patients_names

    if return_views == 'majority_vote':
        x_all = np.hstack((x_methyl, x_rna, x_rna_iso, x_mirna, x_snp, x_clinical))
        x = x_all
        return x, x_methyl, x_rna, x_rna_iso, x_mirna, x_snp, x_clinical, y, features_names, features_names_methyl, \
               features_names_rna, features_names_rna_iso, features_names_mirna, features_names_snp, \
               features_names_clinical, patients_names


def results_analysis(directory, output_text_file):
    """
    An utility function to run the results analysis and output them in a readable way
    Args:
        directory, str, path to the directory containing the pickle files
        data_path, str, path to the data of interest to be loaded to run the analysis
        output_text_file, str, file where to write the results to
    Returns:
        Write results to text file
    """
    os.chdir('{}'.format(directory))
    metrics_train = []
    metrics_test = []
    features_retenus = []
    model_comptes = []
    cnt = Counter()
    cnt_rf = Counter()
    for fichier in glob("*.pck"):
        f = open(fichier, 'rb')
        d = pickle.load(f)
        metrics_train.append(d['train_metrics'])
        metrics_test.append(d['metrics'])
        features_retenus.append(d['rules_str'])
    accuracy_train = [el['accuracy'] for el in metrics_train]
    accuracy_test = [el['accuracy'] for el in metrics_test]
    f1_score_train = [el['f1_score'] for el in metrics_train]
    f1_score_test = [el['f1_score'] for el in metrics_test]
    precision_train = [el['precision'] for el in metrics_train]
    precision_test = [el['precision'] for el in metrics_test]
    recall_train = [el['recall'] for el in metrics_train]
    recall_test = [el['recall'] for el in metrics_test]
    if directory.find('dt') != -1:
        for model in features_retenus:
            temp = []
            for el in model:
                if el[2] > 0:
                    temp.append(el)
            var = ''
            for i, el in enumerate(temp):
                var += '_{}'.format(el[3])
                if i == 2:
                    break
            model_comptes.append(var)
    if directory.find('rf') != -1:
        for model in features_retenus:
            var = ''
            for el in model[:3]:
                var += '_{}'.format(el[3])
            model_comptes.append(var)

    if directory.find('scm') != -1:
        for model in features_retenus:
            temp = []
            for el in model:
                temp.append(el[1])
            var = ''
            for el in temp:
                var += '_{}'.format(el)
            model_comptes.append(var)
    with open(output_text_file, 'w') as f:
        f.write('TRAINING RESULTS\n')
        f.write('-*50\n')
        f.write('Training: Accuracy mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(accuracy_train), 4), np.round(np.std(accuracy_train), 4), np.round(np.max(accuracy_train), 4),
            np.round(np.min(accuracy_train), 4), np.round(np.median(accuracy_train), 4)))
        f.write('Training: f1_score mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(f1_score_train), 4), np.round(np.std(f1_score_train), 4), np.round(np.max(f1_score_train), 4),
            np.round(np.min(f1_score_train), 4), np.round(np.median(f1_score_train), 4)))
        f.write('Training: Precision mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(precision_train), 4), np.round(np.std(precision_train), 4),
            np.round(np.max(precision_train), 4),
            np.round(np.min(precision_train), 4), np.round(np.median(precision_train), 4)))
        f.write('Training: Recall mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(recall_train), 4), np.round(np.std(recall_train), 4), np.round(np.max(recall_train), 4),
            np.round(np.min(recall_train), 4), np.round(np.median(recall_train), 4)))
        f.write('TESTS RESULTS\n')
        f.write('-*50\n')
        f.write('Test: Accuracy mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(accuracy_test), 4), np.round(np.std(accuracy_test), 4), np.round(np.max(accuracy_test), 4),
            np.round(np.min(accuracy_test), 4), np.round(np.median(accuracy_test), 4)))
        f.write('Test: f1_score mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(f1_score_test), 4), np.round(np.std(f1_score_test), 4), np.round(np.max(f1_score_test), 4),
            np.round(np.min(f1_score_test), 4), np.round(np.median(f1_score_test), 4)))
        f.write('Test: Precision mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(precision_test), 4), np.round(np.std(precision_test), 4), np.round(np.max(precision_test), 4),
            np.round(np.min(precision_test), 4), np.round(np.median(precision_test), 4)))
        f.write('Test: Recall mean {} +/- {}; Max value: {}, Min value: {}, Median value: {}\n'.format(
            np.round(np.mean(recall_test), 4), np.round(np.std(recall_test), 4), np.round(np.max(recall_test), 4),
            np.round(np.min(recall_test), 4), np.round(np.median(recall_test), 4)))

        for el in model_comptes:
            cnt[el] += 1
        f.write('Most frequent model\n'.format(cnt.most_common(10)))

    if directory.find('rf') != -1:
        features_retenus_flatten = [el[3] for liste in features_retenus for el in liste[:50]]
        for el in features_retenus_flatten:
            cnt_rf[el] += 1
        most_common_features = cnt_rf.most_common(50)
        most_common_features_names = np.asarray([el[0] for el in most_common_features])
        most_common_features_values = np.asarray([el[1] for el in most_common_features])
        nbResults = len(most_common_features)
        figKW = {"figsize": (nbResults, 8)}
        f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
        barWidth = 0.35
        ax.set_title('{}'.format('RF 50 most common features in the 50 best features for each experiment'))
        rects = ax.bar(range(nbResults), most_common_features_values, barWidth, color="r")
        autolabel(rects, ax)
        # ax.legend(rects[0], 'Counts')
        # ax.set_ylim(-0.1, 1.1)
        ax.set_xticks(np.arange(nbResults) + barWidth)
        ax.set_xticklabels(most_common_features_names, rotation="vertical")
        plt.tight_layout()
        f.savefig('RF_50_Most_Common_Features' + time.strftime("%Y%m%d-%H%M%S") + ".png")
        plt.close()


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, "%.2f" % height, ha='center', va='bottom')


def weighted_sample(y, y_target):
    """ Build a weighted sample array
    Args:
         y: the original target array
         y_target: the y for which we want to build a sample size
    Returns:
        y_samples, weighted
    """
    y = np.asarray(y)
    y_target = np.asarray(y_target)
    y_samples = np.ones((y_target.size,))
    nb_classes = np.unique(y).size
    negative_type = -1
    if 0 in np.unique(y):
        negative_type = 0
    nb_negatives = np.where(y == negative_type)[0].size
    nb_positives = np.where(y == 1)[0].size
    c_n = 1.0 / (nb_classes * nb_negatives)
    c_p = 1.0 / (nb_classes * nb_positives)
    y_samples[np.where(y_target == negative_type)[0]] *= c_n
    y_samples[np.where(y_target == 1)[0]] *= c_p
    return y_samples


def get_metrics(y_test, predictions_binary):
    """Compute the metrics for classifiers predictors
    Args:
        y_test: real labels
        predictions_binary: the predicted labels
    Return: metrics: a dictionnary of the metrics
    """
    y_test = np.asarray(y_test, dtype=np.float)
    predictions_binary = np.asarray(predictions_binary, dtype=np.float)
    metrics = {"accuracy": accuracy_score(y_test, predictions_binary),
               "f1_score": f1_score(y_test, predictions_binary),
               "precision": precision_score(y_test, predictions_binary),
               "recall": recall_score(y_test, predictions_binary)
               }
    return metrics


def get_metrics_balanced(y_test, predictions_binary, weights):
    """Compute the balanced metrics for classifiers predictors
    Args: y_test: real labels
            predictions_binary: the predicted labels
            weights: the weights used for learning
    Return: metrics: a dictionnary of the metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    y_test = np.asarray(y_test, dtype=np.float)
    predictions_binary = np.asarray(predictions_binary, dtype=np.float)

    metrics = {"accuracy": accuracy_score(y_test, predictions_binary, sample_weight=weights),
               "f1_score": f1_score(y_test, predictions_binary),
               "precision": precision_score(y_test, predictions_binary),
               "recall": recall_score(y_test, predictions_binary)
               }
    return metrics


# TODO: To be modified and adapted to the new outing: COme back here soon
def gen_histogram_results(pattern_to_search='*_unbalanced_*.pck', metric='accuracy', directory='results_analysis',
                          results_path='/home/maoss2/PycharmProjects/breast_cancer/experimentations/Results'):
    os.chdir('{}'.format(results_path))
    assert metric in ['accuracy', 'f1_score', 'precision', 'recall'], 'metric {} is not implemented yet'.format(metric)
    noms_fichiers = []
    metric_train = []
    metric_test = []

    for fichier in glob('{}'.format(pattern_to_search)):
        noms_fichiers.append(fichier)
        f = open(fichier, 'r')
        d = pickle.load(f)
        if metric == 'accuracy':
            metric_train.append(d['train_metrics']['accuracy'])
            metric_test.append(d['metrics']['accuracy'])
        if metric == 'precision':
            metric_train.append(d['train_metrics']['precision'])
            metric_test.append(d['metrics']['precision'])
        if metric == 'recall':
            metric_train.append(d['train_metrics']['recall'])
            metric_test.append(d['metrics']['recall'])
        if metric == 'f1_score':
            metric_train.append(d['train_metrics']['f1_score'])
            metric_test.append(d['metrics']['f1_score'])
    noms_fichiers = np.asarray(noms_fichiers)
    metric_test = np.asarray(metric_test)
    metric_train = np.asarray(metric_train)
    nbResults = len(metric_train)
    figKW = {"figsize": (nbResults, 3.0 / 4 * nbResults + 2.0)}
    f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    barWidth = 0.35
    sorted_indices = np.argsort(metric_test)
    testScores = metric_test[sorted_indices]
    trainScores = metric_train[sorted_indices]
    names = noms_fichiers[sorted_indices]
    ax.set_title(''.format(metric))
    rects = ax.bar(range(nbResults), testScores, barWidth, color="r", )
    rect2 = ax.bar(np.arange(nbResults) + barWidth, trainScores, barWidth, color="0.7", )
    autolabel(rects, ax)
    autolabel(rect2, ax)
    ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(np.arange(nbResults) + barWidth)
    ax.set_xticklabels(names, rotation="vertical")
    plt.tight_layout()
    f.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + '_unbalanced_metric_analysis_{}'.format(metric) + ".png")
    plt.close()
