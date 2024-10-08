"""
Evaluating script for cell predictions.

Input format
------------
Centroids in a csv with columns X,Y and class. Both for prediction and GT.

Output
------
F1-score (binary, macro and weighted), accuracy, ROC AUC and error percentage between the prediction and the GT at cell-level.

Copyright (C) 2023  Jose Pérez Cano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact information: joseperez2000@hotmail.es
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from typing import List, Tuple, Dict

import sys
sys.path.append(r"..")
from utils.preprocessing import read_names, read_centroids
from utils.nearest import generate_tree, find_nearest
from utils.classification import metrics_from_predictions

import argparse
from argparse import Namespace
import logging
from logging import Logger


def get_confusion_matrix(
        gt_centroids: List[Tuple[int, int, int]],
        pred_centroids: List[Tuple[int, int, int]],
        eliminate_unmatched_GT: bool = False
        ) -> np.ndarray:
    """
    Calculates the confusion matrix based on the ground truth and predicted centroids.

    Each centroid is represented by a 3-tuple with (X, Y, class).
    The confusion matrix has (N+1)x(N+1) entries, where N is the maximum value encountered for the class.
    There is one additional entry for the background class when no match is found.

    :param gt_centroids: The ground truth centroids represented as a list of 3-tuples (X, Y, class).
    :type gt_centroids: List[Tuple[int, int, int]]
    :param pred_centroids: The predicted centroids represented as a list of 3-tuples (X, Y, class).
    :type pred_centroids: List[Tuple[int, int, int]]

    :return: The confusion matrix.
    :rtype: np.ndarray
    """
    if len(gt_centroids) == 0 and len(pred_centroids) == 0:
        return np.array([[0]])
    if len(gt_centroids) == 0:
        N = np.max(pred_centroids[:, 2])
        M = np.zeros((N + 1, N + 1))
        classes, freqs = np.unique(pred_centroids[:, 2], return_counts=True)
        M[0, classes] += freqs
        return M
    if len(pred_centroids) == 0:
        N = np.max(gt_centroids[:, 2])
        M = np.zeros((N + 1, N + 1))
        classes, freqs = np.unique(gt_centroids[:, 2], return_counts=True)
        M[classes, 0] += freqs
        return M

    if type(gt_centroids) == list:
        gt_centroids = np.array(gt_centroids)
    if type(pred_centroids) == list:
        pred_centroids = np.array(pred_centroids)

    N = int(max(np.max(gt_centroids[:, 2]), np.max(pred_centroids[:, 2])))
    # assert min(np.min(gt_centroids[:, 2]), np.min(pred_centroids[:, 2])) > 0, 'Zero should not be a class.'
    gt_tree = generate_tree(gt_centroids[:, :2])
    pred_tree = generate_tree(pred_centroids[:, :2])

    # Confusion matrix
    M = np.zeros((N + 1, N + 1))
    for point_id, point in enumerate(gt_centroids):
        closest_id, dist = find_nearest(point[:2], pred_tree, return_dist=True)
        closest = pred_centroids[closest_id]
        if point_id == find_nearest(closest[:2], gt_tree) and dist <= 50:
            M[int(point[2])][int(closest[2])] += 1  # 1-1 matchings
        else:
            if not eliminate_unmatched_GT:
                M[int(point[2])][0] += 1  # GT not matched in prediction

    for point_id, point in enumerate(pred_centroids):
        closest_id, dist = find_nearest(point[:2], gt_tree, return_dist=True)
        closest = gt_centroids[closest_id]
        if point_id != find_nearest(closest[:2], pred_tree) or dist > 50: # and dist <= 50
            M[0][int(point[2])] += 1  # Prediction not matched in GT

    return M


def get_pairs(
        gt_centroids: List[Tuple[int, int, int]],
        pred_centroids: List[Tuple[int, int, int]]
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves true and predicted labels ordered by their correspondences.

    Each centroid is represented by a 3-tuple with (X, Y, class).
    The class is represented as 1=non-tumour, 2=tumour.
    The returned labels are ordered starting from 0.

    :param gt_centroids: The ground truth centroids represented as a list of 3-tuples (X, Y, class).
    :type gt_centroids: List[Tuple[int, int, int]]
    :param pred_centroids: The predicted centroids represented as a list of 3-tuples (X, Y, class).
    :type pred_centroids: List[Tuple[int, int, int]]

    :return: A tuple containing the true labels and predicted labels ordered by their correspondences.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    if len(gt_centroids) == 0:
        return None, None

    # Trees
    gt_tree = generate_tree(gt_centroids[:, :2])
    pred_tree = generate_tree(pred_centroids[:, :2])

    # Pairs of labels
    true_labels, pred_labels = [], []
    for point_id, point in enumerate(gt_centroids):
        closest_id, dist = find_nearest(point[:2], pred_tree, return_dist=True)
        closest = pred_centroids[closest_id]

        true_label = point[2] - 1
        pred_label = closest[2] - 1

        if point_id == find_nearest(closest[:2], gt_tree) and dist <= 50:  # 1-1 matchings
            true_labels.append(true_label)
            pred_labels.append(pred_label)
        # The following cases can't be corrected by the GNN, so they are not taken into account.
        """else:                                               # GT not matched in prediction
            true_labels.append(true_label)
            pred_labels.append(-1)"""

    for point_id, point in enumerate(pred_centroids):
        closest_id, dist = find_nearest(point[:2], gt_tree, return_dist=True)
        closest = gt_centroids[closest_id]

        pred_label = point[2] - 1
        # true_label = closest[2] - 1

        if point_id != find_nearest(closest[:2], pred_tree) or dist > 50:    # Prediction not matched in GT
            true_labels.append(-1)
            pred_labels.append(pred_label)
    
    return np.array(true_labels), np.array(pred_labels)


def compute_f1_score_from_matrix(conf_mat: np.ndarray, cls: int) -> float:
    """
    Computes the F1 score of a given class against the rest.

    :param conf_mat: The confusion matrix.
    :type conf_mat: np.ndarray
    :param cls: The class for which to compute the F1 score.
    :type cls: int

    :return: The F1 score of the given class against the rest.
    :rtype: float
    """
    if cls == 0:
        return None
    TP = conf_mat[cls, cls]
    PP = conf_mat[:, cls].sum()
    if PP == 0:
        return None
    P = conf_mat[cls, :].sum()
    if P == 0:
        return None
    precision = TP / PP
    recall = TP / P
    if TP == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def compute_metrics_from_matrix(conf_mat: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes various evaluation metrics from a confusion matrix.

    Given a confusion matrix, this function calculates the F1 score, accuracy,
    ROC AUC, and percentage error.

    :param conf_mat: The confusion matrix.
    :type conf_mat: np.ndarray

    :return: A tuple containing the F1 score, accuracy, ROC AUC, and percentage error.
    :rtype: Tuple[float, float, float, float]
    """
    total = np.sum(conf_mat)
    acc = np.matrix.trace(conf_mat) / total
    n_classes = len(conf_mat)

    macro, weighted = 0, 0
    real_n_classes = n_classes
    adjust_weighted = 1

    for k in range(n_classes):
        f1_class = compute_f1_score_from_matrix(conf_mat, cls=k)
        cls_supp = conf_mat[k, :].sum() / total
        if f1_class is not None:
            macro += f1_class
            weighted += f1_class * cls_supp
        else:
            real_n_classes -= 1
            adjust_weighted -= cls_supp

    # Micro F-Score
    TP = np.matrix.trace(conf_mat[1:,1:])
    FP = np.sum(conf_mat[:,1:]) - TP
    FN = np.sum(conf_mat[1:,:]) - TP
    micro = TP / (TP + 0.5*(FP + FN))

    if real_n_classes != 0: 
        macro /= real_n_classes
        weighted /= adjust_weighted 
    else: 
        macro = -1
        weighted = -1
    
    return macro, weighted, micro


def additional_metrics(conf_mat: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], float]:
    if len(conf_mat) == 3:
        over_seg = (conf_mat[0,1]/np.sum(conf_mat[:,1]) if np.sum(conf_mat[:,1]) != 0 else 0,
                    conf_mat[0,2]/np.sum(conf_mat[:,2]) if np.sum(conf_mat[:,2]) != 0 else 0)
        under_seg = (conf_mat[1,0]/np.sum(conf_mat[1,:]) if np.sum(conf_mat[1,:]) != 0 else 0,
                    conf_mat[2,0]/np.sum(conf_mat[2,:]) if np.sum(conf_mat[2,:]) != 0 else 0)
        class_err = (conf_mat[2,1]/np.sum(conf_mat[1:,1]) if np.sum(conf_mat[1:,1]) != 0 else 0,
                    conf_mat[1,2]/np.sum(conf_mat[1:,2]) if np.sum(conf_mat[1:,2]) != 0 else 0)
        corrected_err = abs(np.sum(conf_mat[2,:])/np.sum(conf_mat[1:,:]) - np.sum(conf_mat[:,2])/np.sum(conf_mat[:,1:]))
    elif len(conf_mat) == 2:
        over_seg = (conf_mat[0,1]/np.sum(conf_mat[:,1]) if np.sum(conf_mat[:,1]) != 0 else 0, -1)
        under_seg = (conf_mat[1,0]/np.sum(conf_mat[1,:]) if np.sum(conf_mat[1,:]) != 0 else 0, -1)
        class_err = (-1, -1)
        corrected_err = abs(np.sum(conf_mat[1,:])/np.sum(conf_mat[1:,:]) - np.sum(conf_mat[:,1])/np.sum(conf_mat[:,1:]))
    
    return over_seg, under_seg, class_err, corrected_err


def add_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Adds two matrices even if their shapes are different.
    If B is bigger than A, A is enlarged with zeros.
    And vice versa.

    :param A: The first matrix.
    :type A: np.ndarray
    :param B: The second matrix.
    :type B: np.ndarray

    :return: The sum of the two matrices.
    :rtype: np.ndarray
    """
    if A.shape == B.shape:
        return A + B
    n, m = A.shape[0], B.shape[0]
    if m > n:
        res = np.zeros((m, m))
        res[:n, :n] += A
        res += B
    else:
        res = np.zeros((n, n))
        res[:m, :m] += B
        res += A
    return res


def compute_perc_error_bkgr(
        gt_centroids: List[Tuple[int, int, int]],
        pred_centroids: List[Tuple[int, int, int]]
        ) -> float:
    """
    Computes absolute difference in percentages of the classes.

    :param gt_centroids: Ground truth.
    :type gt_centroids: List[Tuple[int, int, int]]
    :param pred_centroids: Predictions.
    :type pred_centroids: List[Tuple[int, int, int]]
    :return: The percentage error.
    :rtype: float
    """
    gt_cls = gt_centroids[:, 2]
    pred_cls = pred_centroids[:, 2]
    gt_perc = (gt_cls == 2).sum() / len(gt_cls)
    pred_perc = (pred_cls == 2).sum() / len(pred_cls)
    return abs(gt_perc - pred_perc)


def save_csv(
        metrics: Dict[str, List[float]],
        save_path: str
        ) -> None:
    """
    Saves metrics in CSV format for later use.

    :param metrics: The metrics dictionary containing the data to be saved.
    :type metrics: Dict[str, List[float]]
    :param save_path: The file path where the CSV file will be saved.
    :type save_path: str
    """
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(save_path + '.csv', index=False)


def save_debug_matrix(
        mat: np.ndarray,
        save_path: str
        ) -> None:
    """
    Saves a confusion matrix in CSV format for debug purposes and in a PNG format for visualization.

    :param mat: The confusion matrix to be saved.
    :type mat: np.ndarray
    :param save_path: The file path where the CSV file will be saved.
    :type save_path: str
    """

    # CSV
    conf_mat_df = pd.DataFrame(mat)
    conf_mat_df.to_csv(save_path + '.csv')

    # PNG
    plt.figure(figsize=(15,15))
    g = sns.heatmap(mat, annot=True, fmt='g', annot_kws={"fontsize":16},)
    #g.set_xticklabels(['Background', 'Class 0', 'Class 1', 'Class 2', 'Class 3', 'Non-epith.'], fontsize=14)
    #g.set_yticklabels(['Background', 'Class 0', 'Class 1', 'Class 2', 'Class 3', 'Non-epith.'], fontsize=14)
    #g.set_xticklabels(['Background', 'Negative', 'Positive', 'Non-epith.'], fontsize=14)
    #g.set_yticklabels(['Background', 'Negative', 'Positive', 'Non-epith.'], fontsize=14)
    if len(mat) == 3:
        g.set_xticklabels(['Background', 'Negative', 'Positive'], fontsize=14) #
        g.set_yticklabels(['Background', 'Negative', 'Positive'], fontsize=14) #
    elif len(mat) == 2:
        g.set_xticklabels(['Background', 'Negative'], fontsize=14) #
        g.set_yticklabels(['Background', 'Negative'], fontsize=14) #

    # disp = ConfusionMatrixDisplay(mat)
    # disp.plot()

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)

    plt.savefig(save_path+'.png', bbox_inches='tight')
    plt.close()


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', type=str, required=True,
                        help='Path to txt file with names.')
    parser.add_argument('--gt-path', type=str, required=True,
                        help='Path to GT files.')
    parser.add_argument('--pred-path', type=str, required=True,
                        help='Path to prediction files.')
    parser.add_argument('--save-name', type=str, required=True,
                        help='Name to save the result, without file type.')
    parser.add_argument('--debug-path', type=str, default=None,
                        help='Name of file where to save confusion matrices (optional).')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--hov', type=bool, default=False, help='Whether the HoverNet is being evaluated or the GNN & Hov+GNN are.')
    return parser


def main_with_args(args: Namespace, logger: Logger):

    # Initialize variables
    names = args.names
    if args.num_classes == 2:
        metrics = {
            'Name': [], 'F1': [], 'Accuracy': [], 'ROC_AUC': [], 'Perc_err': [], 'ECE': [],
            'Macro F1 (bkgr)': [], 'Weighted F1 (bkgr)': [], 'Micro F1 (bkgr)': [], 'Perc_err (bkgr)': []
        }
    elif args.hov:
        metrics = {
            'Name': [], 'Macro F1 (Hov)': [], 'Weighted F1 (Hov)': [], 'Micro F1 (Hov)': [],
            'Over segmentation': [], 'Under segmentation': [], 'Classification error':[], 'Corrected error':[]
        }
    else: 
        metrics = {
            'Name': [], 'Macro F1 (GNN)': [], 'Weighted F1 (GNN)': [], 'Micro F1 (GNN)': [],
            'Macro F1 (Hov+GNN)': [], 'Weighted F1 (Hov+GNN)': [], 'Micro F1 (Hov+GNN)': [],
            'Over segmentation': [], 'Under segmentation': [], 'Classification error':[], 'Corrected error':[]
        }

    # Compute metrics
    global_conf_mat, global_conf_mat_nobkgr = None, None
    global_pred, global_true = [], []
    global_gt_centroids, global_pred_centroids = np.array([]).reshape(0,3), np.array([]).reshape(0,3)
    for k, name in enumerate(names):
        logger.info('Progress: {:2d}/{}'.format(k + 1, len(names)))
        metrics['Name'].append(name)

        # Read GT and prediction
        gt_centroids = read_centroids(name, args.gt_path)
        pred_centroids = read_centroids(name, args.pred_path)
        ## Drop the cells predicted as background
        nonbkgr_cells = np.where(pred_centroids[:, 2] != 0)[0]
        pred_centroids_nobkgr = pred_centroids[nonbkgr_cells, :]
        ##
        global_gt_centroids = np.vstack((global_gt_centroids, gt_centroids))
        global_pred_centroids = np.vstack((global_pred_centroids, pred_centroids))

        # Compute pairs and confusion matrix
        if not args.hov:
            conf_mat = get_confusion_matrix(gt_centroids, pred_centroids, eliminate_unmatched_GT=True)
        conf_mat_nobkgr = get_confusion_matrix(gt_centroids, pred_centroids_nobkgr, eliminate_unmatched_GT=False)
        if args.debug_path is not None:
            if not args.hov:
                save_debug_matrix(conf_mat, args.debug_path + '_' + name)
            save_debug_matrix(conf_mat_nobkgr, args.debug_path + '_' + name + '_nobkgr')
        try:
            if not args.hov:
                true_labels, pred_labels = get_pairs(gt_centroids, pred_centroids)
        except Exception as e:
            logger.error(e)
            logger.error(name)
            _metrics = [-1] * 5 if args.num_classes == 2 else [-1] * 4
            macro_nobkgr, weighted_nobkgr, micro_nobkgr = -1, -1, -1
            true_labels, pred_labels = None, None

        # Save labels
        if not args.hov:
            if true_labels is not None and pred_labels is not None:
                global_true.extend(true_labels)
                global_pred.extend(pred_labels)
            if global_conf_mat is None:
                global_conf_mat = conf_mat
            else:
                global_conf_mat = add_matrices(global_conf_mat, conf_mat)
        if global_conf_mat_nobkgr is None:
            global_conf_mat_nobkgr = conf_mat_nobkgr
        else:
            global_conf_mat_nobkgr = add_matrices(global_conf_mat_nobkgr, conf_mat_nobkgr)

        # Compute per image scores and percentages
        try:
            if not args.hov: 
                _metrics = metrics_from_predictions(true_labels, pred_labels, None, args.num_classes)
            macro_nobkgr, weighted_nobkgr, micro_nobkgr = compute_metrics_from_matrix(conf_mat_nobkgr)
            over_seg, under_seg, class_err, corrected_err = additional_metrics(conf_mat_nobkgr)
            if args.num_classes == 2:
                perc_error_bkgr = compute_perc_error_bkgr(gt_centroids, pred_centroids)
        except Exception as e:
            logger.error(e)
            logger.error(name)
            _metrics = [-1] * 5 if args.num_classes == 2 else [-1] * 4
            macro_nobkgr, weighted_nobkgr, micro_nobkgr = -1, -1, -1

        # Append metrics
        if args.num_classes == 2:
            acc, f1, auc, perc_error, ece = _metrics
            metrics['F1'].append(f1)
            metrics['Accuracy'].append(acc)
            metrics['ROC_AUC'].append(auc)
            metrics['Perc_err'].append(perc_error)
            metrics['ECE'].append(ece)
            metrics['Macro F1 (bkgr)'].append(macro_nobkgr)
            metrics['Weighted F1 (bkgr)'].append(weighted_nobkgr)
            metrics['Micro F1 (bkgr)'].append(micro_nobkgr)
            metrics['Perc_err (bkgr)'].append(perc_error_bkgr)
        elif args.hov:
            metrics['Macro F1 (Hov)'].append(macro_nobkgr)
            metrics['Weighted F1 (Hov)'].append(weighted_nobkgr)
            metrics['Micro F1 (Hov)'].append(micro_nobkgr)
            metrics['Over segmentation'].append(over_seg)
            metrics['Under segmentation'].append(under_seg)
            metrics['Classification error'].append(class_err)
            metrics['Corrected error'].append(corrected_err)
        else:
            micro, macro, weighted, ece = _metrics
            metrics['Macro F1 (GNN)'].append(macro)
            metrics['Weighted F1 (GNN)'].append(weighted)
            metrics['Micro F1 (GNN)'].append(micro)
            metrics['Macro F1 (Hov+GNN)'].append(macro_nobkgr)
            metrics['Weighted F1 (Hov+GNN)'].append(weighted_nobkgr)
            metrics['Micro F1 (Hov+GNN)'].append(micro_nobkgr)
            metrics['Over segmentation'].append(over_seg)
            metrics['Under segmentation'].append(under_seg)
            metrics['Classification error'].append(class_err)
            metrics['Corrected error'].append(corrected_err)

    # Save metrics
    save_csv(metrics, args.save_name)
    if args.debug_path is not None:
        if not args.hov:
            save_debug_matrix(global_conf_mat, args.debug_path + '_global')
        save_debug_matrix(global_conf_mat_nobkgr, args.debug_path + '_global_nobkgr')

    # Global scores and percentages
    if not args.hov:
        _global_metrics = metrics_from_predictions(np.array(global_true), np.array(global_pred), None, args.num_classes + 1) ###!!!!!!!! +1?
    macro_nobkgr, weighted_nobkgr, micro_nobkgr = compute_metrics_from_matrix(global_conf_mat_nobkgr)
    over_seg, under_seg, class_err, corrected_err = additional_metrics(global_conf_mat_nobkgr)
    if args.num_classes == 2:
        acc, f1, auc, perc_error, ece = _global_metrics
        perc_error_bkgr = compute_perc_error_bkgr(global_gt_centroids, global_pred_centroids)
        global_metrics = {
            'Name': ['All'], 'F1': [f1], 'Accuracy': [acc], 'ROC_AUC': [auc], 'Perc_err': [perc_error], 'ECE': [ece],
            'Macro F1 (bkgr)': [macro_nobkgr], 'Weighted F1 (bkgr)': [weighted_nobkgr], 'Micro F1 (bkgr)': [micro_nobkgr],
            'Perc_err (bkgr)': [perc_error_bkgr]
        }
    elif args.hov:
        global_metrics = {
            'Name': ['All'], 'Macro F1 (Hov)': [macro_nobkgr], 'Weighted F1 (Hov)': [weighted_nobkgr], 'Micro F1 (Hov)': [micro_nobkgr],
            'Over segmentation': [over_seg], 'Under segmentation': [under_seg], 'Classification error':[class_err], 'Corrected error':[corrected_err]
        }
    else:
        micro, macro, weighted, ece = _global_metrics 
        global_metrics = {
            'Name': ['All'], 'Macro F1 (GNN)': [macro], 'Weighted F1 (GNN)': [micro], 'Micro F1 (GNN)': [weighted],
            'Macro F1 (Hov+GNN)': [macro_nobkgr], 'Weighted F1 (Hov+GNN)': [weighted_nobkgr], 'Micro F1 (Hov+GNN)': [micro_nobkgr],
            'Over segmentation': [over_seg], 'Under segmentation': [under_seg], 'Classification error':[class_err], 'Corrected error':[corrected_err]
        }
        
    save_csv(global_metrics, args.save_name + '_all')


def main():
    parser = _create_parser()
    args = parser.parse_args()

    logger = logging.getLogger('eval_segment')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    main(args, logger)
