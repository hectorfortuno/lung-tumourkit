
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import argparse
from argparse import Namespace
import os
import cv2

# import sys
# sys.path.append(r"../")
from utils.preprocessing import create_dir, parse_path, get_names, read_png, read_json, read_npy, save_graph
from utils.nearest import generate_tree, find_nearest
from utils.postprocessing import get_bounding_box

import logging
from logging import Logger


def parse_centroids_probs(nuc: Dict[str, Any], logger: Optional[Logger] = None, num_classes: Optional[int] = 2) -> List[Tuple[int, int, int]]:
    """
    Parses the centroids and probabilities from the Hovernet JSON nuclei dictionary.

    This function takes the Hovernet JSON nuclei dictionary as input, which is obtained from the modified run_infer.py script.
    It extracts the centroids and probabilities from the dictionary and returns them as a list of tuples.
    Each tuple represents a centroid and consists of (X, Y, prob1) for binary classification or (X, Y, prob1, prob2, ..., probN) for multiclass classification.

    :param nuc: The Hovernet JSON nuclei dictionary.
    :type nuc: Dict[str, Any]
    :param logger: Optional logger object to log warnings.
    :type logger: Optional[Logger]
    :param num_classes: Optional number of classes. Defaults to 2 for binary classification.
    :type num_classes: Optional[int]
    :return: The list of centroids with probabilities.
    :rtype: List[Tuple[int, int, int]]
    """
    centroids_ = []
    for inst in nuc:
        inst_info = nuc[inst]
        inst_centroid = inst_info['centroid']
        inst_prob1 = inst_info['prob1']
        if num_classes > 2:
            inst_probs = []
            for k in range(1, num_classes + 1):
                inst_probs.append(inst_info['prob' + str(k)])
        inst_type = inst_info['type']
        if inst_type == 0:
            if logger is None:
                logging.warning('Found cell with class 0, removing it.')
            else:
                logger.warning('Found cell with class 0, removing it.')
        else:
            if num_classes == 2:
                centroids_.append((inst_centroid[1], inst_centroid[0], inst_prob1))
            else:
                centroids_.append((inst_centroid[1], inst_centroid[0], *inst_probs))
    return centroids_


def add_probability(
        png: np.ndarray,
        graph: pd.DataFrame,
        hov_npy: Dict[str, Any],
        logger: Optional[Logger] = None,
        num_classes: Optional[int] = 2,
        enable_background: Optional[bool] = False,
        ) -> pd.DataFrame:
    """
    Adds probability information from the Hovernet NPY nuclei dictionary to the graph DataFrame.

    This function extracts the type probabilities from the Hovernet NPY nuclei array (`hov_npy`) and adds them as columns to the `graph` DataFrame.
    The join between the `graph` DataFrame and the `hov_npy` dictionary is based on the 'id' field.

    :param graph: The graph DataFrame containing the nodes.
    :type graph: pd.DataFrame
    :param hov_npy: The Hovernet NPY nuclei array.
    :type hov_npy: np.array
    :param logger: Optional logger object to log warnings.
    :type logger: Optional[Logger]
    :param num_classes: Optional number of classes. Defaults to 2 for binary classification.
    :type num_classes: Optional[int]
    :param enable_background: If True, extra cells are included. They are represented with a value of 1 in the column background.
    :type enable_background: Optional[bool]

    :return: The updated graph DataFrame with probability information.
    :rtype: pd.DataFrame
    """

    graph = graph.copy()

    # Initialize "prob1"
    if 'prob1' not in graph.columns:
        n_cols = len(graph.columns)
        graph.insert(n_cols, 'prob1', [-1] * len(graph))
    else:
        graph['prob1'] = -1

    # Background to cells (initialize)
    if enable_background:
        if 'background' not in graph.columns:
            n_cols = len(graph.columns)
            graph.insert(n_cols, 'background', [0] * len(graph))
        else:
            graph['background'] = 0

    # Multi-class probabilities (initialize)
    if num_classes > 2:
        for k in range(2, num_classes + 1):
            if not 'prob' + str(k) in graph.columns:
                n_cols = len(graph.columns)
                graph.insert(n_cols, 'prob' + str(k), [-1] * len(graph))
            else:
                graph['prob' + str(k)] = -1

    # Instances/cells
    centroids = graph[['X', 'Y']].to_numpy(dtype=int)
    for i, point in enumerate(centroids):

        # Cell identifier
        inst_id = png[point[0], point[1]]
        inst_map = (png == inst_id)

        # Bounding box of the cell
        rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])

        # Cell information
        inst_map_crop = png[rmin:rmax, cmin:cmax]
        inst_type_crop = hov_npy[rmin:rmax, cmin:cmax]

        # TP prediction only on the cell, not background
        inst_map_crop = (inst_map_crop == inst_id)  # TODO: duplicated operation, may be expensive
        inst_type = np.argmax(inst_type_crop[inst_map_crop], axis=-1)

        # Compute type of cell and probability of each class
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        type_dict = {v[0]: v[1] for v in type_list}

        # Multi-class probabilities
        if int(num_classes) > 3:
            total_sum = np.sum(list(type_dict.values()))
            for k in range(int(num_classes)):
                if not k in type_dict:
                    type_count = 0
                else:
                    type_count = type_dict[k]
                graph.loc[i, "prob" + str(k+1)] = float(type_count / (total_sum + 1.0e-6))

    return graph


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy-dir', type=str, required=True, help='Path to folder containing HoVer-Net npy outputs (nuclear map prediction).')
    parser.add_argument('--graph-dir', type=str, required=True, help='Path to directory to .nodes.csv containing graph information.')
    parser.add_argument('--output-dir', type=str, required=True, help='Path where to save new .nodes.csv. If same as --graph-dir, overwrites its content.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')
    return parser


def main_with_args(args: Namespace, logger: Optional[Logger] = None) -> None:

    # Data directories
    npy_dir = parse_path(args.npy_dir)
    png_dir = parse_path(args.png_dir)
    graph_dir = parse_path(args.graph_dir)
    output_dir = parse_path(args.output_dir)

    create_dir(output_dir)

    # Add probabilities
    names = get_names(graph_dir, '.nodes.csv')
    for name in tqdm(names):
        try:
            png = read_png(name, png_dir)
            graph = pd.read_csv(os.path.join(graph_dir, name + '.nodes.csv'))
            hov_npy = read_npy(name, npy_dir)
            graph = add_probability(png, graph, hov_npy, logger, args.num_classes, args.enable_background)
            save_graph(graph, os.path.join(output_dir, name + '.nodes.csv'))
        except FileNotFoundError:
            continue
    return


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)