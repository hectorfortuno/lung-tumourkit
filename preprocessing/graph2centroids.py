"""
Converts the .nodes.csv files into .centroids.csv files.

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
import argparse
from argparse import Namespace
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

# import sys
# sys.path.append(r"..")
from utils.preprocessing import get_names, create_dir, parse_path, save_centroids, read_graph

def graph2centroids(
        graph_file: pd.DataFrame,
        num_classes: int,
        enable_background: Optional[bool] = False) -> np.ndarray:
    """
    Extracts X, Y and class attributes from graphs nodes.

    :param graph_file: The Pandas DataFrame containing the graph nodes.
    :type graph_file: pd.DataFrame
    :param enable_background: Enable when model has extra head to correct extra cells.
    :type enable_background: Optional[bool]
    :return: A NumPy array containing the X, Y and class attributes from the graph nodes.
    :rtype: np.ndarray

    If the graph file has a 'class' column, the function returns the 'X', 'Y' and 'class' columns as a NumPy array with integer data type.
    Otherwise, the function assumes that the graph file has a 'prob1' column, which contains the probability of a node belonging to a particular class.
    In this case, the function sets the class attribute to 1 if the probability is greater than 0.5, and to 2 otherwise.
    """

    # Return class information directly if available
    if 'class' in graph_file.columns:
        return graph_file[['X', 'Y', 'class']].to_numpy(dtype=int)

   # Binary class (and background if enabled)
    init_prob = 0 if enable_background else 1
    if num_classes == 2:
        res = graph_file[['X', 'Y', 'prob1']].to_numpy()
        class_col = (res[:, 2] > 0.5) * 1 + 1
        if enable_background:
            prob_cols = graph_file[['prob' + str(k) for k in range(num_classes + 1)]].to_numpy()
            class_col = np.argmax(prob_cols, axis=1)
        res[:, 2] = class_col

    # Multi-class (and background if enabled)
    else:
        res = graph_file[['X', 'Y', 'prob1']].to_numpy()
        prob_cols = graph_file[['prob' + str(k) for k in range(init_prob, num_classes + 1)]].to_numpy()
        class_col = np.argmax(prob_cols, axis=1) + 1*(not enable_background)
        res[:, 2] = class_col

    return np.array(res, dtype=int)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-dir', type=str, required=True, help='Path to folder containing .nodes.csv.')
    parser.add_argument('--centroids-dir', type=str, required=True, help='Path to folder where to save .centroids.csv.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')
    return parser


def main_with_args(args: Namespace) -> None:

    # Data directories
    graph_dir = parse_path(args.graph_dir)
    centroids_dir = parse_path(args.centroids_dir)

    create_dir(centroids_dir)

    # Centroids class information
    names = get_names(graph_dir, '.nodes.csv')
    for name in tqdm(names):
        graph_file = read_graph(name, graph_dir)
        centroids_file = graph2centroids(graph_file, args.num_classes, args.enable_background)
        save_centroids(centroids_file, centroids_dir, name)
    return


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
