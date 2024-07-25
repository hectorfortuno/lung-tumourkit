"""
Converts the .nodes.csv files into .class.csv files, and manages the PNG files.

Copyright (C) 2024  Héctor Fortuño Martí

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

Contact information: hectorfortuno@gmail.com
"""
import argparse
import os
import shutil
from argparse import Namespace
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, get_context
import logging

# import sys
# sys.path.append(r"..")
from utils.preprocessing import get_names, create_dir, parse_path, save_csv, read_graph

def copy_png(
        name: str, 
        input_png_dir: str,
        png_dir: str):
    """
    Copies a .GT_cells.png file from the 'input_png_dir' directory to the 'png_dir' directory.
    """

    source = os.path.join(input_png_dir, name + '.GT_cells.png')
    destination = os.path.join(png_dir, name + '.GT_cells.png')
    shutil.copy(source, destination)

def graph2csv(
        graph_file: pd.DataFrame,
        num_classes: int,
        enable_background: Optional[bool] = False) -> pd.DataFrame:
    """
    Extracts id and class attributes from graphs nodes.

    :param graph_file: The Pandas DataFrame containing the graph nodes.
    :type graph_file: pd.DataFrame
    :param enable_background: Enable when model has extra head to correct extra cells.
    :type enable_background: Optional[bool]
    :return: A Pandas dataframe containing the class attributes from the graph nodes.
    :rtype: pd.DataFrame

    If the graph file has a 'class' column, the function returns the 'X', 'Y' and 'class' columns as a NumPy array with integer data type.
    Otherwise, the function assumes that the graph file has a 'prob1' column, which contains the probability of a node belonging to a particular class.
    In this case, the function sets the class attribute to 1 if the probability is greater than 0.5, and to 2 otherwise.
    """

    # Return class information directly if available
    if 'class' in graph_file.columns:
        return graph_file[['class']].to_numpy(dtype=int)

   # Binary class (and background if enabled)
    init_prob = 0 if enable_background else 1
    if num_classes == 2:
        res = graph_file[['prob1']].to_numpy()
        class_col = (res[:, 0] > 0.5) * 1 + 1
        if enable_background:
            prob_cols = graph_file[['prob' + str(k) for k in range(num_classes + 1)]].to_numpy()
            class_col = np.argmax(prob_cols, axis=1)
        res[:, 0] = class_col

    # Multi-class (and background if enabled)
    else:
        res = graph_file[['prob1']].to_numpy()
        prob_cols = graph_file[['prob' + str(k) for k in range(init_prob, num_classes + 1)]].to_numpy()
        class_col = np.argmax(prob_cols, axis=1) + 1*(not enable_background)
        res[:, 0] = class_col

    return pd.DataFrame(res, dtype=int)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-dir', type=str, required=True, help='Path to folder containing .nodes.csv.')
    parser.add_argument('--input-png-dir', type=str, required=True, help='Path to folder containing .GT_cells.png.')
    parser.add_argument('--png-dir', type=str, required=True, help='Path to folder where to save  .GT_cells.png.')
    parser.add_argument('--csv-dir', type=str, required=True, help='Path to folder where to save .class.csv.')    
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')
    return parser


def main_with_args(args: Namespace):

    # Data directories
    graph_dir = parse_path(args.graph_dir)
    input_png_dir = parse_path(args.input_png_dir)
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)

    create_dir(png_dir)
    create_dir(csv_dir)

    # Centroids class information
    names = get_names(graph_dir, '.nodes.csv')
    for name in tqdm(names):
        graph_file = read_graph(name, graph_dir)
        csv_file = graph2csv(graph_file, args.num_classes, args.enable_background)
        copy_png(name, input_png_dir, png_dir)
        save_csv(csv_file, csv_dir, name)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
