"""
Computes a graph representation from the images and pngcsv labels.

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
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, get_context
from typing import Dict, Any, List, Tuple, Optional
import logging
from tqdm import tqdm

# import sys
# sys.path.append(r"..")
from utils.preprocessing import read_png, save_graph, parse_path, create_dir, get_names, get_mask, apply_mask, add_node, extract_features, read_image, read_csv_prob


def pngprob2graph(
        img: np.ndarray, 
        png: np.ndarray, 
        prob: pd.DataFrame,
        num_classes: Optional[int] = 2, 
        enable_background: Optional[bool] = False,
        train: Optional[bool] = False,
        ) -> pd.DataFrame:
    """
    Given an original image and a segmentation mask in PNG format, this function extracts the nodes and their attributes
    and returns them in a pandas DataFrame. The following attributes are computed for each node:

    * X: The X-coordinate of the centroid.
    * Y: The Y-coordinate of the centroid.
    * Area: The area of the cell.
    * Perimeter: The perimeter of the cell.
    * Variance: The variance of the grayscale values inside the cell.
    * Histogram: The normalized histogram of the grayscale values inside the cell (5 bins).
    * Probabilites: Predicted probability by HoverNet for each class.

    :param img: The original image as a numpy array.
    :type img: np.ndarray
    :param png: The segmentation mask in PNG format as a numpy array.
    :type png: np.ndarray
    :param prob: The instance predicted probabilities in CSV format as a Pandas Dataframe.
    :type prob: pd.DataFrame
    :param num_classes: Number of classes to consider for classification (background not included).
    :type num_classes: int
    :param enable_background: If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.
    :type enable_background: bool
    :param train: Whether the creation of the graph is for training or not.
    :type train: bool
    :return: A pandas DataFrame containing the extracted nodes and their attributes.
    :rtype: pd.DataFrame
    """
    if enable_background:
        num_classes += 1
    
    graph = {}
    for idx in np.unique(png):
        if idx == 0:
            continue

        mask = get_mask(png, idx)
        msk_img, msk, X, Y = apply_mask(img, mask)

        try:
            feats = extract_features(msk_img, msk)
        except Exception:
            feats = extract_features(msk_img, msk, debug=True)
        if len(feats) > 0:
            feats['id'] = idx
            feats['X'] = X
            feats['Y'] = Y

        sum_probs = 0
        probabilities = prob.iloc[idx-1]
        probs = []
        for cls in range(num_classes-1):
            feats['prob' + str(cls)] = probabilities['prob' + str(cls)]
            probs.append(probabilities['prob' + str(cls)])
            sum_probs += probabilities['prob' + str(cls)]
        feats['prob' + str(num_classes-1)] = 1 - sum_probs
        probs.append(feats['prob' + str(num_classes-1)])
        if train:
            feats['class'] = np.argmax(probs)

        add_node(graph, feats)

    return pd.DataFrame(graph)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--png-dir', type=str, required=True,
                        help='Path to png files.')
    parser.add_argument('--csv-dir', type=str, required=True,
                        help='Path to csv file.')
    parser.add_argument('--orig-dir', type=str, required=True,
                        help='Path to original images.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save files.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--train', type=bool, default=False, help='Whether the creation of the graph is for training or not.')
    return parser


def process_file(args):
    """
    A wrapper function to use multiprocessing.
    
    :param name: The name of the image file (without extension).
    :type name: str
    :param png_dir: The directory path containing PNG segmentation masks.
    :type png_dir: str
    param csv_dir: The directory path containing .prob.csv files.
    :type png_dir: str
    :param orig_dir: The directory path containing original RGB images.
    :type orig_dir: str
    :param output_path: The directory path to save output files.
    :type output_path: str
    :param num_classes: Number of classes to consider for classification (background not included).
    :type num_classes: int
    :param enable_background: If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.
    :type enable_background: bool
    :param train: Whether the creation of the graph is for training or not.
    :type train: bool
    """
    name, png_dir, csv_dir, orig_dir, output_path, num_classes, enable_background, train = args
    try:
        png = read_png(name, png_dir)
        prob = read_csv_prob(name, csv_dir, num_classes)
        img = read_image(name, orig_dir)
        graph = pngprob2graph(img, png, prob, num_classes, enable_background, train)
        save_graph(graph, os.path.join(output_path, name + '.nodes.csv'))
    except Exception as e:
        logging.warning(e)
        logging.warning(f'Failed at: {name}')


def main_with_args(args):

    # Data directories
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)
    orig_dir = parse_path(args.orig_dir)
    output_path = parse_path(args.output_path)

    create_dir(output_path)

    # PNG to graph representation
    names = get_names(png_dir, '.GT_cells.png')
    num_workers = min(args.num_workers, len(names))
    if num_workers > 1:
        task_args = [(name, png_dir, csv_dir, orig_dir, output_path, args.num_classes, args.enable_background, args.train) for name in names]
        with tqdm(total=len(names)) as pbar:
            with get_context('spawn').Pool(processes=num_workers) as pool:
                for _ in pool.imap_unordered(process_file, task_args):
                    pbar.update(1)
    else:
        for name in tqdm(names):
            png = read_png(name, png_dir)
            prob = read_csv_prob(name, csv_dir, args.num_classes)
            img = read_image(name, orig_dir)
            graph = pngprob2graph(img, png, prob, args.num_classes, args.enable_background, args.train)
            save_graph(graph, os.path.join(output_path, name + '.nodes.csv'))

def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
