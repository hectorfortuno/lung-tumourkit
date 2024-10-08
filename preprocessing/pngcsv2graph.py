"""
Computes a graph representation from the images and pngcsv labels.

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
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm

# import sys
# sys.path.append(r"..")
from utils.preprocessing import get_mask, read_labels_idx, parse_path, create_dir, save_graph, get_names, apply_mask, extract_features, add_node, read_image


def pngcsv2graph(img: np.ndarray, png: np.ndarray, csv: pd.DataFrame) -> pd.DataFrame:
    """
    Converts an original image and PNGCSV labels into a graph representation.

    Given the original image and PNGCSV labels, this function extracts attributes
    from each label and constructs a graph representation in the form of a DataFrame.

    The current attributes extracted for each label include:
        - X, Y coordinates of the centroid
        - Area
        - Perimeter
        - Variance
        - Regularity (not yet implemented)
        - Histogram (5 bins)

    :param img: The original image.
    :type img: np.ndarray
    :param png: The PNGCSV labels.
    :type png: np.ndarray
    :param csv: The CSV file containing label information.
    :type csv: pd.DataFrame

    :return: The graph representation in the form of a DataFrame.
    :rtype: pd.DataFrame
    """

    # Graph representation: features of cells
    graph = {}
    for idx, cls in csv.itertuples(index=False, name=None):

        # Mask of the cell, bounding box
        mask = get_mask(png, idx)
        msk_img, msk, X, Y = apply_mask(img, mask)

        # Extract features of cells
        try:
            feats = extract_features(msk_img, msk)
        except Exception:
            feats = extract_features(msk_img, msk, debug=True)

        if len(feats) > 0:
            feats['class'] = cls
            feats['id'] = idx
            feats['X'] = X
            feats['Y'] = Y
        add_node(graph, feats)

    return pd.DataFrame(graph)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--png-dir', type=str, required=True,
                        help='Path to png files.')
    parser.add_argument('--csv-dir', type=str, required=True,
                        help='Path to csv files.')
    parser.add_argument('--orig-dir', type=str, required=True,
                        help='Path to original images.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save files.')
    parser.add_argument('--num-workers', type=int, default=1)
    return parser


def main_subthread(
        name: str,
        png_dir: str,
        csv_dir: str,
        orig_dir: str,
        output_path: str,
        pbar: tqdm,
        ) -> None:
    """
    Wrapper to use multiprocessing
    """
    try:
        png, csv = read_labels_idx(name, png_dir, csv_dir)
        img = read_image(name, orig_dir)
        graph = pngcsv2graph(img, png, csv)
        save_graph(graph, os.path.join(output_path, name + '.nodes.csv'))
    except Exception as e:
        logging.warning(e)
        logging.warning('Failed at:', name)
    finally:
        pbar.update(1)


def main_with_args(args):

    # Data directories
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)
    orig_dir = parse_path(args.orig_dir)
    output_path = parse_path(args.output_path)

    create_dir(output_path)

    # PNG and CSV to graph representation (morphological and colour features of cells)
    names = get_names(png_dir, '.GT_cells.png')
    pbar = tqdm(total=len(names))
    if args.num_workers > 0:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for name in names:
                executor.submit(main_subthread, name, png_dir, csv_dir, orig_dir, output_path, pbar)
    else:
        for name in names:
            main_subthread(name, png_dir, csv_dir, orig_dir, output_path, pbar)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
