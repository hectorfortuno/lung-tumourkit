"""
Computes centroids csv from png <-> csv probs.

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
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Tuple

# import sys
# sys.path.append(r"..")
from utils.preprocessing import read_png, read_csv_prob, parse_path, create_dir, get_names, save_centroids, get_centroid_by_id

def extract_centroids(img: np.ndarray, csv: pd.DataFrame) -> List[Tuple[int, int, int]]:
    """
    Extracts the centroids of cells from a labeled image. The third coordinate is the class.

    :param img: A 2D NumPy array representing the labeled image. Each unique non-zero value represents a different cell.
    :type img: np.ndarray
    :param csv: A pandas DataFrame representing the labels. Just two colums, one for probability of class 0 and another for the one of class 1.
    :type csv: pd.DataFrame

    :return: A list of tuples containing the x and y coordinates of the centroid, and the cell class.
    :rtype: List[Tuple[int,int,int]]
    """
    centroids = []
    for i, row in csv.iterrows():
        x, y = get_centroid_by_id(img, i+1)
        if x == -1:
            continue
        prob2 = 1 - row.prob0 - row.prob1 
        centroids.append((x, y, np.argmax([row.prob0, row.prob1, prob2])))
    return centroids


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--png-dir', type=str, required=True, help='Path to png files.')
    parser.add_argument('--csv-dir', type=str, required=True, help='Path to csv files.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save files.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')

    return parser


def main_with_args(args):

    # Data directories
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)
    output_path = parse_path(args.output_path)

    create_dir(output_path)

    # PNG and CSV to centroids
    names = get_names(png_dir, '.GT_cells.png')
    for name in tqdm(names):
        img = read_png(name, png_dir)
        csv = read_csv_prob(name, csv_dir, args.num_classes)
        centroids = extract_centroids(img, csv)
        save_centroids(centroids, output_path, name)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
