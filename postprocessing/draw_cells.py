"""
Draws the cells into an image.
Input format: PNG / CSV
Output format: PNG

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
from tqdm import tqdm
import os
import cv2
import numpy as np
import pandas as pd
import json
from typing import Dict, Tuple, List

from skimage.measure import label, regionprops, regionprops_table, find_contours

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# sys.path.append(r"../")

import sys
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import get_names, read_labels, read_labels_idx, read_csv_prob, read_png


def draw_cells_contours(orig, png, csv, type_info, line_thickness=2):

    overlay = np.copy((orig))

    # Draw contours of each cell on top of the image
    for i, row in csv.iterrows():
        if row['label'] != 0:
            inst_contour = cv2.findContours((png==(i+1)).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            inst_colour = type_info[str(row['label'])][1]
            cv2.drawContours(overlay, inst_contour[0], -1, inst_colour, line_thickness)

    return overlay

def draw_cells_contours_prob(orig, png, csv, type_info, line_thickness=2):

    overlay = np.copy((orig))

    # Draw contours of each cell on top of the image
    for i, (prob0, prob1) in csv.iterrows():
        probs = [prob0, prob1, 1 - prob0 - prob1]
        cell_label = probs.index(max(probs))
        inst_contour = cv2.findContours((png==(i+1)).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        inst_colour = type_info[str(cell_label)][1]
        cv2.drawContours(overlay, inst_contour[0], -1, inst_colour, line_thickness)

    return overlay

def draw_cells(
        orig: np.ndarray,
        png: np.ndarray,
        csv: pd.DataFrame,
        type_info: Dict[str, Tuple[str, List[int]]]
        ) -> np.ndarray:
    """
    Draws cell labels on the original image based on the PNG labels and CSV data.

    Given the original image, PNG labels, CSV data containing cell labels,
    and type information, this function draws the cell labels on the original image.
    The cell labels are blended with the original image by alpha compositing.

    :param orig: The original image.
    :type orig: np.ndarray
    :param png: The PNG labels representing cell IDs.
    :type png: np.ndarray
    :param csv: The CSV data containing cell labels.
    :type csv: pd.DataFrame
    :param type_info: The type information dictionary mapping cell labels to colors.
    :type type_info: Dict[str, Tuple[str, List[int]]]

    :return: The image with cell labels drawn on it.
    :rtype: np.ndarray
    """
    blend = orig.copy()
    for i, (idx, cell_label) in csv.iterrows():
        blend[png == idx] = 0.3 * np.array(type_info[str(cell_label)][1]) + 0.7 * blend[png == idx]
    return blend


def visualize_overlay(overlay, type_info, output_dir, name):

    plt.figure(figsize=(15, 15))

    plt.imshow(overlay); plt.axis("off")

    colors = [np.array(rgb_tuple) / 255 for _, rgb_tuple in type_info.values()]
    labels = [name for name, _ in type_info.values()]
    #labels[-1] = "Non-epithelial"

    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(1, len(labels))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, fontsize=12)

    plt.savefig(os.path.join(output_dir, name + '_contours.png'), bbox_inches='tight')
    plt.close()

def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-dir', type=str, help='Path to base images folder. Must be in .png format.')
    parser.add_argument('--png-dir', type=str, help='Path to folder with png of the labels.')
    parser.add_argument('--csv-dir', type=str, help='Path to folder with csv of the labels.')
    parser.add_argument('--output-dir', type=str, help='Path to folder where to save results.')
    parser.add_argument('--type-info', type=str, help='Path to type_info.json.')
    return parser


def main_with_args(args: Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    names = get_names(args.png_dir, '.GT_cells.png')
    with open(args.type_info, 'r') as f:
        type_info = json.load(f)
    for name in tqdm(names):
        orig = cv2.imread(os.path.join(args.orig_dir, name + '.png'), cv2.IMREAD_COLOR)[:, :, ::-1]
        png, csv = read_labels(name, args.png_dir, args.csv_dir)
        out = draw_cells_contours(orig, png, csv, type_info)
        """png = read_png(name, args.png_dir)
        csv = read_csv_prob(name, args.csv_dir, 3)
        out = draw_cells_contours_prob(orig, png, csv, type_info)"""
        # cv2.imwrite(os.path.join(args.output_dir, name + '_contours.png'), out)
        visualize_overlay(out, type_info, args.output_dir, name)

    return

def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)

if __name__ == '__main__':
    main()