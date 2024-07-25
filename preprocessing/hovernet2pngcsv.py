"""
Converts from HoVernet json to PNG and CSV.

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
from typing import Dict, Any, Tuple, Optional
import argparse
from skimage.draw import polygon
from tqdm import tqdm
from multiprocessing import Pool, get_context
import pandas as pd
import numpy as np
import logging

# import sys
# sys.path.append(r"..")
from utils.preprocessing import parse_path, get_names, create_dir, read_json, save_pngcsv, save_csv


def hovernet2pngcsv(nuc: Dict[str, Any], num_classes: Optional[int] = 2) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Parses contours of cells from the given HoverNet JSON dictionary and returns the PNG and class information in both .class.csv and .prob.csv.

    :param nuc: A dictionary containing HoverNet nuclei information.
    :type nuc: Dict[str, Any]
    :param num_classes: The number of classes (default 2).
    :type num_classes: Optional[int]
    :return: The PNG, .class.csv, and .prob.csv files characterizing the segmentation and classification of Hover-Net.
    :rtype: Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]

    Mostly used when the image with coloured cells is wanted, otherwise use hovernet2pngprob.
    """

    # Initialize variables
    png = np.zeros((1024, 1024), dtype=np.uint16)
    if num_classes == 2:
        prob_csv = pd.DataFrame([], columns=['prob0', 'prob1'])
        class_csv = pd.DataFrame([], columns=['label'])
    for k, inst in enumerate(nuc):
        inst_info = nuc[inst]

        # PNG
        contour = inst_info['contour']
        poly = np.array(contour)
        rr, cc = polygon(poly[:, 0], poly[:, 1], png.shape)
        png[cc, rr] = k + 1

        # Probabilities prediction
        if num_classes == 2:
            prob0 = float(inst_info['prob0'])
            prob1 = float(inst_info['prob1'])
            prob_csv = pd.concat([prob_csv, pd.DataFrame([[prob0, prob1]], columns=['prob0', 'prob1'])])
            label = np.argmax([prob0, prob1, 1 - prob0 - prob1])
            class_csv = pd.concat([class_csv, pd.DataFrame([[label]], columns=['label'])])

    return png, prob_csv, class_csv


def hovernet2pngprob(nuc: Dict[str, Any], num_classes: Optional[int] = 2) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Parses contours of cells from the given HoverNet JSON dictionary and returns the PNG and class information in .prob.csv.

    :param nuc: A dictionary containing HoverNet nuclei information.
    :type nuc: Dict[str, Any]
    :param num_classes: The number of classes (default 2).
    :type num_classes: Optional[int]
    :return: The PNG and .prob.csv files characterizing the segmentation and classification of Hover-Net.
    :rtype: Tuple[np.ndarray, pd.DataFrame]
    """

    # Initialize variables
    png = np.zeros((1024, 1024), dtype=np.uint16)
    if num_classes == 2:
        prob_csv = pd.DataFrame([], columns=['prob0', 'prob1'])

    for k, inst in enumerate(nuc):
        inst_info = nuc[inst]

        # PNG
        contour = inst_info['contour']
        poly = np.array(contour)
        rr, cc = polygon(poly[:, 0], poly[:, 1], png.shape)
        png[cc, rr] = k + 1

        # Probabilities prediction
        if num_classes == 2:
            prob0 = float(inst_info['prob0'])
            prob1 = float(inst_info['prob1'])
            prob_csv = pd.concat([prob_csv, pd.DataFrame([[prob0, prob1]], columns=['prob0', 'prob1'])])

    return png, prob_csv


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-dir', type=str, default='./',
                        help='Path to json files.')
    parser.add_argument('--png-dir', type=str, required=True,
                        help='Path to save the png files.')
    parser.add_argument('--csv-dir', type=str, required=True,
                        help='Path to save the csv files.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--num-workers', type=int, default=0),
    parser.add_argument('--class-csv', type=bool, default=False, help='Whether to save a .class.csv for image with coloured cells visualization.')
    return parser


def process_file(args):
    """
    A wrapper function to use multiprocessing.

    :param name: The name of the image file (without extension).
    :type name: str
    :param json_dir: The directory path containing the Hover-Net JSON output files.
    :type json_dir: str
    :param png_dir: The directory path containing PNG segmentation masks.
    :type png_dir: str
    param csv_dir: The directory path containing .prob.csv files.
    :type png_dir: str
    :param num_classes: Number of classes to consider for classification (background not included).
    :type num_classes: int
    :param obtain_class_csv: Whether to save a .class.csv for image with coloured cells visualization.
    :type obtain_class_csv: bool
    """
    name, json_dir, png_dir, csv_dir, num_classes, obtain_class_csv = args
    try:
        nuc = read_json(json_dir + name + '.json')
        if obtain_class_csv:
            png, prob_csv, class_csv = hovernet2pngcsv(nuc, num_classes)
            save_csv(class_csv, csv_dir, name)
        else:
            png, prob_csv = hovernet2pngprob(nuc, num_classes)
        save_pngcsv(png, prob_csv, png_dir, csv_dir, name)
    except Exception as e:
        logging.warning(e)
        logging.warning(f'Failed at: {name}')


def main_with_args(args):

    # Data directories
    json_dir = parse_path(args.json_dir)
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)

    create_dir(png_dir)
    create_dir(csv_dir)

    # JSON to PNG, .class.csv and .prob.csv
    names = get_names(json_dir, '.json') 
    num_workers = min(args.num_workers, len(names))
    if num_workers > 1:
        task_args = [(name, json_dir, png_dir, csv_dir, args.num_classes, args.class_csv) for name in names]
        with tqdm(total=len(names)) as pbar:
            with get_context('spawn').Pool(processes=num_workers) as pool:
                for _ in pool.imap_unordered(process_file, task_args):
                    pbar.update(1)
    else:
        for name in tqdm(names):
            nuc = read_json(json_dir + name + '.json')
            if args.class_csv:
                png, prob_csv, class_csv = hovernet2pngcsv(nuc, args.num_classes)
                save_csv(class_csv, csv_dir, name)
            else:
                png, prob_csv = hovernet2pngprob(nuc, args.num_classes)
            save_pngcsv(png, prob_csv, png_dir, csv_dir, name)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)