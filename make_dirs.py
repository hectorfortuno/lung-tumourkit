"""
Auxiliary script to generate folder structure.

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
import os
import argparse
import logging
from typing import Union, Dict, List
import json

from utils.preprocessing import create_dir


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes (up to 7) to consider in the classification problem.')
    return parser


def create_subfolders(node: Union[Dict, List], current_folder: str) -> None:
    """
    Creates folders and subfolders recursively.

    :param node: The folder structure represented as a dictionary or list.
    :type node: Union[Dict, List]

    :param current_folder: The current folder where the folders and subfolders will be created.
    :type current_folder: str
    """
    if isinstance(node, Dict):
        for key, value in node.items():
            subfolder = os.path.join(current_folder, key)
            os.mkdir(subfolder)
            create_subfolders(value, subfolder)
    elif isinstance(node, List):
        for value in node:
            subfolder = os.path.join(current_folder, value)
            os.mkdir(subfolder)
    else:
        assert False, 'Wrong folder structure format.'


def main():
    parser = _create_parser()
    args = parser.parse_args()
    if os.path.exists(args.root_dir):
        logging.warning('Root folder already exists, aborting.')
    else:
        structure = {
            'data': {
                'train': {
                    'png': [], 'csv': [], 'gson': [], 'json': [],
                    'graphs': ['raw', 'preds', 'GT'], 'npy': [],
                    'centroids': []
                },
                'validation': {
                    'png': [], 'csv': [], 'gson': [], 'json': [],
                    'graphs': ['raw', 'preds', 'GT'], 'npy': [],
                    'centroids': []
                },
                # 'test': {
                #     'png': [], 'csv': [], 'gson': [], 'json': [],
                #     'graphs': ['raw', 'preds', 'GT'], 'npy': [],
                #     'centroids': []
                # },
                'orig': []
            },
            'weights': {
                'segmentation': {
                    'hovernet': [],
                    # 'cellnet': ['count', 'segment']
                },
                'classification': {
                    # 'automl': [], 'xgb': [],
                    'gnn': ['confs', 'normalizers', 'weights']
                }
            }
        }
        create_dir(args.root_dir)
        create_subfolders(structure, args.root_dir)
        type_info = {
            "0": ["background", [0, 0, 0]],
            "1": ["nontumour", [255, 0, 0]],
            "2": ["tumour", [0, 255, 0]]
        }
        if args.num_classes != 2:
            colors = [
                 [0, 0, 0],
                 [0, 0, 255],
                 [255, 0, 255],
                 [0, 255, 0],
                 [255, 0, 0],
                 [0, 255, 255],
                 [0, 0, 255],
                 [255, 255, 0],
                 [255, 255, 255]
            ][:args.num_classes + 1]
            type_info = {str(k): ["Class" + str(k), v] for k, v in enumerate(colors)}
            type_info['0'] = ["background", [0, 0, 0]]
        with open(os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet', 'type_info.json'), 'w') as f:
            json.dump(type_info, f)

if __name__ == '__main__':
    main()