import argparse
import os
import numpy as np
import pandas as pd

import sys
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import read_csv, get_names, create_dir


def get_features_modified_cells_image(image_name, csv_dir_hov, csv_dir_gnn, features_dir, save_features_path):

    # Read classes HoverNet
    csv_path_hov = os.path.join(csv_dir_hov, image_name+'.class.csv')
    classes_hov = read_csv(image_name, csv_dir_hov)
    classes_hov = classes_hov['label'].values

    # Read classes GNN
    csv_path_gnn = os.path.join(csv_dir_gnn, image_name + '.class.csv')
    classes_gnn = read_csv(image_name, csv_dir_gnn)
    classes_gnn = classes_gnn['label'].values

    # Read features of cells
    features_path = os.path.join(features_dir, image_name+'.nodes.csv')
    features = pd.read_csv(features_path)

    # Check differences in classifications
    idxs = np.where(classes_hov != classes_gnn)[0]
    # idxs = classes_hov['id'].iloc(idxs)

    # Read features of modified cells
    features_modified = features.iloc[idxs]

    # Save features in a file
    features_modified.to_csv(os.path.join(save_features_path, image_name + '.features.csv'), index=False, header=False)

    return features_modified


def get_features_modified_cells(args):

    # Paths
    csv_dir_hov = os.path.join(args.root_dir, 'data', 'validation', 'csv_hov' + args.data_suffix_hov)
    csv_dir_gnn = os.path.join(args.root_dir, 'data', 'validation', 'csv_gnn' + args.data_suffix_gnn)

    features_dir = os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds')
    save_features_path = os.path.join(args.root_dir, 'data', 'validation', 'Analysis', 'features_cells_modified_class')
    create_dir(save_features_path)

    # Images
    names = get_names(csv_dir_hov, '.class.csv')
    features_all = None
    for i, name in enumerate(names):
        features_image = get_features_modified_cells_image(name, csv_dir_hov, csv_dir_gnn, features_dir, save_features_path)

        if i == 0:
            features_all = features_image
        else:
            features_all = pd.concat([features_all, features_image])

    # Save global csv
    features_all.to_csv(os.path.join(save_features_path, 'global.features.csv'), index=False, header=False)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--data-suffix-hov', type=str, default="", help="Suffix of the folders to load the results of HoverNet.")
    parser.add_argument('--data-suffix-gnn', type=str, default="", help="Suffix of the folders to load the results of GNN.")

    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')

    return parser


def main():

    # Arguments
    parser = _create_parser()
    args = parser.parse_args()
    if args.data_suffix_hov != "":
        args.data_suffix_hov = '_' + args.data_suffix_hov
    if args.data_suffix_gnn != "":
        args.data_suffix_gnn = '_' + args.data_suffix_gnn

    # Features of the modified cells
    get_features_modified_cells(args)


if __name__ == '__main__':
    main()