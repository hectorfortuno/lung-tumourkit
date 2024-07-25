"""
Script to generate predictions from given GNN model.

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
from typing import Dict, Tuple, Any, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import dgl
from dgl.dataloading import GraphDataLoader
from torch_geometric.loader import DataLoader
import json
import pickle
import numpy as np
import pandas as pd
import argparse
import os
import cv2
import copy

from .train_graphs import load_model
from .read_graph import GraphDataset
# from ..segmentation.stroma_unet.run_infer import predict_stroma

# import sys
# sys.path.append(r"../")
from utils.preprocessing import parse_path, create_dir


def load_saved_model(weights_path: str, conf_path: str, num_classes: int, num_feats: int, enable_background: bool) -> nn.Module:
    """
    Loads a saved model from the given weights_path and conf_path.

    :param weights_path: The path to the saved weights file.
    :type weights_path: str
    :param conf_path: The path to the configuration file.
    :type conf_path: str
    :param num_classes: The number of classes for the model.
    :type num_classes: int
    :param num_feats: The number of features for the model.
    :type num_feats: int
    :param enable_background: Enable when model has extra head to correct extra cells.
    :type enable_background: bool
    :return: The loaded model.
    :rtype: nn.Module
    """
    state_dict = torch.load(weights_path, map_location='cpu')
    with open(conf_path, 'r') as f:
        conf = json.load(f)
    model = load_model(conf, num_classes, num_feats, enable_background)
    model.load_state_dict(state_dict)
    return model

def load_stroma_model(self):

        # Stroma model
        path_networks = '/home/usuaris/imatge/sonia.rabanaque/KI67/VH22/Xarxes/Version2/'
        name_model = 'stroma_model_KI67.pth'
        stroma_model = torch.load(os.path.join(path_networks, name_model), map_location=torch.device('cpu'))
        stroma_model.segmentation_head._modules['2'] = nn.Identity()
        stroma_model.to("cuda")

        return stroma_model


def load_normalizer(norm_path: str) -> Tuple[Any]:
    """
    Loads the normalizers used in training from the given norm_path.

    :param norm_path: The path to the saved normalizers file.
    :type norm_path: str
    :return: The loaded normalizers.
    :rtype: Tuple[Any]
    """
    with open(norm_path, 'rb') as f:
        normalizers = pickle.load(f)
    return normalizers


def run_inference(
        model: nn.Module,
        loader: GraphDataLoader,
        device: str,
        num_classes: int,
        enable_background: bool,
        # stroma_model,
        # root_dir: str,
        ) -> Dict[str, np.ndarray]:
    """
    Runs inference using the specified model on the provided data loader.

    :param model: The model used for inference.
    :type model: nn.Module
    :param loader: The graph data loader.
    :type loader: GraphDataLoader
    :param device: The device used for inference (e.g., 'cpu' or 'cuda').
    :type device: str
    :param num_classes: The number of classes.
    :type num_classes: int
    :param enable_background: Enable when model has extra head to correct extra cells.
    :type enable_background: bool

    :return: The probabilities for all the nodes.
    :rtype: Dict[str, np.ndarray]
    """

    return_attention_weights = None
    if return_attention_weights is not None:
        save_attention_path = r"/home/usuaris/imatge/sonia.rabanaque/TFM/Mama_nuclear/RE/Attention_weights/"
        create_dir(save_attention_path)

    # Run inference
    probs = {}
    for g, name in loader:
        g = g.to(device)

        # Self-loops
        # g = dgl.remove_self_loop(g)
        # g = dgl.add_self_loop(g)

        # Data
        features = g.x

        # Forward pass
        if return_attention_weights is None:
            logits = model(features, g.edge_index)
        
        else:
            logits = model(features, g.edge_index, return_attention_weights=return_attention_weights)
            logits, attention_edges, attention_weights = logits

            # Save attention information
            for i, (att_edges, att_weights) in enumerate(zip(attention_edges, attention_weights)):
                save_attention_image = os.path.join(save_attention_path, name[0][:-10]+'_'+str(i))
                att_edges = att_edges.detach().numpy()
                att_weights = att_weights.detach().numpy()
                np.savez_compressed(save_attention_image, edges=att_edges, weights=att_weights)

        # Predictions
        if num_classes == 2:
            prob = F.softmax(logits, dim=1).detach().numpy()[:, 1].reshape(-1, 1)
        else:
            prob = F.softmax(logits, dim=1).detach().numpy()

        """
        # Stroma prediction
        if stroma_model is not None:
            img_path = os.path.join(root_dir, 'data', 'orig')
            src_image = cv2.imread(os.path.join(img_path, name+'.png'), cv2.IMREAD_COLOR)[:, :, ::-1]
            pred_stroma = predict_stroma(src_image, stroma_model)
        """

        probs[name[0]] = prob

    return probs


def save_probs(
        probs: Dict[str, np.ndarray],
        node_dir: str,
        output_dir: str,
        num_classes: int,
        enable_background: bool,
        stroma_mask: bool,
        ) -> None:
    """
    Saves the probabilities in .nodes.csv files by appending a column to the original .nodes.csv file.

    :param probs: The probabilities for each graph.
    :type probs: Dict[str, np.ndarray]
    :param node_dir: The directory containing the original .nodes.csv files.
    :type node_dir: str
    :param output_dir: The directory where the updated .nodes.csv files will be saved.
    :type output_dir: str
    :param enable_background: Enable when model has extra head to correct extra cells.
    :type enable_background: bool
    :param num_classes: The number of classes.
    :type num_classes: int
    """

    # Save probabilities
    for name, prob in probs.items():

        # Write probabilities in file
        orig = pd.read_csv(os.path.join(node_dir, name))
        if enable_background:
            orig['prob0'] = prob[:, 0]
            prob = prob[:, 1:]
        if num_classes == 2 and not stroma_mask:
            orig['prob1'] = prob
        else:
            for k in range(1, num_classes):
                orig['prob' + str(k)] = prob[:, (k - 1)]
        orig.to_csv(output_dir + name, index=False)


def _create_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--node-dir', type=str, required=True, help='Folder containing .nodes.csv')
    parser.add_argument('--output-dir', type=str, required=True, help='Folder to save .nodes.csv containing probabilities.')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights.')
    parser.add_argument('--conf', type=str, required=True, help='Configuration file for the model.')
    parser.add_argument('--normalizers', type=str, required=True, help='Path to normalizer objects for the model.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--disable-prior', action='store_true', help='If True, remove hovernet probabilities from node features.')
    parser.add_argument('--disable-area', action='store_true', help='If True, remove area feature from node features.')
    parser.add_argument('--disable-perimeter', action='store_true', help='If True, remove perimeter feature from node features.')
    parser.add_argument('--disable-std', action='store_true', help='If True, remove std feature from node features.')
    parser.add_argument('--disable-hist', action='store_true', help='If True, remove histogram features from node features.')
    parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')
    return parser


def main_with_args(args):

    # Data directories
    node_dir = parse_path(args.node_dir)
    output_dir = parse_path(args.output_dir)

    if args.degree is None:
        args.degree = 15
    if args.distance is None:
        args.distance = 200

    create_dir(output_dir)

    # Classes to consider
    num_classes = copy.copy(args.num_classes)
    if args.enable_background:
        num_classes += 1

    # Data normalizers
    normalizers = load_normalizer(args.normalizers)

    # Dataset
    eval_dataset = GraphDataset(
        node_dir=node_dir, return_names=True, is_inference=True,
        max_dist=args.distance, max_degree=args.degree, normalizers=normalizers,
        remove_area=args.disable_area, remove_perimeter=args.disable_perimeter, remove_std=args.disable_std, 
        remove_hist=args.disable_hist, remove_prior=args.disable_prior, remove_coords=args.disable_coordinates)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Features to be used
    num_feats = 20 + (1 if num_classes == 2 else num_classes)
    if args.disable_coordinates:
        num_feats -= 2
    if args.disable_area:
        num_feats -= 1
    if args.disable_perimeter:
        num_feats -= 1
    if args.disable_std:
        num_feats -= 1
    if args.disable_hist:
        num_feats -= 15
    if args.disable_prior:
        num_feats -= (1 if num_classes == 2 else num_classes)
    if num_feats == 0:
        num_feats = 1

    # Model in inference mode
    model = load_saved_model(args.weights, args.conf, num_classes, num_feats, args.enable_background)
    model.eval()

    # Stroma model and images dataset
    """
    stroma_model = None
    if args.stroma_mask:

        # Model
        stroma_model = load_stroma_model()
        stroma_model.eval()        
    """

    # Inference
    probs = run_inference(model, eval_dataloader, 'cpu', num_classes, args.enable_background)
    save_probs(probs, node_dir, output_dir, num_classes, args.enable_background, args.stroma_mask)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main(args)
