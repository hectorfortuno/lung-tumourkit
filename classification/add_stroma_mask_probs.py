import argparse
from argparse import Namespace
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import cv2

import torch
import torch.nn as nn

from utils.preprocessing import get_names, create_dir, parse_path, save_centroids, read_graph
from segmentation.stroma_unet.run_infer import predict_stroma
from segmentation.hovernet.misc.utils import get_bounding_box, remove_small_objects


def load_stroma_model():

        # Stroma model
        path_networks = '/home/usuaris/imatge/sonia.rabanaque/KI67/VH22/Xarxes/Version2/'
        name_model = 'stroma_model_KI67.pth'
        stroma_model = torch.load(os.path.join(path_networks, name_model), map_location=torch.device('cpu'))
        stroma_model.segmentation_head._modules['2'] = nn.Identity()
        stroma_model.to("cuda")

        return stroma_model


def update_probs(pred_inst, pred_map_stroma, graph_file, nr_types):
     
    # Predicted cells (HoverNet)
    inst_id_list = np.unique(pred_inst)[1:]
    for i, inst_id in enumerate(inst_id_list):
        inst_map = pred_inst == inst_id

        # Cell prediction
        rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
        inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
        
        # Stroma mask above current prediction
        inst_stroma_map_crop = pred_map_stroma[rmin:rmax, cmin:cmax]
        stroma_mask = (inst_stroma_map_crop >= 0.5).astype(np.uint8)
        
        inst_map_crop = (inst_map_crop == inst_id)
        inst_stroma_mask = stroma_mask[inst_map_crop]
        
        type_list, type_pixels = np.unique(inst_stroma_mask, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 1:
            graph_file.at[i, 'prob' + str(nr_types)] = 1      # stroma max. probability
        else:
            graph_file.at[i, 'prob' + str(nr_types)] = 0    

    return graph_file

    
def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-dir', type=str, required=True, help='Path to original images.')
    parser.add_argument('--graph-dir', type=str, required=True, help='Path to folder containing .nodes.csv.')
    parser.add_argument('--png-dir', type=str, required=True, help='Path to png files.')
    
    return parser


def main_with_args(args: Namespace) -> None:

    # Data directories
    orig_dir = parse_path(args.orig_dir)
    graph_dir = parse_path(args.graph_dir)
    png_dir = parse_path(args.png_dir)

    # Stroma model
    stroma_model = load_stroma_model()
    stroma_model.eval()

    # Centroids class information
    names = get_names(graph_dir, '.nodes.csv')
    for name in tqdm(names):
        orig = cv2.imread(os.path.join(orig_dir, name + '.png'), cv2.IMREAD_COLOR)[:, :, ::-1]
        pred_stroma = predict_stroma(orig, stroma_model)
        cells = cv2.imread(os.path.join(png_dir, name+'.GT_cells.png'), -1)
        graph_file = read_graph(name, graph_dir)

        graph_file = update_probs(cells, pred_stroma, graph_file, args.num_classes)

        graph_file.to_csv(graph_dir + name + '.nodes.csv', index=False)
    
    return


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)