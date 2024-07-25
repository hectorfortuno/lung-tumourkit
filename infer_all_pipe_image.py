import argparse
from argparse import Namespace
import logging
from logging import Logger
import os
import shutil
from typing import Dict
import time
import numpy as np

import torch

from preprocessing import geojson2pngcsv_main, hovernet2geojson_main, png2graph_main, graph2centroids_main, centroidspng2csv_main, pngcsv2geojson_main
from postprocessing import join_hovprob_graph_main
from segmentation import hov_infer
from classification import gnn_infer, stromamask_main
from utils.preprocessing import get_names, create_dir

from segmentation.hovernet.infer.tile import InferManager


def hovernet_postprocessing(args: Dict) -> None:
    """
    Moves Json files to their corresponding folders based on the split and converts to different formats.

    This function performs the following steps:
        1. Moves json files to their corresponding folders based on the split.
        2. Converts the json format to geojson format.
        3. Converts the geojson format to pngcsv format.

    :param args: The arguments for the post-processing pipeline.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    # JSON to GeoJSON
    newargs = Namespace(
        json_dir=args['output_dir'],
        gson_dir=args['output_dir'],
        num_classes=args['num_classes'],
    )
    hovernet2geojson_main(newargs)

    # GeoJSON to PNG and CSV
    newargs = Namespace(
        gson_dir=args['output_dir'],
        png_dir=args['output_dir'],
        csv_dir=args['output_dir'],
        num_classes=args['num_classes'],
    )
    geojson2pngcsv_main(newargs)

    return


def hovernet_infer(args: Dict, infer_hov: InferManager, path: str) -> None:
    """
    Predicts cell contours in json format using HoverNet.

    This function performs the following steps:
        1. Performs inference using the trained Hovernet model on the input images.
        2. Saves the predicted cell contours in json format.

    :param args: The arguments for the Hovernet inference pipeline.
    :type args: Namespace

    :param path: Path of the image.
    :type path: str
    """

    # Inference HoverNet
    run_args = {
        'batch_size': 8,
        'h': args["h"],
        'k': args["k"],
        
        'nr_inference_workers': 0,
        'nr_post_proc_workers': 0,
        
        'patch_input_shape': 518,
        'patch_output_shape': 328,

        'file_path': path,
        'output_dir': args['output_dir'],
    }

    infer_hov.process_file(run_args)

    return


def gnn_preprocessing(args: Dict) -> None:
    """
    Converts the input format to the graph format containing ground truth (GT) and predictions.

    This function performs the following steps:
        1. Converts the json format to geojson format (optional).
        2. Converts the geojson format to pngcsv format (optional).
        3. Converts the pngcsv format to nodes.csv format.
        4. Extracts centroids from ground truth (GT) data.
        5. Adds GT labels to the nodes.csv file.
        6. Adds Hovernet predictions to the nodes.csv file.

    :param args: The arguments for the post-processing pipeline.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """
    
    # PNG and CSV to graph representation (morphological and colour features of cells)
    newargs = Namespace(
        png_dir=args['output_dir'],
        orig_dir=os.path.join(args['output_dir'], 'Original_img'),
        output_path=args['output_dir'],
        num_workers=0,
    )
    png2graph_main(newargs)

    # Add HoverNet predictions to nodes
    newargs = Namespace(
        json_dir=args['output_dir'],
        graph_dir=args['output_dir'],
        output_dir=args['output_dir'],
        num_classes=args['num_classes'],
        enable_background=args['enable_background'],
    )
    join_hovprob_graph_main(newargs)

    return


def gnn_postprocessing(args: Dict) -> None:
    """
    Performs post-processing steps on GNN output.

    :param args: The arguments for running post-processing on GNN output.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """
    # logger.info('Parsing gnn output.')

    if args['gnn_stroma_mask']:
        args['num_classes'] = args['num_classes'] + 1
        
        newargs = Namespace(
            orig_dir=os.path.join(args['output_dir'], 'Original_img'),
            graph_dir=args['output_dir'],
            png_dir=args['output_dir'],
            num_classes=args['num_classes'],
        )
        stromamask_main(newargs)

    # Centroids class information
    newargs = Namespace(
        graph_dir=args['output_dir'],
        centroids_dir=args['output_dir'],
        num_classes=args['num_classes'],
        enable_background=args['enable_background'],
    )
    graph2centroids_main(newargs)

    # Centroids to PNG and CSV
    newargs = Namespace(
        centroids_dir=args['output_dir'],
        input_png_dir=args['output_dir'],
        png_dir=args['output_dir'],
        csv_dir=args['output_dir'],
    )
    centroidspng2csv_main(newargs)

    # PNG and CSV to GeoJSON
    newargs = Namespace(
        png_dir=args['output_dir'],
        csv_dir=args['output_dir'],
        gson_dir=args['output_dir'],
        num_classes=args['num_classes'],
    )
    pngcsv2geojson_main(newargs)

    return


def main():

    # Variables
    root_dir = r'/home/usuaris/imatge/sonia.rabanaque/TFM/Mama_nuclear/RE/'
    
    model_path = os.path.join(root_dir, 'weights')
    type_info_path = os.path.join(root_dir, 'type_info.json')

    images_dir = os.path.join(root_dir, 'data', 'orig')
    tmp_dir = os.path.join(root_dir, 'data', 'tmp_inference')
    create_dir(os.path.join(tmp_dir, 'Original_img'))
    result_dir = os.path.join(root_dir, 'Final_results')
    create_dir(result_dir)

    # Arguments # TODO: valors
    args = {
        'num_classes': 5,
        'hov_stroma_mask': False,
        'gnn_stroma_mask': False,
        'gpu': '0',
        'h': 0.5,
        'k': 0.4,
        'type_info_path': type_info_path,
        'output_dir': tmp_dir,
        'enable_background': True,
    }

    # Arguments
    hov_args = {'nr_gpus': torch.cuda.device_count(),
                'method' : {'model_args': {'nr_types': (args['num_classes'] + 1) if not args['hov_stroma_mask'] else args['num_classes'],
                                            'mode'   : 'original',
                                            },
                            'model_path': os.path.join(model_path, 'segmentation', 'hovernet_label_smoothing', '01', 'net_epoch=50.tar'),
                            'stroma_mask': args['hov_stroma_mask'],
                            },
                'type_info_path': args['type_info_path'],
                'mem_usage': 0.2,
                }
    
    # TODO: PATHS MODELS, VARIABLES
    gnn_args = Namespace(node_dir=args['output_dir'],
                         output_dir=args['output_dir'],
                         
                         weights=os.path.join(model_path, 'classification', 'gnn_2nd_version_k=50_d=300', 'weights', 'best_TRANSFORMER_5_0.0_None.pth'),
                         conf=os.path.join(model_path, 'classification', 'gnn_2nd_version_k=50_d=300', 'confs', 'best_TRANSFORMER_5_0.0_None.json'),
                         normalizers=os.path.join(model_path, 'classification', 'gnn_2nd_version_k=50_d=300', 'normalizers', 'best_TRANSFORMER_5_0.0_None.pkl'),

                         num_classes=args['num_classes'],
                         degree=50,
                         distance=300,

                         disable_prior=False,
                         disable_morph_feats=False,
                         disable_coordinates=True,
                         enable_background=args['enable_background'],
                         stroma_mask=args['gnn_stroma_mask'],
    )
    
    # Initialization
    infer_hov = InferManager(**hov_args)

    # Inference images in directory
    images_names = [image_name[:-4] for image_name in os.listdir(images_dir)]
    for image_name in images_names:

        image_path = os.path.join(images_dir, image_name+'.png')

        # Copy image
        shutil.copyfile(image_path, os.path.join(tmp_dir, 'Original_img', 'img.png'))

        ##################################################
        # HoVer-Net
        ##################################################

        # HoverNet inference
        hovernet_infer(args, infer_hov, image_path)

        # HoverNet post-processing
        hovernet_postprocessing(args)

        ##################################################
        # GNN
        ##################################################

        # GNN pre-processing
        gnn_preprocessing(args)

        # GNN inference
        gnn_infer(gnn_args)

        # GNN post-processing
        gnn_postprocessing(args)

        # Copy result
        shutil.copyfile(os.path.join(tmp_dir, 'img.GT_cells.png'), os.path.join(result_dir, image_name+'.GT_cells.png'))
        shutil.copyfile(os.path.join(tmp_dir, 'img.class.csv'), os.path.join(result_dir, image_name+'.class.csv'))


if __name__ == '__main__':
    main()