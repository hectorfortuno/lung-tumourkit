"""
Training pipeline.

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
import logging
from logging import Logger
import os
import shutil

from preprocessing import geojson2pngcsv_main, pngcsv2graph_main, hovernet2geojson_main, pngcsv2centroids_main, pngcsv2geojson_main
from segmentation import pngcsv2npy, hov_train, hov_infer
from postprocessing import join_graph_gt_main, join_hovprob_graph_main
from classification import train_gnn
from utils.preprocessing import get_names

def run_preproc_pipe(args: Namespace, logger: Logger) -> None:
    """
    Runs the preprocessing pipeline to convert the gson or pngcsv format to other formats.

    This function performs the following steps:
        1. Converts the geojson files to pngcsv format, or the pngcsv format to geojson files.
        2. Converts the pngcsv files to npy format.

    :param args: The arguments for the preprocessing pipeline.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    for split in ['train', 'validation']: # , 'test']:
        logger.info(f'Parsing {split} split')

        # GeoJSON to PNG and CSV or PNG and CSV to GeoJSON
        if args.format == 'geojson':
            newargs = Namespace(
                gson_dir=os.path.join(args.root_dir, 'data', split, 'gson'),
                png_dir=os.path.join(args.root_dir, 'data', split, 'png'),
                csv_dir=os.path.join(args.root_dir, 'data', split, 'csv'),
                num_classes=args.num_classes,
            )
            geojson2pngcsv_main(newargs)
        else:
            newargs = Namespace(
                png_dir=os.path.join(args.root_dir, 'data', split, 'png'),
                csv_dir=os.path.join(args.root_dir, 'data', split, 'csv'),
                gson_dir=os.path.join(args.root_dir, 'data', split, 'gson'),
                num_classes=args.num_classes,
            )
            pngcsv2geojson_main(newargs)

        # PNG and CSV to NPY
        newargs = Namespace(
            orig_dir=os.path.join(args.root_dir, 'data', 'orig'),
            png_dir=os.path.join(args.root_dir, 'data', split, 'png'),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv'),
            out_dir=os.path.join(args.root_dir, 'data', split, 'npy'),
            save_example=False, use_labels=True, split=False,
            shape='518'
        )
        pngcsv2npy(newargs)

    return


def run_hov_pipe(args: Namespace, logger: Logger) -> None:
    """
    Trains Hovernet and predicts cell contours in json format.

    This function performs the following steps:
        1. Trains Hovernet on the training data.
        2. Performs inference using the trained Hovernet model on the input images.
        3. Saves the predicted cell contours in json format.

    :param args: The arguments for the Hovernet pipeline.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    """
    # Train HoverNet
    logger.info('Starting training.')
    newargs = Namespace(
        gpu=args.gpu, view=None, save_name=None,
        log_dir=os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet'),
        train_dir=os.path.join(args.root_dir, 'data', 'train', 'npy'),
        valid_dir=os.path.join(args.root_dir, 'data', 'validation', 'npy'),
        pretrained_path=args.pretrained_path,
        shape='518',
        num_classes=args.num_classes if not args.stroma_mask else args.num_classes-1,
        stroma_mask=args.stroma_mask
    )
    hov_train(newargs)
    """
    
    # Inference HoverNet
    logger.info('Starting inference.')
    newargs = {
        'nr_types': str(args.num_classes + 1) if not args.stroma_mask else str(args.num_classes),
        'type_info_path': os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet', 'type_info.json'),
        'gpu': args.gpu,
        'nr_inference_workers': '0',
        'model_path': os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet', '01', 'net_epoch=50.tar'),
        'batch_size': '10',
        'shape': '518',
        'nr_post_proc_workers': '0',
        'model_mode': 'original',
        'stroma_mask': args.stroma_mask,
        # 'stroma_model_path': ...,
        'h': args.h,
        'k': args.k,
        'help': False
    }
    newsubargs = {
        'input_dir': os.path.join(args.root_dir, 'data', 'orig'),
        'output_dir': os.path.join(args.root_dir, 'data', 'tmp_hov'),
        'draw_dot': False,
        'save_qupath': False,
        'save_raw_map': False,
        'mem_usage': '0.2'
    }
    hov_infer(newargs, newsubargs, 'tile')

    return


def run_postproc_pipe(args: Namespace, logger: Logger) -> None:
    """
    Converts the json format to the graph format containing ground truth (GT) and predictions.

    This function performs the following steps:
        1. Moves json files to their corresponding folders based on the split.
        2. Converts the json format to geojson format.
        3. Converts the geojson format to pngcsv format.
        4. Converts the pngcsv format to nodes.csv format.
        5. Extracts centroids from ground truth (GT) data.
        6. Adds GT labels to the nodes.csv file.
        7. Adds Hovernet predictions to the nodes.csv file.

    :param args: The arguments for the post-processing pipeline.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    logger.info('Moving json files to corresponding folders.')
    tr_files = set(get_names(os.path.join(args.root_dir, 'data', 'train', 'gson'), '.geojson'))
    val_files = set(get_names(os.path.join(args.root_dir, 'data', 'validation', 'gson'), '.geojson'))
    ts_files = set(get_names(os.path.join(args.root_dir, 'data', 'test', 'gson'), '.geojson'))
    json_files = set(get_names(os.path.join(args.root_dir, 'data', 'tmp_hov', 'json'), '.json'))
    for folder_name, split_files in zip(['train', 'validation', 'test'], [tr_files, val_files, ts_files]):
        for file in json_files.intersection(split_files):
            shutil.copy(
                os.path.join(args.root_dir, 'data', 'tmp_hov', 'json', file + '.json'),
                os.path.join(args.root_dir, 'data', folder_name, 'json')
            )

    for split in ['train', 'validation']: #, 'test']:
        logger.info(f'Parsing {split} split')
        logger.info('   From json to geojson.')
        newargs = Namespace(
            json_dir=os.path.join(args.root_dir, 'data', split, 'json'),
            gson_dir=os.path.join(args.root_dir, 'data', split, 'gson_hov'),
            num_classes=args.num_classes
        )
        hovernet2geojson_main(newargs)

        logger.info('   From geojson to pngcsv.')
        newargs = Namespace(
            gson_dir=os.path.join(args.root_dir, 'data', split, 'gson_hov'),
            png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv_hov'),
            num_classes=args.num_classes,
        )
        geojson2pngcsv_main(newargs)

        logger.info('   From pngcsv to nodes.csv.')
        newargs = Namespace(
            png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv_hov'),
            orig_dir=os.path.join(args.root_dir, 'data', 'orig'),
            output_path=os.path.join(args.root_dir, 'data', split, 'graphs', 'raw'),
            num_workers=args.num_workers
        )
        pngcsv2graph_main(newargs)

        logger.info('   Extracting centroids from GT.')
        newargs = Namespace(
            png_dir=os.path.join(args.root_dir, 'data', split, 'png'),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv'),
            output_path=os.path.join(args.root_dir, 'data', split, 'centroids')
        )
        pngcsv2centroids_main(newargs)

        logger.info('   Adding GT labels to .nodes.csv.')
        newargs = Namespace(
            graph_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'raw'),
            centroids_dir=os.path.join(args.root_dir, 'data', split, 'centroids'),
            output_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'GT')
        )
        join_graph_gt_main(newargs)

        logger.info('   Adding hovernet predictions to .nodes.csv.')
        newargs = Namespace(
            json_dir=os.path.join(args.root_dir, 'data', split, 'json'),
            graph_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'GT'),
            output_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'preds'),
            num_classes=args.num_classes,
            enable_background=args.enable_background,
        )
        join_hovprob_graph_main(newargs, logger)

    return


def run_graph_pipe(args: Namespace, logger: Logger) -> None:
    """
    Trains the graph models.

    This function trains the graph models using the following steps:
        1. Trains the GCN (Graph Convolutional Network) model.
        2. Trains the GAT (Graph Attention Network) model.

    :param args: The arguments for the graph pipeline.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """
    
    # Train Graph Convolutional Network (GCN)
    newargs = Namespace(
        train_node_dir=os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds'),
        validation_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        # test_node_dir=os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds'),
        test_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        log_dir=os.path.join(args.root_dir, 'gnn_logs'),
        early_stopping_rounds=10,
        batch_size=20,
        model_name='GCN',
        # save_file=os.path.join(args.root_dir, 'gnn_logs', 'gcn_results'),
        save_file=os.path.join(args.root_dir, 'gnn_logs', 'gcn_results'+args.suffix),
        num_confs=32,
        save_dir=os.path.join(args.root_dir, 'weights', 'classification', 'gnn'+args.suffix),
        device='cpu' if args.gpu == '' else 'cuda',
        num_workers=args.num_workers,
        checkpoint_iters=-1,
        num_classes=args.num_classes,
        disable_prior=False,
        disable_morph_feats=False,
        enable_background=args.enable_background,
    )
    train_gnn(newargs)
    
    # Train Graph Attention Network (GAT)
    newargs = Namespace(
        train_node_dir=os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds'),
        validation_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        # test_node_dir=os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds'),
        test_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        log_dir=os.path.join(args.root_dir, 'gnn_logs'),
        early_stopping_rounds=10,
        batch_size=20,
        model_name='ATT',
        # save_file=os.path.join(args.root_dir, 'gnn_logs', 'gat_results'),
        save_file=os.path.join(args.root_dir, 'gnn_logs', 'gat_results'+args.suffix),
        num_confs=32,
        save_dir=os.path.join(args.root_dir, 'weights', 'classification', 'gnn'+args.suffix),
        device='cpu' if args.gpu == '' else 'cuda',
        num_workers=args.num_workers,
        checkpoint_iters=-1,
        num_classes=args.num_classes,
        disable_prior=False,
        disable_morph_feats=False,
        enable_background=args.enable_background,
    )
    train_gnn(newargs)
    
    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--pretrained-path', type=str, help='Path to initial Hovernet weights.')
    parser.add_argument('--format', type=str, default='geojson', help="Format of the input GT.", choices=['geojson', 'pngcsv'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--stroma-mask', action='store_true', help="If enabled, the stroma initial classification of the HoverNet will be done with a specific network.")
    parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')
    parser.add_argument('--suffix', type=str, default="", help='Suffix to add to the name of the models and logs.')
    parser.add_argument('--h', type=float, default=0.5)
    parser.add_argument('--k', type=float, default=0.4)
    return parser


def main():

    # Arguments
    parser = _create_parser()
    args = parser.parse_args()
    if args.suffix != "":
        args.suffix = '_'+args.suffix

    # Logger
    logger = logging.getLogger('train_pipe')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    """
    # Preprocessing: format conversion
    logger.info('Starting preprocessing pipeline.')
    run_preproc_pipe(args, logger)
    logger.info('Finished preprocessing pipeline.')
    
    # Train HoverNet
    logger.info('Starting Hovernet pipeline.')
    run_hov_pipe(args, logger)
    logger.info('Finished Hovernet pipeline.')
    """
    
    # Post-processing HoverNet
    logger.info('Starting postprocessing pipeline.')
    run_postproc_pipe(args, logger)
    logger.info('Finished postprocessing pipeline.')

    # Train Graph Neural Networks
    logger.info('Starting graph pipeline.')
    run_graph_pipe(args, logger)
    logger.info('Finished graph pipeline.')


if __name__ == '__main__':
    main()