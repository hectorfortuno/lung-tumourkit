import argparse
from argparse import Namespace
import logging
from logging import Logger
import os
import shutil
import pandas as pd

from preprocessing import graph2csv_main
from preprocessing import pngprob2graph_main, hovernet2pngcsv_main
from classification import gnn_infer, stromamask_main

def set_best_configuration(args: Namespace, logger: Logger) -> None:
    """
    Sets the best configuration from training based on the F1 score.

    :param args: The arguments for setting the best configuration.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    logger.info('Configuration not provided, using best configuration from training based on F1 score.')

    # Read metrics values for each configuration
    if args.graph_construction:
        if args.best_arch == 'GCN':
            save_file = os.path.join(args.root_dir, 'gnn_logs', 'Graph_construction', 'gcn_results'+args.result_suffix+'.csv')
        elif args.best_arch == 'ATT':
            save_file = os.path.join(args.root_dir, 'gnn_logs', 'Graph_construction', 'gat_results'+args.result_suffix+'.csv')
        elif args.best_arch == 'TRANSFORMER':
            save_file = os.path.join(args.root_dir, 'gnn_logs', 'Graph_construction', 'gtn_results' + args.result_suffix + '.csv')
        elif args.best_arch == 'INITIAL_RESIDUAL':
            save_file = os.path.join(args.root_dir, 'gnn_logs', 'Graph_construction', 'girn_results' + args.result_suffix + '.csv')
        elif args.best_arch == 'RESIDUAL':
            save_file = os.path.join(args.root_dir, 'gnn_logs', 'Graph_construction', 'grn_results' + args.result_suffix + '.csv')
        else:
            assert False, 'Architecture not supported'
    else:
        if args.best_arch == 'GCN':
            save_file = os.path.join(args.root_dir, 'gnn_logs', 'gcn_results'+args.model_suffix+'.csv')
        elif args.best_arch == 'ATT':
            save_file = os.path.join(args.root_dir, 'gnn_logs', 'gat_results'+args.model_suffix+'.csv')
        elif args.best_arch == 'TRANSFORMER':
            save_file = os.path.join(args.root_dir, 'gnn_logs', 'gtn_results' + args.model_suffix + '.csv')
        elif args.best_arch == 'INITIAL_RESIDUAL':
            save_file = os.path.join(args.root_dir, 'gnn_logs', 'girn_results' + args.model_suffix + '.csv')
        elif args.best_arch == 'RESIDUAL':
            save_file = os.path.join(args.root_dir, 'gnn_logs', 'grn_results' + args.model_suffix + '.csv')
        else:
            assert False, 'Architecture not supported'
    gnn_results = pd.read_csv(save_file)

    # Obtain configuration with best value
    num_classes = args.num_classes if not args.enable_background else args.num_classes + 1
    if num_classes == 2:
        best_conf = gnn_results.sort_values(by='F1 Score', ascending=False).iloc[0]
    else:
        best_conf = gnn_results.sort_values(by='Weighted F1', ascending=False).iloc[0]
    args.best_num_layers = str(best_conf['NUM_LAYERS'])
    args.best_dropout = str(best_conf['DROPOUT'])
    args.best_norm_type = str(best_conf['NORM_TYPE'])

    logger.info("BEST CONFIGURATION with "+args.best_arch+' is NUM_LAYERS='+args.best_num_layers+', DROPOUT='+args.best_dropout+', NORM_TYPE='+args.best_norm_type)

    return


def run_graph_preproc_pipe(args: Namespace, logger: Logger) -> None:
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

    # Convert format
    for split in ['train', 'validation', 'test']:
        logger.info(f'Parsing {split} split')

        newargs = Namespace(
            json_dir=os.path.join(args.root_dir, 'data', split, 'json'+args.data_suffix),
            png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'+args.data_suffix),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv_hov'+args.data_suffix),
            num_classes=args.num_classes,
            num_workers=args.num_workers,
            class_csv = False
        )
        hovernet2pngcsv_main(newargs)

        # PNG and CSV to graph representation (morphological and colour features of cells, and hovernet probabilities prediction)
        # logger.info('   From pngcsv to nodes.csv.')
        newargs = Namespace(
            png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'+args.data_suffix),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv_hov'+args.data_suffix),
            orig_dir=os.path.join(args.root_dir, 'data', 'orig'),
            output_path=os.path.join(args.root_dir, 'data', split, 'graphs', 'hovpreds', args.data_suffix),
            num_classes=args.num_classes,
            enable_background=args.enable_background,
            num_workers=args.num_workers,
            train=False
        )
        pngprob2graph_main(newargs)

    return


def run_graph_infer_pipe(args: Namespace, logger: Logger) -> None:
    """
    Runs the graph inference.

    :param args: The arguments for running graph inference.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    # Variables
    model_name = 'best_'+args.best_arch+'_'+args.best_num_layers+'_'+args.best_dropout+'_'+args.best_norm_type

    # Inference GNN
    logger.info('Starting graph inference.')
    for split in ['train', 'validation', 'test']:
        newargs = Namespace(
            # root_dir=args.root_dir,
            node_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'hovpreds', args.data_suffix),
            output_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'gnn_preds', args.result_suffix),
            
            weights=os.path.join(args.root_dir, 'weights', 'classification', 'gnn'+args.model_suffix, 'weights', model_name + '.pth'),
            conf=os.path.join(args.root_dir, 'weights', 'classification', 'gnn'+args.model_suffix, 'confs', model_name + '.json'),
            normalizers=os.path.join(args.root_dir, 'weights', 'classification', 'gnn'+args.model_suffix, 'normalizers', model_name + '.pkl'),

            num_classes=args.num_classes,
            degree=args.degree,
            distance=args.distance,

            disable_prior= args.disable_prior,
            disable_area= args.disable_area,
            disable_perimeter= args.disable_perimeter,
            disable_std= args.disable_std,
            disable_hist= args.disable_hist,
            disable_coordinates= args.disable_coordinates,
            enable_background=args.enable_background,
            stroma_mask=args.stroma_mask,
        )
        gnn_infer(newargs)

    return


def run_graph_postproc_pipe(args: Namespace, logger: Logger) -> None:
    """
    Performs post-processing steps on GNN output.

    :param args: The arguments for running post-processing on GNN output.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """
    # logger.info('Parsing gnn output.')

    if args.stroma_mask:
        args.num_classes = args.num_classes + 1

    post = 0

    for split in ['train', 'validation', 'test']:

        # Stroma prediction
        if args.stroma_mask:
            
            newargs = Namespace(
                orig_dir=os.path.join(args.root_dir, 'data', 'orig'),
                graph_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'gnn_preds', args.result_suffix),
                png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'+args.data_suffix),
                num_classes=args.num_classes
            )
            stromamask_main(newargs)

        # Graph to PNG and CSV
        logger.info('   Converting .nodes.csv to .class.csv. and copying .GT_cells.png')
        newargs = Namespace(
            graph_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'gnn_preds', args.result_suffix),
            input_png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'+args.data_suffix),
            png_dir=os.path.join(args.root_dir, 'data', split, 'png_gnn'+args.result_suffix),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv_gnn'+args.result_suffix),
            num_classes=args.num_classes,
            enable_background = args.enable_background
        )
        graph2csv_main(newargs)

    return

def _create_parser():
    parser = argparse.ArgumentParser()

    # TODO: input and output dir???
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--data-suffix', type=str, default="", help="Suffix of the folders to load the results.")
    parser.add_argument('--result-suffix', type=str, default="", help="Suffix of the folders to save the results.")

    # parser.add_argument('--pretrained-path', type=str, help='Path to initial Hovernet weights.')
    parser.add_argument('--format', type=str, default='pngcsv', help="Format of the input GT.", choices=['geojson', 'pngcsv'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    
    parser.add_argument('--stroma-mask', action='store_true', help="If enabled, the stroma classification (after GNN) will be done with a specific network.")
    parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')

    parser.add_argument('--disable-prior', action='store_true', help="If enabled, remove prior features.")
    parser.add_argument('--disable-area', action='store_true', help="If enabled, remove area feature.")
    parser.add_argument('--disable-perimeter', action='store_true', help="If enabled, remove perimeter feature.")
    parser.add_argument('--disable-std', action='store_true', help="If enabled, remove std feature.")
    parser.add_argument('--disable-hist', action='store_true', help="If enabled, remove histogram features.")
    parser.add_argument('--disable-coordinates', action='store_true', help="If enabled, remove coordinate features.")

    parser.add_argument('--model-suffix', type=str, default="", help='Suffix of the model.')
    parser.add_argument('--best-num-layers', type=str, help='Optimal number of layers when training GNNs.')
    parser.add_argument('--best-dropout', type=str, help='Optimal dropout rate when training GNNs')
    parser.add_argument('--best-norm-type', type=str, help='Optimal type of normalization layers when training GNNs')
    parser.add_argument('--best-arch', type=str, help='Best architecture (convolutional, attention, ...) when training GNNs', required=True, choices=['GCN', 'ATT', 'TRANSFORMER', 'RESIDUAL', 'INITIAL_RESIDUAL'])
    parser.add_argument('--degree', type=int, default=None)
    parser.add_argument('--distance', type=int, default=None)
    parser.add_argument('--graph-construction', action='store_true', help='If enabled, infer in a configuration for the graph construction (degree and distance).')

    return parser


def main():

    # Arguments
    parser = _create_parser()
    args = parser.parse_args()
    if args.model_suffix != "":
        args.model_suffix = '_' + args.model_suffix
    if args.graph_construction:
        args.model_suffix = args.model_suffix + '/' + args.result_suffix
    if args.data_suffix != "":
        args.data_suffix = '_' + args.data_suffix
    if args.result_suffix != "":
        args.result_suffix = '_' + args.result_suffix

    # Logger
    logger = logging.getLogger('train_pipe')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Configuration to make inference
    args.num_classes = args.num_classes if not args.stroma_mask else args.num_classes - 1
    if args.best_num_layers is None or args.best_dropout is None or args.best_norm_type is None:
        set_best_configuration(args, logger)

    # Pre-processing Graph (post-processing HoverNet)
    logger.info('Starting graph preprocessing pipeline.')
    run_graph_preproc_pipe(args, logger)
    logger.info('Finished graph preprocessing pipeline.')

    # GNN inference
    logger.info('Starting graph inference pipeline.')
    run_graph_infer_pipe(args, logger)
    logger.info('Finished graph inference pipeline.')

    # Postprocessing: format conversion
    logger.info('Starting graph postprocessing pipeline.')
    run_graph_postproc_pipe(args, logger)
    logger.info('Finished graph postprocessing pipeline.')


if __name__ == '__main__':
    main()