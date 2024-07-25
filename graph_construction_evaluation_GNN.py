import argparse
from argparse import Namespace
import logging
from logging import Logger
import os
import shutil
import pandas as pd
import copy

from preprocessing import graph2centroids_main, centroidspng2csv_main, pngcsv2geojson_main
from preprocessing import geojson2pngcsv_main, pngcsv2graph_main, hovernet2geojson_main, pngcsv2centroids_main, png2graph_main
from segmentation import pngcsv2npy, hov_train, hov_infer
from postprocessing import join_graph_gt_main, join_hovprob_graph_main, add_hovprob_graph_main
from classification import gnn_infer, stromamask_main
from utils.preprocessing import get_names, create_dir

from segmentation.evaluate import main_with_args as eval_segment


def set_best_configuration(args: Namespace, logger: Logger, log_file: str) -> [int, float, str]:
    """
    Sets the best configuration from training based on the F1 score.

    :param args: The arguments for setting the best configuration.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    logger.info('Configuration not provided, using best configuration from training based on F1 score.')

    # Read metrics values for each configuration
    save_file = os.path.join(args.root_dir, 'gnn_logs', args.logs_dir_suffix, log_file)
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

    return int(args.best_num_layers), float(args.best_dropout), args.best_norm_type

def create_results_file(filename: str) -> None:
    """
    Creates header of .csv result file to append results.
    filename must not contain extension.
    """
    with open(filename + '.csv', 'w') as f:
        conf = 'DEGREE,DISTANCE,NUM_LAYERS,DROPOUT,NORM_TYPE'
        metrics = 'Macro F1,Weighted F1,Micro F1,ECE,Macro F1 (bkgr),Weighted F1 (bkgr),Micro F1 (bkgr)'
        print(conf+','+metrics, file=f)

"""
def append_results(
        filename: str,
        degree: int, distance: int, num_layers: int, dropout: float, bn_type: str,
        macro: float, weighted: float, micro: float, ece: float,
        macro_bkgr: float, weighted_bkgr: float, micro_bkgr: float,
        ) -> None:
"""
def append_results(filename: str, read_filename: str,
                   degree: int, distance: int, num_layers: int, dropout: float, bn_type: str) -> None:
    """
    Appends result to given filename.
    filename must not contain extension.
    """

    # Read csv
    gnn_results = pd.read_csv(read_filename)
    macro, weighted, micro = gnn_results[['Macro F1', 'Weighted F1', 'Micro F1']].values[0]
    ece = gnn_results['ECE'].values[0]
    macro_bkgr, weighted_bkgr, micro_bkgr = gnn_results[['Macro F1 (bkgr)', 'Weighted F1 (bkgr)', 'Micro F1 (bkgr)']].values[0]

    # Save results
    with open(filename + '.csv', 'a') as f:
        print(degree, distance, num_layers, dropout, bn_type,
              macro, weighted, micro, ece, macro_bkgr, weighted_bkgr, micro_bkgr,
              file=f, sep=',')


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
    for split in ['train', 'validation']: #, 'test']:
        logger.info(f'Parsing {split} split')

        """
        # Convert to PNG and CSV
        if args.format != 'pngcsv':

            # JSON to GeoJSON
            # logger.info('   From json to geojson.')
            newargs = Namespace(
                json_dir=os.path.join(args.root_dir, 'data', split, 'json'),
                gson_dir=os.path.join(args.root_dir, 'data', split, 'gson_hov'),
                num_classes=args.num_classes
            )
            hovernet2geojson_main(newargs)

            # GeoJSON to PNG and CSV
            # logger.info('   From geojson to pngcsv.')
            newargs = Namespace(
                gson_dir=os.path.join(args.root_dir, 'data', split, 'gson_hov'),
                png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'),
                csv_dir=os.path.join(args.root_dir, 'data', split, 'csv_hov'),
                num_classes=args.num_classes,
            )
            geojson2pngcsv_main(newargs)
        """

        # PNG and CSV to graph representation (morphological and colour features of cells)
        # logger.info('   From pngcsv to nodes.csv.')
        newargs = Namespace(
            png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'+args.data_suffix),
            orig_dir=os.path.join(args.root_dir, 'data', 'orig'),
            output_path=os.path.join(args.root_dir, 'data', split, 'graphs', 'raw'),
            num_workers=args.num_workers
        )
        png2graph_main(newargs)

        # Add HoverNet predictions to nodes
        # logger.info('   Adding hovernet predictions to .nodes.csv.')
        newargs = Namespace(
            json_dir=os.path.join(args.root_dir, 'data', split, 'json'+args.data_suffix),
            graph_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'raw'),
            output_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'hovpreds'),
            num_classes=args.num_classes,
            enable_background=args.enable_background
        )
        join_hovprob_graph_main(newargs, logger)
        # add_hovprob_graph_main(newargs, logger)

    return


def run_graph_infer_pipe(args: Namespace, logger: Logger, conf_name: str) -> None:
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
    for split in ['train', 'validation']:  # , 'test']:
        newargs = Namespace(
            # root_dir=args.root_dir,
            node_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'hovpreds'),
            output_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'gnn_preds'),

            weights=os.path.join(args.root_dir, 'weights', 'classification', 'gnn'+args.model_suffix, conf_name, 'weights', model_name + '.pth'),
            conf=os.path.join(args.root_dir, 'weights', 'classification', 'gnn'+args.model_suffix, conf_name, 'confs', model_name + '.json'),
            normalizers=os.path.join(args.root_dir, 'weights', 'classification', 'gnn'+args.model_suffix, conf_name, 'normalizers', model_name + '.pkl'),

            degree=args.degree,
            distance=args.distance,

            num_classes=args.num_classes,
            disable_prior=False,
            disable_morph_feats=False,
            disable_coordinates=False,
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

    for split in ['train', 'validation']:  # , 'test']:

        # Stroma prediction
        if args.stroma_mask:
            
            orig_dir=os.path.join(args.root_dir, 'data', 'orig')
            newargs = Namespace(
                orig_dir=os.path.join(args.root_dir, 'data', 'orig'),
                graph_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'gnn_preds'),
                png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'+args.data_suffix),
                num_classes=args.num_classes
            )
            stromamask_main(newargs)

        # Centroids class information
        # logger.info('   Converting .nodes.csv to .centroids.csv.')
        newargs = Namespace(
            graph_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'gnn_preds'),
            centroids_dir=os.path.join(args.root_dir, 'data', split, 'centroids_gnn'+args.result_suffix),
            num_classes=args.num_classes,
            enable_background=args.enable_background,
        )
        graph2centroids_main(newargs)

        # Centroids to PNG and CSV
        # logger.info('   Converting .centroids.csv and .GT_cells.png to .class.csv.')
        newargs = Namespace(
            centroids_dir=os.path.join(args.root_dir, 'data', split, 'centroids_gnn'+args.result_suffix),
            input_png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'+args.data_suffix),
            png_dir=os.path.join(args.root_dir, 'data', split, 'png_gnn'+args.result_suffix),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv_gnn'+args.result_suffix),
        )
        centroidspng2csv_main(newargs)

        # PNG and CSV to GeoJSON
        # logger.info('   Converting png/csv to geojson.')
        newargs = Namespace(
            png_dir=os.path.join(args.root_dir, 'data', split, 'png_gnn'+args.result_suffix),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv_gnn'+args.result_suffix),
            gson_dir=os.path.join(args.root_dir, 'data', split, 'gson_gnn'+args.result_suffix),
            num_classes=args.num_classes
        )
        pngcsv2geojson_main(newargs)

    return

def run_evaluation(args: Namespace, logger: Logger) -> None:
    """
    Runs the evaluation of graph output for different splits.

    :param args: The arguments for the evaluation.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    for split in ['train', 'validation']: #, 'test']:

        # Names of images
        names = get_names(os.path.join(args.root_dir, 'data', split, 'png'), '.GT_cells.png')

        # Confusion matrices and metrics
        logger.info(f'    Evaluating {split} split')
        create_dir(os.path.join(args.save_dir, 'gnn', args.result_suffix, split))
        if args.debug:
            create_dir(os.path.join(args.save_dir, 'gnn', args.result_suffix, split, 'conf-matrices', 'gnn_individual'), exist_ok=True)
        newargs = Namespace(
            names=names,
            gt_path=os.path.join(args.root_dir, 'data', split, 'centroids'),
            pred_path=os.path.join(args.root_dir, 'data', split, 'centroids_gnn'+args.result_suffix),
            save_name=os.path.join(args.save_dir, 'gnn', args.result_suffix, split),
            debug_path=os.path.join(args.save_dir, 'gnn', args.result_suffix, split, 'conf-matrices', 'gnn_individual',
                                    'debug_gnn') if args.debug else None,
            num_classes=args.num_classes,
        )
        eval_segment(newargs, logger)

        # Compute ECE
        # compute_ece(args, split, is_hov=False)

        if args.debug:
            shutil.move(
                os.path.join(args.save_dir, 'gnn', args.data_suffix, split, 'conf-matrices', 'gnn_individual', 'debug_gnn_global.csv'),
                os.path.join(args.save_dir, 'gnn', args.data_suffix, split, 'conf-matrices', 'debug_gnn_global_' + split + '.csv'))

            shutil.move(
                os.path.join(args.save_dir, 'gnn', args.data_suffix, split, 'conf-matrices', 'gnn_individual', 'debug_gnn_global.png'),
                os.path.join(args.save_dir, 'gnn', args.data_suffix, split, 'conf-matrices', 'debug_gnn_global_' + split + '.png'))

    return


def _create_parser():
    parser = argparse.ArgumentParser()

    # TODO: input and output dir???
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--data-suffix', type=str, default="", help="Suffix of the folders to load the results.")
    parser.add_argument('--logs-dir-suffix', type=str, default="", help="Suffix of the folder where the logs are stored.")
    parser.add_argument('--result-suffix', type=str, default="", help="Suffix of the folders to save the results.")
    parser.add_argument('--save-dir', type=str, required=True, help='Folder to save the results, without file type.')

    # parser.add_argument('--pretrained-path', type=str, help='Path to initial Hovernet weights.')
    parser.add_argument('--format', type=str, default='pngcsv', help="Format of the input GT.", choices=['geojson', 'pngcsv'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--debug', action='store_true', help='Whether to save confusion matrices.')
    
    parser.add_argument('--stroma-mask', action='store_true', help="If enabled, the stroma classification (after GNN) will be done with a specific network.")
    parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')

    parser.add_argument('--model-suffix', type=str, default="", help='Suffix of the model.')
    parser.add_argument('--best-num-layers', type=str, help='Optimal number of layers when training GNNs.')
    parser.add_argument('--best-dropout', type=str, help='Optimal dropout rate when training GNNs')
    parser.add_argument('--best-norm-type', type=str, help='Optimal type of normalization layers when training GNNs')
    parser.add_argument('--best-arch', type=str, help='Best architecture (convolutional, attention, ...) when training GNNs', required=True, choices=['GCN', 'ATT', 'TRANSFORMER', 'RESIDUAL', 'INITIAL_RESIDUAL'])

    return parser


def main():

    # Arguments
    parser = _create_parser()
    args = parser.parse_args()
    if args.model_suffix != "":
        args.model_suffix = '_' + args.model_suffix
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

    # File to save best results of each configuration
    logs_path = os.path.join(args.root_dir, 'gnn_logs', args.logs_dir_suffix)
    save_final_logs = os.path.join(logs_path, 'best_results_graph_construction')
    create_results_file(save_final_logs)
    
    args.num_classes = args.num_classes if not args.stroma_mask else args.num_classes - 1
    num_classes = copy.copy(args.num_classes)

    # Pre-processing Graph (post-processing HoverNet)
    logger.info('Starting graph preprocessing pipeline.')
    run_graph_preproc_pipe(args, logger)
    logger.info('Finished graph preprocessing pipeline.')

    # Logs files
    logs_files = sorted([log_file for log_file in os.listdir(logs_path) if log_file.endswith('.csv') and log_file.startswith("gtn")])
    for log_file in logs_files:

        conf_name = log_file[12:-4] # 'degree=' + str(degree) + '_' + 'distance=' + str(distance)
        args.degree = int(conf_name.split('_')[0][7:])
        args.distance = int(conf_name.split('_')[1][9:])

        logger.info("PROCESSING " + conf_name)

        # Configuration to make inference
        best_num_layers, best_dropout, best_norm_type = set_best_configuration(args, logger, log_file)
        args.num_classes = num_classes
        # print("NUM CLASSES:", args.num_classes)

        # GNN inference
        logger.info('Starting graph inference pipeline.')
        run_graph_infer_pipe(args, logger, conf_name)
        logger.info('Finished graph inference pipeline.')

        # Postprocessing: format conversion
        logger.info('Starting graph postprocessing pipeline.')
        run_graph_postproc_pipe(args, logger)
        logger.info('Finished graph postprocessing pipeline.')

        # GNN evaluation
        logger.info('Starting graph evaluation pipeline.')
        run_evaluation(args, logger)
        logger.info('Finished graph evaluation pipeline.')

        # Information in global file
        read_logs = os.path.join(args.save_dir, 'gnn', args.result_suffix, 'validation_all.csv')
        append_results(save_final_logs, read_logs, args.degree, args.distance, best_num_layers, best_dropout, best_norm_type)

if __name__ == '__main__':
    main()