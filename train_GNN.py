import argparse
from argparse import Namespace
import logging
from logging import Logger
import os
import shutil

from preprocessing import pngcsv2centroids_main, pngprob2graph_main
from postprocessing import join_graph_gt_main
from classification import gnn_train


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

        # PNG and CSV to graph representation (morphological and colour features of cells)
        # logger.info('   From pngcsv to nodes.csv.')
        newargs = Namespace(
            png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'+args.data_suffix),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv_hov'+args.data_suffix),
            orig_dir=os.path.join(args.root_dir, 'data', 'orig'),
            output_path=os.path.join(args.root_dir, 'data', split, 'graphs', 'raw', args.data_suffix),
            num_classes=args.num_classes,
            enable_background=args.enable_background,
            num_workers=args.num_workers,
            train=True
        )
        pngprob2graph_main(newargs)

        # Extract centroids (from GT)
        logger.info('   Extracting centroids from GT.')
        newargs = Namespace(
            png_dir=os.path.join(args.root_dir, 'data', split, 'png'),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv'),
            output_path=os.path.join(args.root_dir, 'data', split, 'centroids'),
            eval=False
        )
        pngcsv2centroids_main(newargs)

        # Find 1-1 matching and take GT labels to nodes
        logger.info('   Adding GT labels to .nodes.csv.')
        newargs = Namespace(
            graph_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'raw', args.data_suffix),
            centroids_dir=os.path.join(args.root_dir, 'data', split, 'centroids'),
            output_dir=os.path.join(args.root_dir, 'data', split, 'graphs', 'preds', args.data_suffix),
            background_class=args.enable_background,
        )
        join_graph_gt_main(newargs)

    return


def run_graph_train_pipe(args: Namespace, logger: Logger) -> None:
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

    if args.graph_construction:

        # Train Graph Transformer Network (GTN)
        logger.info('    Transformer')
        newargs = Namespace(
            train_node_dir=os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds', args.data_suffix),
            validation_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds', args.data_suffix),
            test_node_dir=os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds', args.data_suffix),
            #test_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
            log_dir=os.path.join(args.root_dir, 'gnn_logs'),
            early_stopping_rounds=10,
            batch_size=20,
            model_name='TRANSFORMER',
            save_file=os.path.join(args.root_dir, 'gnn_logs', 'gtn_results'+args.model_suffix),
            num_confs=32, # 24,
            save_dir=os.path.join(args.root_dir, 'weights', 'classification', 'gnn'+args.model_suffix),
            device='cpu' if args.gpu == '' else 'cuda',
            num_workers=0, #args.num_workers
            checkpoint_iters=-1,
            num_classes=args.num_classes,
            disable_prior=args.disable_prior,
            disable_area=args.disable_area,
            disable_perimeter=args.disable_perimeter,
            disable_std=args.disable_std,
            disable_hist=args.disable_hist,
            disable_coordinates=args.disable_coordinates,
            enable_background=args.enable_background,
            stroma_mask=args.stroma_mask,
            graph_construction=args.graph_construction,
        )
        gnn_train(newargs)
        
        """# Train Residual Connection (GRN)
        logger.info('    Residual')
        newargs = Namespace(
            train_node_dir=os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds', args.data_suffix),
            validation_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds', args.data_suffix),
            test_node_dir=os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds', args.data_suffix),
            #test_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
            log_dir=os.path.join(args.root_dir, 'gnn_logs'),
            early_stopping_rounds=10,
            batch_size=20,
            model_name='RESIDUAL',
            save_file=os.path.join(args.root_dir, 'gnn_logs', 'grn_results' + args.model_suffix),
            num_confs=32,
            save_dir=os.path.join(args.root_dir, 'weights', 'classification', 'gnn' + args.model_suffix),
            device='cpu' if args.gpu == '' else 'cuda',
            num_workers=0, #args.num_workers
            checkpoint_iters=-1,
            num_classes=args.num_classes,
            disable_prior=args.disable_prior,
            disable_area=args.disable_area,
            disable_perimeter=args.disable_perimeter,
            disable_std=args.disable_std,
            disable_hist=args.disable_hist,
            disable_coordinates=args.disable_coordinates,
            enable_background=args.enable_background,
            stroma_mask=args.stroma_mask,
            graph_construction=args.graph_construction,
        )
        gnn_train(newargs)"""

        return

    # Train Graph Convolutional Network (GCN)
    logger.info('    GCN')
    newargs = Namespace(
        train_node_dir=os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds', args.data_suffix),
        validation_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds', args.data_suffix),
        test_node_dir=os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds', args.data_suffix),
        #test_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        log_dir=os.path.join(args.root_dir, 'gnn_logs'),
        early_stopping_rounds=10,
        batch_size=20,
        model_name='GCN',
        save_file=os.path.join(args.root_dir, 'gnn_logs', 'gcn_results'+args.model_suffix),
        num_confs=32,
        save_dir=os.path.join(args.root_dir, 'weights', 'classification', 'gnn'+args.model_suffix),
        device='cpu' if args.gpu == '' else 'cuda',
        num_workers=0, #args.num_workers
        checkpoint_iters=-1,
        num_classes=args.num_classes,
        disable_prior=args.disable_prior,
        disable_area=args.disable_area,
        disable_perimeter=args.disable_perimeter,
        disable_std=args.disable_std,
        disable_hist=args.disable_hist,
        disable_coordinates=args.disable_coordinates,
        enable_background=args.enable_background,
        stroma_mask=args.stroma_mask,
        graph_construction=args.graph_construction,
    )
    gnn_train(newargs)

    # Train Graph Transformer Network (GTN)
    logger.info('    Transformer')
    newargs = Namespace(
        train_node_dir=os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds', args.data_suffix),
        validation_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds', args.data_suffix),
        test_node_dir=os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds', args.data_suffix),
        #test_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        log_dir=os.path.join(args.root_dir, 'gnn_logs'),
        early_stopping_rounds=10,
        batch_size=20,
        model_name='TRANSFORMER',
        save_file=os.path.join(args.root_dir, 'gnn_logs', 'gtn_results'+args.model_suffix),
        num_confs=32,
        save_dir=os.path.join(args.root_dir, 'weights', 'classification', 'gnn'+args.model_suffix),
        device='cpu' if args.gpu == '' else 'cuda',
        num_workers=0, #args.num_workers
        checkpoint_iters=-1,
        num_classes=args.num_classes,
        disable_prior= args.disable_prior,
        disable_area= args.disable_area,
        disable_perimeter= args.disable_perimeter,
        disable_std= args.disable_std,
        disable_hist= args.disable_hist,
        disable_coordinates= args.disable_coordinates,
        enable_background=args.enable_background,
        stroma_mask=args.stroma_mask,
        graph_construction=args.graph_construction,
    )
    gnn_train(newargs)

    # Train Initial Residual Connection (GIRN)
    logger.info('    Initial Residual')
    newargs = Namespace(
        train_node_dir=os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds', args.data_suffix),
        validation_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds', args.data_suffix),
        test_node_dir=os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds', args.data_suffix),
        #test_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        log_dir=os.path.join(args.root_dir, 'gnn_logs'),
        early_stopping_rounds=10,
        batch_size=20,
        model_name='INITIAL_RESIDUAL',
        save_file=os.path.join(args.root_dir, 'gnn_logs', 'girn_results' + args.model_suffix),
        num_confs=32,
        save_dir=os.path.join(args.root_dir, 'weights', 'classification', 'gnn' + args.model_suffix),
        device='cpu' if args.gpu == '' else 'cuda',
        num_workers=0, #args.num_workers
        checkpoint_iters=-1,
        num_classes=args.num_classes,
        disable_prior=args.disable_prior,
        disable_area=args.disable_area,
        disable_perimeter=args.disable_perimeter,
        disable_std=args.disable_std,
        disable_hist=args.disable_hist,
        disable_coordinates=args.disable_coordinates,
        enable_background=args.enable_background,
        stroma_mask=args.stroma_mask,
        graph_construction=args.graph_construction,
    )
    gnn_train(newargs)

    # Train Residual Connection (GRN)
    logger.info('    Residual')
    newargs = Namespace(
        train_node_dir=os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds', args.data_suffix),
        validation_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds', args.data_suffix),
        test_node_dir=os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds', args.data_suffix),
        #test_node_dir=os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        log_dir=os.path.join(args.root_dir, 'gnn_logs'),
        early_stopping_rounds=10,
        batch_size=20,
        model_name='RESIDUAL',
        save_file=os.path.join(args.root_dir, 'gnn_logs', 'grn_results' + args.model_suffix),
        num_confs=32,
        save_dir=os.path.join(args.root_dir, 'weights', 'classification', 'gnn' + args.model_suffix),
        device='cpu' if args.gpu == '' else 'cuda',
        num_workers=0, #args.num_workers
        checkpoint_iters=-1,
        num_classes=args.num_classes,
        disable_prior=args.disable_prior,
        disable_area=args.disable_area,
        disable_perimeter=args.disable_perimeter,
        disable_std=args.disable_std,
        disable_hist=args.disable_hist,
        disable_coordinates=args.disable_coordinates,
        enable_background=args.enable_background,
        stroma_mask=args.stroma_mask,
        graph_construction=args.graph_construction,
    )
    gnn_train(newargs)

    return



def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--data-suffix', type=str, default="", help="Suffix of the folders to load the results.")
    # parser.add_argument('--input-dir', type=str, help='Folder containing patches to process.', required=True)
    # parser.add_argument('--output-dir', type=str, help='Folder where to save results. Additional subfolder will be created.', required=True)

    # parser.add_argument('--pretrained-path', type=str, help='Path to initial Hovernet weights.')
    parser.add_argument('--model-suffix', type=str, default="", help='Suffix of the model.')

    parser.add_argument('--format', type=str, default='pngcsv', help="Format of the input GT.", choices=['geojson', 'pngcsv'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    
    parser.add_argument('--disable-prior', action='store_true', help="If enabled, remove prior features.")
    parser.add_argument('--disable-area', action='store_true', help="If enabled, remove area feature.")
    parser.add_argument('--disable-perimeter', action='store_true', help="If enabled, remove perimeter feature.")
    parser.add_argument('--disable-std', action='store_true', help="If enabled, remove std feature.")
    parser.add_argument('--disable-hist', action='store_true', help="If enabled, remove histogram features.")
    parser.add_argument('--disable-coordinates', action='store_true', help="If enabled, remove coordinate features.")

    parser.add_argument('--stroma-mask', action='store_true', help="If enabled, the stroma classification (after GNN) will be done with a specific network.")
    parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')
    parser.add_argument('--graph-construction', action='store_true', help='If enabled, train different configurations for the graph construction (degree and distance).')
    return parser


def main():

    # Arguments
    parser = _create_parser()
    args = parser.parse_args()
    if args.model_suffix != "":
        args.model_suffix = '_' + args.model_suffix
    if args.data_suffix != "":
        args.data_suffix = '_' + args.data_suffix

    # Logger
    logger = logging.getLogger('train_pipe')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Pre-processing Graph (post-processing HoverNet)
    logger.info('Starting graph preprocessing pipeline.')
    run_graph_preproc_pipe(args, logger)
    logger.info('Finished graph preprocessing pipeline.')

    # Train Graph Neural Networks
    logger.info('Starting graph train pipeline.')
    run_graph_train_pipe(args, logger)
    logger.info('Finished graph train pipeline.')


if __name__ == '__main__':
    main()