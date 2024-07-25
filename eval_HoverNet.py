import argparse
from argparse import Namespace
import logging
from logging import Logger
import os
import shutil

import numpy as np
import pandas as pd

from segmentation.evaluate import main_with_args as eval_segment

from preprocessing import pngprob2centroids_main
from utils.preprocessing import get_names


def run_preprocessing(args: Namespace, logger: Logger) -> None:
    """
    Runs the preprocessing steps to convert the gson format to other formats.

    :param args: The arguments for the preprocessing.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    # First preprocessing
    for split in ['train', 'validation', 'test']:

        """# Extract centroids from PNG and CSV (from GT)
        logger.info('   Extracting centroids from GT.')
        newargs = Namespace(
            png_dir=os.path.join(args.root_dir, 'data', split, 'png'),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv'),
            output_path=os.path.join(args.root_dir, 'data', split, 'centroids')
        )
        pngcsv2centroids_main(newargs)"""

        # Extract centroids from PNG and CSV (from inference)
        logger.info('   Extracting centroids from inference.')
        newargs = Namespace(
            png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'+args.data_suffix),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv_hov'+args.data_suffix),
            output_path=os.path.join(args.root_dir, 'data', split, 'centroids_hov'+args.data_suffix),
            num_classes= args.num_classes-1
        )
        pngprob2centroids_main(newargs)

    return


def run_evaluation(args: Namespace, logger: Logger) -> None:
    """
    Runs the evaluation of Hovernet output for different splits.

    :param args: The arguments for the evaluation.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    for split in ['train', 'validation', 'test']:

        # Names of images
        names = get_names(os.path.join(args.root_dir, 'data', split, 'png'), '.GT_cells.png')

        # Confusion matrices and metrics
        logger.info(f'    Evaluating {split} split')
        if args.debug:
            os.makedirs(os.path.join(args.save_dir, 'hov', args.data_suffix, split, 'conf-matrices', 'hov_individual'), exist_ok=True)
        newargs = Namespace(
            names=names,
            gt_path=os.path.join(args.root_dir, 'data', split, 'centroids'),
            pred_path=os.path.join(args.root_dir, 'data', split, 'centroids_hov'+args.data_suffix),
            save_name=os.path.join(args.save_dir, 'hov', args.data_suffix, split),
            debug_path=os.path.join(args.save_dir, 'hov', args.data_suffix, split, 'conf-matrices', 'hov_individual',
                                    'debug_hov') if args.debug else None,
            num_classes=args.num_classes,
            hov=True,
        )
        eval_segment(newargs, logger)

        if args.debug:
            shutil.move(
                os.path.join(args.save_dir, 'hov', args.data_suffix, split, 'conf-matrices', 'hov_individual', 'debug_hov_global_nobkgr.csv'),
                os.path.join(args.save_dir, 'hov', args.data_suffix, split, 'conf-matrices', 'debug_hov_global_nobkgr_' + split + '.csv'))

            shutil.move(
                os.path.join(args.save_dir, 'hov', args.data_suffix, split, 'conf-matrices', 'hov_individual', 'debug_hov_global_nobkgr.png'),
                os.path.join(args.save_dir, 'hov', args.data_suffix, split, 'conf-matrices', 'debug_hov_global_nobkgr_' + split + '.png'))

    return

def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--data-suffix', type=str, default="", help="Suffix of the folders to save the result.")
    parser.add_argument('--save-dir', type=str, required=True, help='Folder to save the results, without file type.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--debug', action='store_true', help='Whether to save confusion matrices.')
    return parser


def main():
    
    parser = _create_parser()
    args = parser.parse_args()
    if args.data_suffix != "":
        args.data_suffix = '_'+args.data_suffix

    logger = logging.getLogger('eval_pipe')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Preprocessing: format conversion
    logger.info('Starting preprocessing pipeline.')
    run_preprocessing(args, logger)
    logger.info('Finished preprocessing pipeline.')

    # HoverNet evaluation
    logger.info('Starting Hovernet evaluation pipeline.')
    run_evaluation(args, logger)
    logger.info('Finished Hovernet evaluation pipeline.')

if __name__ == '__main__':
    main()