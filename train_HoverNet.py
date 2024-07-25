import argparse
from argparse import Namespace
import logging
from logging import Logger
import os
import shutil
import json

from preprocessing import geojson2pngcsv_main, pngcsv2geojson_main
from segmentation import pngcsv2npy, hov_train
from utils.preprocessing import create_dir

def create_type_info(args: Namespace):

    type_info = {
        "0": ["background", [0, 0, 0]],
        "1": ["nontumour", [255, 0, 0]],
        "2": ["tumour", [0, 255, 0]]
    }
    if args.num_classes != 2:
        colors = [
                     [0, 0, 0],
                     [0, 0, 255],
                     [255, 255, 0],
                     [0, 255, 0],
                     [255, 0, 0],
                     [0, 255, 255],
                     [0, 0, 255],
                     [255, 0, 255],
                     [255, 255, 255]
                 ][:args.num_classes + 1]
        type_info = {str(k): ["Class" + str(k), v] for k, v in enumerate(colors)}
        type_info['0'] = ["background", [0, 0, 0]]

    create_dir(os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet'+args.suffix))
    with open(os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet'+args.suffix, 'type_info.json'), 'w') as f:
        json.dump(type_info, f)


def run_hov_preproc_pipe(args: Namespace, logger: Logger) -> None:
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

    # Create inforation for the model
    create_type_info(args)

    # Convert formats
    for split in ['train', 'validation','test']:
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
            out_dir=os.path.join(args.root_dir, 'data', split, 'npy', '256'),
            save_example=False, use_labels=True, split=False,
            shape='256'
        )
        pngcsv2npy(newargs)

    return

def run_hov_train_pipe(args: Namespace, logger: Logger) -> None:
    """
    Trains Hovernet.

    This function performs the following steps:
        1. Trains Hovernet on the training data.

    :param args: The arguments for the Hovernet pipeline.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    # Train HoverNet
    logger.info('Starting training.')
    newargs = Namespace(
        gpu=args.gpu, view=None, save_name=None,
        log_dir=os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet'+args.suffix),
        train_dir=os.path.join(args.root_dir, 'data', 'train', 'npy', '256'),
        valid_dir=os.path.join(args.root_dir, 'data', 'validation', 'npy', '256'),
        pretrained_path=args.pretrained_path,
        shape='256',
        num_classes=args.num_classes if not args.stroma_mask else args.num_classes-1,
        stroma_mask=args.stroma_mask
    )
    hov_train(newargs)

    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--pretrained-path', type=str, help='Path to initial Hovernet weights.')
    parser.add_argument('--format', type=str, default='geojson', help="Format of the input GT.", choices=['geojson', 'pngcsv'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--stroma-mask', action='store_true', help="If enabled, the stroma initial classification (before GNN) will be done with a specific network.")
    # parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')
    parser.add_argument('--suffix', type=str, default="", help='Suffix to add to the folder of the models and logs.')
    # parser.add_argument('--h', type=float, default=0.5)
    # parser.add_argument('--k', type=float, default=0.4)
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

    # Preprocessing: format conversion
    logger.info('Starting Hovernet preprocessing pipeline.')
    run_hov_preproc_pipe(args, logger)
    logger.info('Finished Hovernet preprocessing pipeline.')

    # Train HoverNet
    logger.info('Starting Hovernet train pipeline.')
    run_hov_train_pipe(args, logger)
    logger.info('Finished Hovernet train pipeline.')

if __name__ == '__main__':
    main()