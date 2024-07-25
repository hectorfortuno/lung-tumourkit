import argparse
from argparse import Namespace
import logging
from logging import Logger
import os
import shutil
import subprocess

from preprocessing import hovernet2pngcsv_main
from segmentation import hov_infer
from utils.preprocessing import get_names, create_dir


def run_hov_postproc_pipe(args: Namespace, logger: Logger) -> None:
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

    logger.info('Moving json files to corresponding folders.')
    # Images of each subset
    tr_files = set(get_names(os.path.join(args.root_dir, 'data', 'train', 'gson'), '.geojson'))
    val_files = set(get_names(os.path.join(args.root_dir, 'data', 'validation', 'gson'), '.geojson'))
    ts_files = set(get_names(os.path.join(args.root_dir, 'data', 'test', 'gson'), '.geojson'))

    # Predicted HoverNet results (in Json format)
    json_files = set(get_names(os.path.join(args.root_dir, 'data', 'tmp_hov', 'json'), '.json'))
    if args.create_GT:
        npy_files = set(get_names(os.path.join(args.root_dir, 'data', 'tmp_hov', 'npy'), '.npy'))

    # Move Json file to corresponding subset
    for folder_name, split_files in zip(['train', 'validation', 'test'], [tr_files, val_files, ts_files]):
    #for folder_name, split_files in zip(['train', 'validation'], [tr_files, val_files]):

        # Json
        create_dir(os.path.join(args.root_dir, 'data', folder_name, 'json'+args.data_suffix))
        for file in json_files.intersection(split_files):
            shutil.copy(
                os.path.join(args.root_dir, 'data', 'tmp_hov', 'json', file+'.json'),
                os.path.join(args.root_dir, 'data', folder_name, 'json'+args.data_suffix)
            )

        # GT GNN (npy)
        if args.create_GT:
            create_dir(os.path.join(args.root_dir, 'data', folder_name, 'graphs', 'GT', 'npy'+args.data_suffix))
            for file in json_files.intersection(split_files):
                shutil.copy(
                    os.path.join(args.root_dir, 'data', 'tmp_hov', 'npy', file+'.npy'),
                    os.path.join(args.root_dir, 'data', folder_name, 'graphs', 'GT', 'npy'+args.data_suffix)
                )
                shutil.copy(
                    os.path.join(args.root_dir, 'data', 'tmp_hov', 'npy', file + '_np.npy'),
                    os.path.join(args.root_dir, 'data', folder_name, 'graphs', 'GT', 'npy' + args.data_suffix)
                )


    # Convert format
    for split in ['train', 'validation', 'test']:

        logger.info(f'Parsing {split} split')

        # JSON to PNG, .class.csv and .prob.csv
        # logger.info('   From json to pngcsv.')
        newargs = Namespace(
            json_dir=os.path.join(args.root_dir, 'data', split, 'json'+args.data_suffix),
            png_dir=os.path.join(args.root_dir, 'data', split, 'png_hov'+args.data_suffix),
            csv_dir=os.path.join(args.root_dir, 'data', split, 'csv_hov'+args.data_suffix),
            num_classes=args.num_classes,
            num_workers=args.num_workers,
            class_csv = False
        )
        hovernet2pngcsv_main(newargs)

    return


def run_hov_infer_pipe(args: Namespace, logger: Logger) -> None:
    """
    Predicts cell contours in json format using HoverNet.

    This function performs the following steps:
        1. Performs inference using the trained Hovernet model on the input images.
        2. Saves the predicted cell contours in json format.

    :param args: The arguments for the Hovernet inference pipeline.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    # Variables
    model_path = os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet'+args.model_suffix, '01', 'net_epoch=50.tar')
    type_info_path = os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet'+args.model_suffix, 'type_info.json')
    if args.model_path is not None:
        model_path = args.model_path
        components_path = model_path.split(os.sep)
        type_info_path =os.path.join(components_path[:-2])

    input_dir = args.input_dir
    if args.input_dir is None:
        input_dir = os.path.join(args.root_dir, 'data', 'orig')

    output_dir = args.output_dir
    if args.output_dir is None:
        output_dir = os.path.join(args.root_dir, 'data', 'tmp_hov')

    # Inference HoverNet
    logger.info('Starting inference with h='+str(args.h)+' and k='+str(args.k)+'.')
    newargs = {
        'nr_types': str(args.num_classes + 1) if not args.stroma_mask else str(args.num_classes),
        'type_info_path': type_info_path,
        'gpu': args.gpu,
        'nr_inference_workers': args.num_workers,
        'model_path': model_path,
        'batch_size': '10',
        'shape': '256',
        'nr_post_proc_workers': args.num_workers,
        'model_mode': 'fast',
        'stroma_mask': args.stroma_mask,
        # 'stroma_model_path': ...,
        'h': args.h,
        'k': args.k,
        'help': False
    }
    newsubargs = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'create_gt': args.create_GT,
        'draw_dot': False,
        'save_qupath': False,
        'save_raw_map': False,
        'mem_usage': '0.2'
    }

    hov_infer(newargs, newsubargs, 'tile')

    # If "data" folder of the root directory, move files to the corresponding subset
    if args.input_dir is None:
        run_hov_postproc_pipe(args, logger)

    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--data-suffix', type=str, default="", help="Suffix of the folders to save the result.")
    parser.add_argument('--input-dir', type=str, help='Folder containing patches to process.')
    parser.add_argument('--output-dir', type=str, help='Folder where to save results. Additional subfolder will be created.')

    parser.add_argument('--model-path', type=str, help='Path to Hovernet weights to make the inference.')
    parser.add_argument('--model-suffix', type=str, default="", help='Suffix of the model.')

    parser.add_argument('--format', type=str, default='geojson', help="Format of the input GT.", choices=['geojson', 'pngcsv'])
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--stroma-mask', action='store_true', help="If enabled, the stroma initial classification (before GNN) will be done with a specific network.")
    # parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')
    parser.add_argument('--h', type=float, default=0.5)
    parser.add_argument('--k', type=float, default=0.4)

    parser.add_argument('--create-GT', action='store_true', help="If enabled, used to create the GT to be used for the GNN.")
    # parser.add_argument('--GT-dir', type=str, default=None, help="Folder where to save the created GT to be used for the GNN.")

    return parser


def main():
    # Arguments
    parser = _create_parser()
    args = parser.parse_args()
    if args.model_suffix != "":
        args.model_suffix = '_'+args.model_suffix
    if args.data_suffix != "":
        args.data_suffix = '_'+args.data_suffix

    # Logger
    logger = logging.getLogger('train_pipe')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # HoverNet inference
    logger.info('Starting Hovernet inference pipeline.')
    run_hov_infer_pipe(args, logger)
    logger.info('Finished Hovernet inference pipeline.')

if __name__ == '__main__':
    main()