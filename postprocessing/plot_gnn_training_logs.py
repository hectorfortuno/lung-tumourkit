import argparse
import pandas as pd
import numpy as np
import os
import altair as alt
import json
import matplotlib.pyplot as plt

from argparse import Namespace
from logging import Logger

from typing import List, Dict


def get_best_configuration(args: Namespace) -> None:
    """
    Gets the best configuration from training based on the F1 score.

    :param args: The arguments for setting the best configuration.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """

    if args.best_arch == 'GCN':
        save_file = os.path.join(args.root_dir, 'gnn_logs', 'gcn_results' + args.model_suffix + '.csv')
    elif args.best_arch == 'ATT':
        save_file = os.path.join(args.root_dir, 'gnn_logs', 'gat_results' + args.model_suffix + '.csv')
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
    if args.num_classes == 2:
        best_conf = gnn_results.sort_values(by='F1 Score', ascending=False).iloc[0]
    else:
        best_conf = gnn_results.sort_values(by='Weighted F1', ascending=False).iloc[0]

    best_num_layers = str(best_conf['NUM_LAYERS'])
    best_dropout = str(best_conf['DROPOUT'])
    best_norm_type = str(best_conf['NORM_TYPE'])

    config = args.best_arch+"_"+best_num_layers+"_"+best_dropout+"_"+best_norm_type

    return config

def plot_vis(values: Dict, metrics_names: List, path: str) -> None:

    name = {'loss': 'Cross Entropy Loss', 'ece': 'ECE',
             'weighted_f1': 'Weighted F1', 'macro_f1': 'Macro F1', 'micro_f1': 'Micro F1'}

    for metric_name in metrics_names:

        train_metric_values = values[metric_name]["train"]
        valid_metric_values = values[metric_name]["valid"]

        plt.plot(train_metric_values, label="Train")
        plt.plot(valid_metric_values, label="Valid")
        plt.xlabel("Epochs"); plt.ylabel(name[metric_name])
        plt.title(name[metric_name]); plt.legend()

        plt.savefig(os.path.join(path, metric_name+'.png'), bbox_inches="tight")
        plt.close()


def get_metrics_names(metrics):

    # Eliminate repetitions because "train" or "valid"
    metrics_names = list(metrics.keys())
    metrics_names = np.unique([metric_name.split("-")[1] for metric_name in metrics_names])

    return list(metrics_names)


def get_metric_values(values):

    # Metrics names
    metrics_names = get_metrics_names(values["1"])      # values of the "first" epoch

    # Change dictionary format
    metrics_values = {metric_name: {"train": [], "valid": []} for metric_name in metrics_names}
    for _, metrics_epoch in values.items():

        for metric_name in metrics_names:

            train_value = metrics_epoch["train-"+metric_name]
            valid_value = metrics_epoch["valid-"+metric_name]

            metrics_values[metric_name]["train"].append(train_value)
            metrics_values[metric_name]["valid"].append(valid_value)

    return metrics_values, metrics_names


def _create_parser():
    parser = argparse.ArgumentParser('Plot logs in a better way.')
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to load data and models.')
    parser.add_argument('--orig-path', type=str, required=True, help='Path to the folder where the logs obtained during training are obtained.')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to folder where to save plots.')

    parser.add_argument('--model-suffix', type=str, default="", help='Suffix of the model.')

    parser.add_argument('--best-config', action='store_true', help="If enabled, the best configuration is the one to be plotted. Otherwise, 'config' has to be provided.")
    parser.add_argument('--best-arch', type=str, help='Best architecture (convolutional, attention, ...) when training GNNs', required=True, choices=['GCN', 'ATT', 'TRANSFORMER', 'RESIDUAL', 'INITIAL_RESIDUAL'])
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')

    parser.add_argument('--config', type=str, default="", help="Configuration name to be plotted.")

    return parser

def main():
    parser = _create_parser()
    args = parser.parse_args()
    if args.model_suffix != "":
        args.model_suffix = '_' + args.model_suffix

    # Configuration to plot
    config = args.config
    if args.best_config:
        config = get_best_configuration(args)

    args.output_dir = os.path.join(args.output_dir, args.model_suffix, config)
    os.makedirs(args.output_dir, exist_ok=True)

    f = open(os.path.join(args.orig_path, config+'.json'), "r")
    values = json.loads(f.read())

    values, metrics_names = get_metric_values(values)
    plot_vis(values, metrics_names, args.output_dir)

if __name__ == '__main__':
    main()