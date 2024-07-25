"""
Script to train and save several GNN configurations.

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
import math
from typing import Optional, Tuple, Dict, List, Any
from sklearn.preprocessing import Normalizer
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch.optim import Optimizer
import dgl
from dgl.dataloading import GraphDataLoader
from torch_geometric.loader import DataLoader

from .read_graph import GraphDataset
# from .models.gcn import GCN
from .models_pytorch_geometric.gcn import GCN
from .models_pytorch_geometric.gtn import GTN
from .models_pytorch_geometric.grn import GInitResN, GResN
from .models.hgao import HardGAT
from .models.gat import GAT

import argparse
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import warnings
import os
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
import random
import copy

import sys
# sys.path.append(r"../")
from utils.preprocessing import parse_path, create_dir
from utils.classification import metrics_from_predictions

warnings.filterwarnings('ignore')


def check_manual_seed(seed):
    """ If manual seed is not specified, choose a
    random one and communicate it to the user.

    Args:
        seed: seed to check

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ia.random.seed(seed)

    #print("Using manual seed: {seed}".format(seed=seed))
    return

def seed_worker(worker_id):
    print("Using Dataloader seed setting")
    worker_seed = 42
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    #torch.cuda.manual_seed_all(worker_seed)
    dgl.seed(worker_seed)
    dgl.random.seed(worker_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_metrics_names(num_classes, phase=None):

    if num_classes == 2:
        values_names = ["acc", "f1_score", "roc_auc", "perc_err", "ece"]
    else:
        values_names = ["micro_f1", "macro_f1", "weighted_f1", "ece"]

    if phase is not None:
        values_names = [phase+"-"+value_name for value_name in values_names]

    return values_names


def get_serializable_values(engine_name, num_classes, values):
    
    values_names = get_metrics_names(num_classes)
    
    log_dict = {}
    for value_name, value in zip(values_names, values):
        value_name = "%s-%s" % (engine_name, value_name)
        log_dict[value_name] = value
    
    return log_dict


def evaluate(
        loader: GraphDataLoader,
        model: nn.Module,
        device: str,
        log_info: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        log_suffix: Optional[str] = None,
        num_classes: Optional[int] = 2,
        enable_background: Optional[bool] = False,
        loss_class = None,
        ) -> List:
    """
    Evaluates model in loader.
    Logs to tensorboard with suffix log_suffix.
    Returns the model in evaluation mode.
    """

    # Logs
    if log_info is not None:
        writer = log_info["writer"]
        json_file = log_info["json_file"]
        with open(log_info["json_file"]) as json_file:
            json_data = json.load(json_file)

    # Inference mode
    model.eval()

    # Initialize variables
    preds, labels, probs = np.array([]).reshape(0, 1), np.array([]).reshape(0, 1), np.array([]).reshape(0, 1 if num_classes == 2 else num_classes)
    # preds_bkgr, labels_bkgr, probs_bkgr = np.array([]).reshape(0, 1), np.array([]).reshape(0, 1), np.array([]).reshape(0, 1)

    running_loss = 0

    # Evaluate on the whole data
    for g in loader:

        g = g.to(device)

        # Data
        features = g.x
        label = g.y

        # Forward pass
        logits = model(features, g.edge_index)

        # Loss function
        if loss_class is not None:
            loss = loss_class(logits, label)
        else:
            loss = F.cross_entropy(logits, label)

        """
        # Prova penalty
        l1 = 0; lambda_l1 = 0.01
        for w in model.parameters():
            l1 += w.abs().sum()
        loss += lambda_l1 * l1
        """

        running_loss += loss.item()

        # Classification
        pred = logits.argmax(1).detach().cpu().numpy().reshape(-1, 1)
        label = label.detach().cpu().numpy().reshape(-1, 1)
        if num_classes == 2:
            prob = F.softmax(logits, dim=1).detach().cpu().numpy()[:, 1].reshape(-1, 1)
        else:
            prob = F.softmax(logits, dim=1).detach().cpu().numpy()

        preds = np.vstack((preds, pred))
        probs = np.vstack((probs, prob))
        labels = np.vstack((labels, label))

        """
        # Background classification
        if enable_background:
            pred_bkgr = logits_bkgr.argmax(1).detach().cpu().numpy().reshape(-1, 1)
            prob_bkgr = F.softmax(logits_bkgr, dim=1).detach().cpu().numpy()[:, 1].reshape(-1, 1)
            label_bkgr = label_bkgr.detach().cpu().numpy().reshape(-1, 1)

            preds_bkgr = np.vstack((preds_bkgr, pred_bkgr))
            probs_bkgr = np.vstack((probs_bkgr, prob_bkgr))
            labels_bkgr = np.vstack((labels_bkgr, label_bkgr))
        """

    """
    # Metrics from background
    if enable_background:

        # Metrics
        acc_bkgr, f1_bkgr, auc_bkgr, perc_err_bkgr, ece_bkgr = metrics_from_predictions(labels_bkgr, preds_bkgr, probs_bkgr, 2)

        # Tensorboard
        if log_info is not None:
            writer.add_scalar('Accuracy-bkgr/' + log_suffix, acc_bkgr, epoch)
            writer.add_scalar('F1-bkgr/' + log_suffix, f1_bkgr, epoch)
            writer.add_scalar('ROC_AUC-bkgr/' + log_suffix, auc_bkgr, epoch)
            writer.add_scalar('ECE-bkgr/' + log_suffix, ece_bkgr, epoch)
            writer.add_scalar('Percentage Error-bkgr/' + log_suffix, perc_err_bkgr, epoch)
    """

    # Compute metrics from the predictions
    if num_classes == 2:

        # Compute metrics
        metrics = metrics_from_predictions(labels, preds, probs, 2)
        acc, f1, auc, perc_err, ece = metrics

        # Tensorboard
        if log_info is not None:
            assert (log_suffix is not None and epoch is not None)
            writer.add_scalar('Accuracy/' + log_suffix, acc, epoch)
            writer.add_scalar('F1/' + log_suffix, f1, epoch)
            writer.add_scalar('ROC_AUC/' + log_suffix, auc, epoch)
            writer.add_scalar('Percentage Error/' + log_suffix, perc_err, epoch)
            writer.add_scalar('ECE/' + log_suffix, ece, epoch)

        # return [f1, acc, auc, perc_err, ece]

    else:

        # Compute metrics
        metrics = metrics_from_predictions(labels, preds, probs, num_classes)
        micro, macro, weighted, ece = metrics

        # Tensorboard
        if log_info is not None:
            writer.add_scalar('Accuracy/' + log_suffix, micro, epoch)
            writer.add_scalar('Macro F1/' + log_suffix, macro, epoch)
            writer.add_scalar('Weighted F1/' + log_suffix, weighted, epoch)
            writer.add_scalar('ECE/' + log_suffix, ece, epoch)
        
        # return [micro, macro, weighted, ece]
    
    
    # Json (add validation information to train values)
    if log_info is not None:

        stat_dict = get_serializable_values("valid", num_classes, metrics)
        stat_dict.update({'valid-loss': running_loss / len(loader)})

        old_stat_dict = json_data[str(epoch)]
        stat_dict.update(old_stat_dict)

        current_epoch_dict = {str(epoch): stat_dict}
        json_data.update(current_epoch_dict)

        with open(log_info["json_file"], "w") as json_file:
            json.dump(json_data, json_file)

    return metrics


class ScalarMovingAverage():
    """Calculate the running average for all scalar output."""

    def __init__(self, num_classes, alpha=0.95, phase="train"):
        super().__init__()
        self.alpha = alpha
        self.metrics_names = get_metrics_names(num_classes, phase)
        self.tracking_dict = {}

    def update(self, metrics):
        
        for metric_name, current_value in zip(self.metrics_names, metrics):
            if metric_name in self.tracking_dict:
                old_value = self.tracking_dict[metric_name]
                # calculate the exponential moving average
                new_value = (
                    old_value * self.alpha + (1.0 - self.alpha) * current_value
                )
                self.tracking_dict[metric_name] = new_value
            else:  # init for variable which appear for the first time
                new_value = current_value
                self.tracking_dict[metric_name] = new_value

        return


def train_one_iter(
        tr_loader: GraphDataLoader,
        model: nn.Module,
        device: str,
        optimizer: Optimizer,
        epoch: int,
        log_info: Dict[str, Any],
        num_classes: int,
        enable_background: Optional[bool] = False,
        loss_class=None,
        ) -> None:
    """
    Trains for one iteration, as the name says.
    """

    # Logs
    writer = log_info["writer"]
    with open(log_info["json_file"]) as json_file:
        json_data = json.load(json_file)
    metrics = ScalarMovingAverage(num_classes)

    # Train mode
    model.train()

    # Train with all batches of training data
    running_loss = 0
    for step, tr_g in enumerate(tr_loader):
        tr_g = tr_g.to(device)

        # Data
        features = tr_g.x
        labels = tr_g.y
        
        # Forward step
        logits = model(features, tr_g.edge_index)
        
        """
        chunk_size = 100
        num_chunks = (labels.numel() + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, labels.numel())
            for num in labels[start_idx:end_idx]:
                if isinstance(num, torch.Tensor):
                    print(num)
                elif num == -1:
                    print(num)
        num_chunks = (logits.numel() + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, logits.numel())
            for num in logits[start_idx:end_idx]:
                if isinstance(num, torch.Tensor):
                    print(num)
                elif num == -1:
                    print(num)
        """
                    
        # Loss function
        if loss_class is not None:
            loss = loss_class(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels)

        """
        # Prova penalty
        l1 = 0; lambda_l1 = 0.1
        for w in model.parameters():
            l1 += w.abs().sum()
        loss = loss + lambda_l1 * l1
        """

        running_loss += loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics on training data
        preds = logits.argmax(1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        if num_classes == 2:

            # Metrics
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[:, 1]
            metrics_step = metrics_from_predictions(labels, preds, probs, 2)
            train_acc, train_f1, train_auc, train_perc_err, train_ece = metrics_step

            # Update
            metrics.update(metrics_step)
            
            # Tensorboard
            writer.add_scalar('Accuracy/train', train_acc, step + len(tr_loader) * epoch)
            writer.add_scalar('F1/train', train_f1, step + len(tr_loader) * epoch)
            writer.add_scalar('ROC_AUC/train', train_auc, step + len(tr_loader) * epoch)
            writer.add_scalar('ECE/train', train_ece, step + len(tr_loader) * epoch)
            writer.add_scalar('Percentage Error/train', train_perc_err, step + len(tr_loader) * epoch)
            
        else:

            # Metrics
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            metrics_step = metrics_from_predictions(labels, preds, probs, num_classes)
            train_micro, train_macro, train_weighted, train_ece = metrics_step

            # Update
            metrics.update(metrics_step)
            
            # Tensorboard
            writer.add_scalar('Accuracy/train', train_micro, step + len(tr_loader) * epoch)
            writer.add_scalar('Macro F1/train', train_macro, step + len(tr_loader) * epoch)
            writer.add_scalar('Weighted F1/train', train_weighted, step + len(tr_loader) * epoch)
            writer.add_scalar('ECE/train', train_ece, step + len(tr_loader) * epoch)

        """
        # Background classification
        if enable_background:

            # Metrics
            preds_bkgr = logits_bkgr.argmax(1).detach().cpu().numpy()
            labels_bkgr = labels_bkgr.detach().cpu().numpy()
            probs_bkgr = F.softmax(logits_bkgr, dim=1).detach().cpu().numpy()[:, 1]
            train_acc_bkgr, train_f1_bkgr, train_auc_bkgr, train_perc_err_bkgr, train_ece_bkgr = metrics_from_predictions(labels_bkgr, preds_bkgr, probs_bkgr, 2)
            
            # Tensorboard
            writer.add_scalar('Accuracy-bkgr/train', train_acc_bkgr, step + len(tr_loader) * epoch)
            writer.add_scalar('F1-bkgr/train', train_f1_bkgr, step + len(tr_loader) * epoch)
            writer.add_scalar('ROC_AUC-bkgr/train', train_auc_bkgr, step + len(tr_loader) * epoch)
            writer.add_scalar('ECE-bkgr/train', train_ece_bkgr, step + len(tr_loader) * epoch)
            writer.add_scalar('Percentage Error-bkgr/train', train_perc_err_bkgr, step + len(tr_loader) * epoch)
        """

    # Loss
    train_loss = running_loss / len(tr_loader)
    loss_dict = {'train-loss': train_loss}

    # Json
    # stat_dict = get_serializable_values("train", num_classes, metrics)
    stat_dict = metrics.tracking_dict
    stat_dict.update(loss_dict)
    current_epoch_dict = {str(epoch): stat_dict}
    json_data.update(current_epoch_dict)

    with open(log_info["json_file"], "w") as json_file:
        json.dump(json_data, json_file)


def train(
        save_dir: str,
        save_weights: bool,
        tr_loader: GraphDataLoader,
        val_loader: GraphDataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        log_info: Dict[str, Any],
        n_early: int,
        device: Optional[str] = 'cpu',
        check_iters: Optional[int] = -1,
        conf: Optional[Dict[str, Any]] = None,
        normalizers: Optional[Tuple[Normalizer]] = None,
        num_classes: Optional[int] = 2,
        enable_background: Optional[bool] = False,
        loss_class=None,
        ) -> None:
    """
    Train the model with early stopping on F1 score (weighted) or until 1000 iterations.
    """
    
    # Initial configuration
    model = model.to(device)
    n_epochs = 1000
    best_val_f1 = 0
    early_stop_rounds = 0
    
    # Train N epochs
    for epoch in range(n_epochs):

        # Train and evaluate current epoch
        train_one_iter(tr_loader, model, device, optimizer, epoch, log_info, num_classes, enable_background=enable_background,loss_class=loss_class)
        val_metrics = evaluate(val_loader, model, device, log_info, epoch, 'validation', num_classes=num_classes, enable_background=enable_background,loss_class=loss_class)
        if num_classes == 2:
            val_acc, val_f1, val_auc, val_perc_error, val_ece = val_metrics
        else:
            val_micro, val_macro, val_f1, val_ece = val_metrics
        
        # Save checkpoint
        if save_weights and check_iters != -1 and epoch % check_iters == 0:
            save_model(save_dir, model, conf, normalizers, prefix='last_')
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_rounds = 0
            if save_weights:
                save_model(save_dir, model, conf, normalizers, prefix='best_')
        elif early_stop_rounds < n_early:
            early_stop_rounds += 1
        else:
            return


def load_dataset(
        train_node_dir: str,
        val_node_dir: str,
        test_node_dir: str,
        bsize: int,
        degree: Optional[int] = 10,
        distance: Optional[int] = 200,
        remove_prior: Optional[bool] = False,
        remove_area: Optional[bool] = False,
        remove_perimeter: Optional[bool] = False,
        remove_std: Optional[bool] = False,
        remove_hist: Optional[bool] = False,
        remove_coords: Optional[bool] = False,
        enable_background: Optional[bool] = False,
        stroma_mask: Optional[bool] = False,
        stroma_label: Optional[int] = -1,
        ) -> Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
    """
    Creates Torch dataloaders for training.
    Folder structure:
    node_dir:
     - train
      - graphs
       - file1.nodes.csv
       ...
     - validation
      - graphs
       - file1.nodes.csv
       ...
     - test
      - graphs
       - file1.nodes.csv
       ...
    """

    g = torch.Generator()
    g.manual_seed(42)

    # Train dataset
    train_dataset = GraphDataset(
        node_dir=train_node_dir, remove_area=remove_area, remove_perimeter=remove_perimeter, 
        remove_std=remove_std, remove_hist=remove_hist, remove_prior=remove_prior, remove_coords=remove_coords,
        max_dist=distance, max_degree=degree, column_normalize=True,
        enable_background=enable_background, stroma_mask=stroma_mask, stroma_label=stroma_label)
    train_dataloader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, worker_init_fn=seed_worker, generator=g)

    # Validation dataset
    val_dataset = GraphDataset(
        node_dir=val_node_dir, remove_area=remove_area, remove_perimeter=remove_perimeter, 
        remove_std=remove_std, remove_hist=remove_hist, remove_prior=remove_prior, remove_coords=remove_coords,
        max_dist=distance, max_degree=degree, normalizers=train_dataset.get_normalizers(),
        enable_background=enable_background, stroma_mask=stroma_mask, stroma_label=stroma_label)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Test dataset
    test_dataset = GraphDataset(
        node_dir=test_node_dir, remove_area=remove_area, remove_perimeter=remove_perimeter, 
        remove_std=remove_std, remove_hist=remove_hist, remove_prior=remove_prior, remove_coords=remove_coords,
        max_dist=distance, max_degree=degree, normalizers=train_dataset.get_normalizers(),
        enable_background=enable_background, stroma_mask=stroma_mask, stroma_label=stroma_label)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def generate_graph_configurations() -> List[Dict[str, int]]:
    """
    Generates a grid in the search space with no more than max_confs configurations.
    Parameters changed: NUM_LAYERS, DROPOUT, NORM_TYPE
    """

    # Generate all configurations
    confs = []
    for num_degree in [50, 10, 15, 20, 30, 40]:
        for distance in [100, 200, 300, 400, 500]:
            conf = {}
            conf['DEGREE'] = int(num_degree)
            conf['DISTANCE'] = int(distance)
            confs.append(conf)

    return confs


def generate_configurations(max_confs: int, model_name: str) -> List[Dict[str, int]]:
    """
    Generates a grid in the search space with no more than max_confs configurations.
    Parameters changed: NUM_LAYERS, DROPOUT, NORM_TYPE
    """

    # Parameters values
    num_layers_confs = int(math.sqrt(max_confs / 2))
    num_dropout_confs = int(max_confs // (2 * num_layers_confs))
    assert (2 * num_layers_confs * num_dropout_confs <= max_confs)
    assert num_layers_confs <= 15, 'Too many layers'

    # Generate all configurations
    confs = []
    for num_layers in np.linspace(1, 15, num_layers_confs):
        num_layers = int(num_layers)
        for dropout in np.linspace(0, 0.9, num_dropout_confs):
            conf = {}
            conf['MODEL_NAME'] = model_name
            conf['NUM_LAYERS'] = num_layers
            conf['DROPOUT'] = dropout
            conf['NORM_TYPE'] = 'bn'
            confs.append(conf)

            conf = {}
            conf['MODEL_NAME'] = model_name
            conf['NUM_LAYERS'] = num_layers
            conf['DROPOUT'] = dropout
            conf['NORM_TYPE'] = None
            confs.append(conf)

    return confs


def load_model(conf: Dict[str, Any], num_classes: int, num_feats: int, enable_background: bool) -> nn.Module:
    """
    Available models: GCN, ATT, HATT, SAGE, BOOST
    Configuration space: NUM_LAYERS, DROPOUT, NORM_TYPE
    """
    hidden_feats = 100
    if conf['MODEL_NAME'] == 'GCN':
        return GCN(num_feats, hidden_feats, num_classes, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'], enable_background)
    if conf['MODEL_NAME'] == 'ATT' or conf['MODEL_NAME'] == 'HATT':
        num_heads = 8
        num_out_heads = 1
        heads = ([num_heads] * conf['NUM_LAYERS']) + [num_out_heads]
        if conf['MODEL_NAME'] == 'ATT':
            return GAT(num_feats, hidden_feats, num_classes, heads, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'], enable_background)
        return HardGAT(num_feats, hidden_feats, num_classes, heads, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'], enable_background)
    if conf['MODEL_NAME'] == 'TRANSFORMER':
        return GTN(num_feats, hidden_feats, num_classes, 1, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'], enable_background)
    if conf['MODEL_NAME'] == 'INITIAL_RESIDUAL':
        return GInitResN(num_feats, num_classes, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'], enable_background)
    if conf['MODEL_NAME'] == 'RESIDUAL':
        return GResN(num_feats, hidden_feats, num_classes, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'], enable_background)
    assert False, 'Model not implemented.'


def create_results_file(filename: str, num_classes: int) -> None:
    """
    Creates header of .csv result file to append results.
    filename must not contain extension.
    """
    if num_classes == 2:
        with open(filename + '.csv', 'w') as f:
            print('F1 Score,Accuracy,ROC AUC,PERC ERR,ECE,NUM_LAYERS,DROPOUT,NORM_TYPE', file=f)
    else:
        with open(filename + '.csv', 'w') as f:
            print('Micro F1,Macro F1,Weighted F1,ECE,NUM_LAYERS,DROPOUT,NORM_TYPE', file=f)


def append_results(
        filename: str,
        f1: float, acc: float, auc: float,
        num_layers: int, dropout: float, bn_type: str,
        ece: float, perc_err: Optional[float] = None
        ) -> None:
    """
    Appends result to given filename.
    filename must not contain extension.
    """
    with open(filename + '.csv', 'a') as f:
        if perc_err is not None:
            print(f1, acc, auc, perc_err, ece, num_layers, dropout, bn_type, file=f, sep=',')
        else:
            print(f1, acc, auc, ece, num_layers, dropout, bn_type, file=f, sep=',')


def name_from_conf(conf: Dict[str, Any]) -> str:
    """
    Generates a name from the configuration object.
    """
    return conf['MODEL_NAME'] + '_' + str(conf['NUM_LAYERS']) + '_' \
        + str(conf['DROPOUT']) + '_' + str(conf['NORM_TYPE'])


def save_model(
        save_dir: str,
        model: nn.Module,
        conf: Dict[str, Any],
        normalizers: Tuple[Normalizer],
        prefix: Optional[str] = ''
        ) -> None:
    """
    Save model weights and configuration file to SAVE_DIR
    """
    name = prefix + name_from_conf(conf)
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(save_dir, 'weights', name + '.pth'))
    with open(os.path.join(save_dir, 'confs', name + '.json'), 'w') as f:
        json.dump(conf, f)
    with open(os.path.join(save_dir, 'normalizers', name + '.pkl'), 'wb') as f:
        pickle.dump(normalizers, f)


def train_one_conf(
        args: Namespace,
        conf: Dict[str, Any],
        log_dir: str,
        save_weights: bool,
        save_dir: str,
        num_classes: int,
        num_feats: int,
        loss_class,

        ) -> Tuple[List[float], nn.Module, Dict[str, Any]]:

    # Logs
    writer = SummaryWriter(log_dir=os.path.join(log_dir, name_from_conf(conf)))
    json_log_file = os.path.join(args.save_dir, 'logs', name_from_conf(conf)+".json")
    with open(json_log_file, "w") as json_file:
        json.dump({}, json_file)  # create empty file
    log_info = {
        "json_file": json_log_file,
        "writer": writer,
    }

    print("  Training configuration", conf)
    check_manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True) # At least scatter_reduce_cuda does NOT have a deterministic implementation
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    train_dataloader, val_dataloader, test_dataloader = load_dataset(
        train_node_dir=args.train_node_dir,
        val_node_dir=args.validation_node_dir,
        test_node_dir=args.test_node_dir,
        bsize=args.batch_size,
        degree=args.degree,
        distance=args.distance,
        remove_prior=args.disable_prior,
        remove_area=args.disable_area,
        remove_perimeter=args.disable_perimeter,
        remove_std=args.disable_std,
        remove_hist=args.disable_hist,
        remove_coords=args.disable_coordinates,
        enable_background=args.enable_background,
        stroma_mask=args.stroma_mask,
        stroma_label=num_classes
    )

    # Model
    model = load_model(conf, num_classes, num_feats, args.enable_background)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    train(
        save_dir, save_weights, train_dataloader, val_dataloader,
        model, optimizer, log_info, args.early_stopping_rounds,
        args.device, args.checkpoint_iters, conf, train_dataloader.dataset.get_normalizers(),
        num_classes=num_classes, enable_background=args.enable_background,loss_class=loss_class
    )
    
    # Evaluate on test metrics
    test_metrics = evaluate(
        test_dataloader, model, args.device, num_classes=num_classes, enable_background=args.enable_background,loss_class=loss_class
    )
    model = model.cpu()
    if num_classes == 2:
        test_acc, test_f1, test_auc, test_perc_err, test_ece = test_metrics
        return test_f1, test_acc, test_auc, test_perc_err, test_ece, model, conf, train_dataloader
    else:
        test_micro, test_macro, test_weighted, test_ece = test_metrics
        return test_micro, test_macro, test_weighted, test_ece, model, conf, train_dataloader


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-node-dir', type=str, required=True,
                        help='Path to folder containing train folder with .nodes.csv files.')
    parser.add_argument('--validation-node-dir', type=str, required=True,
                        help='Path to folder containing validation folder with .nodes.csv files.')
    parser.add_argument('--test-node-dir', type=str, required=True,
                        help='Path to folder containing test folder with .nodes.csv files.')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='Path to save tensorboard logs.')
    parser.add_argument('--early-stopping-rounds', type=int, required=True,
                        help='Number of epochs needed to consider convergence when worsening.')
    parser.add_argument('--batch-size', type=int, required=True,
                        help='Batch size. No default.')
    parser.add_argument('--model-name', type=str, required=True, choices=['GCN', 'ATT', 'HATT', 'SAGE', 'BOOST'],
                        help='Which model to use. Options: GCN, ATT, HATT, SAGE, BOOST')
    parser.add_argument('--save-file', type=str, required=True,
                        help='Name to file where to save the results. Must not contain extension.')
    parser.add_argument('--num-confs', type=int, default=50,
                        help='Upper bound on the number of configurations to try.')
    parser.add_argument('--save-dir', type=str,
                        help='Folder to save models weights and confs.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                        help='Device to execute. Either cpu or cuda. Default: cpu.')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of processors to use. Default: 1.')
    parser.add_argument('--checkpoint-iters', type=int, default=-1,
                        help='Number of iterations at which to save model periodically while training. Set to -1 for no checkpointing. Default: -1.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--disable-prior', action='store_true', help='If True, remove hovernet probabilities from node features.')
    parser.add_argument('--disable-area', action='store_true', help='If True, remove area feature from node features.')
    parser.add_argument('--disable-perimeter', action='store_true', help='If True, remove perimeter feature from node features.')
    parser.add_argument('--disable-std', action='store_true', help='If True, remove std feature from node features.')
    parser.add_argument('--disable-hist', action='store_true', help='If True, remove histogram features from node features.')
    parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')
    return parser


def train_all_configurations_graph(args: Namespace):

    # Save model and logs
    log_dir = parse_path(args.log_dir)
    create_dir(log_dir)
    save_weights = False
    if args.save_dir is not None:
        save_weights = True
        save_dir = parse_path(args.save_dir)
        create_dir(save_dir)
        create_dir(save_dir + 'weights')
        create_dir(save_dir + 'confs')
        create_dir(save_dir + 'normalizers')
        create_dir(save_dir + 'logs')

    # Num. classes to consider
    num_classes = args.num_classes
    if args.enable_background:
        num_classes += 1

    loss_class = None
    
    # Calculate weights for balanced loss
    """labels = None
    for g in train_dataloader:
        label = g.y
        label = label.detach().cpu().numpy().reshape(-1, 1)
        if labels is None:
            labels = label
        else:
            labels = np.vstack((labels, label))
    for g in val_dataloader:
        label = g.y
        label = label.detach().cpu().numpy().reshape(-1, 1)
        labels = np.vstack((labels, label))
    for g in test_dataloader:
        label = g.y
        label = label.detach().cpu().numpy().reshape(-1, 1)
        labels = np.vstack((labels, label))    
    labels = labels.ravel()
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=torch.device('cuda:0'))
    loss_class = nn.CrossEntropyLoss(weight=weights_tensor)"""
    

    # Configurations to train
    confs = generate_configurations(args.num_confs, args.model_name)
    create_results_file(args.save_file, num_classes)

    # Features to be used
    num_feats = 20 + (1 if num_classes == 2 else num_classes)
    if args.disable_coordinates:
        num_feats -= 2
    if args.disable_area:
        num_feats -= 1
    if args.disable_perimeter:
        num_feats -= 1
    if args.disable_std:
        num_feats -= 1
    if args.disable_hist:
        num_feats -= 15
    if args.disable_prior:
        num_feats -= (1 if num_classes == 2 else num_classes)
    if num_feats == 0:
        num_feats = 1
    

    # Train all the configurations
    if args.num_workers > 0:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:

            # Train each configuration
            futures = []
            for conf in confs:
                future = executor.submit(
                    train_one_conf,
                    args, conf, log_dir, save_weights, save_dir,
                    num_classes, num_feats, loss_class
                )
                futures.append(future)
            
            # Results of each configuration to a file
            for future in futures:
                if num_classes == 2:
                    test_f1, test_acc, test_auc, test_perc_err, test_ece, model, conf, train_dataloader = future.result()
                    append_results(args.save_file, test_f1, test_acc, test_auc, conf['NUM_LAYERS'],
                                   conf['DROPOUT'], conf['NORM_TYPE'], test_ece, test_perc_err)
                else:
                    test_micro, test_macro, test_weighted, test_ece, model, conf, train_dataloader = future.result()
                    append_results(args.save_file, test_micro, test_macro, test_weighted, conf['NUM_LAYERS'],
                                   conf['DROPOUT'], conf['NORM_TYPE'], test_ece)
                
                if save_weights:
                    save_model(save_dir, model, conf, train_dataloader.dataset.get_normalizers(), 'last_')
    
    else:
        
        # Train and evaluate each configuration, save the weights of the model
        for conf in confs:
            if num_classes == 2:
                future = train_one_conf(
                    args, conf, log_dir, save_weights, save_dir,
                    num_classes, num_feats, loss_class
                )
                test_f1, test_acc, test_auc, test_perc_err, test_ece, model, conf, train_dataloader = future
                append_results(args.save_file, test_f1, test_acc, test_auc, conf['NUM_LAYERS'],
                               conf['DROPOUT'], conf['NORM_TYPE'], test_ece, test_perc_err)
            else:
                future = train_one_conf(
                    args, conf, log_dir, save_weights, save_dir,
                    num_classes, num_feats, loss_class
                )
                test_micro, test_macro, test_weighted, test_ece, model, conf, train_dataloader = future
                append_results(args.save_file, test_micro, test_macro, test_weighted, conf['NUM_LAYERS'],
                               conf['DROPOUT'], conf['NORM_TYPE'], test_ece)
            
            if save_weights:
                save_model(save_dir, model, conf, train_dataloader.dataset.get_normalizers(), 'last_')


def main_with_args(args: Namespace):

    newargs = copy.deepcopy(args)

    # Train several graph configurations to construct the graph
    if args.graph_construction:

        newargs.log_dir = os.path.join(args.log_dir, 'Graph_construction')

        # Configurations of degree and distance
        confs = generate_graph_configurations()
        for conf in confs:

            degree = int(conf['DEGREE'])
            distance = int(conf['DISTANCE'])

            newargs.degree = degree
            newargs.distance = distance

            conf_name = 'degree='+str(degree) + '_' + 'distance='+str(distance)
            newargs.save_dir = os.path.join(args.save_dir, conf_name)
            newargs.save_file = os.path.join(args.log_dir, 'Graph_construction', 'gtn_results_' + conf_name)

            train_all_configurations_graph(newargs)

    else:
        newargs.degree = 40
        newargs.distance = 400
        train_all_configurations_graph(newargs)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
