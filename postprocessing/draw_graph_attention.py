"""
Draws the graph into an image.
Input format: PNG / CSV
Output format: PNG

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
from tqdm import tqdm
import os
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor
import dgl
from dgl import DGLHeteroGraph
import networkx as nx
import torch_geometric
from torch_geometric.utils import remove_self_loops

import sys, os
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import get_names
from preprocessing import pngcsv2graph_main
from classification.read_graph import GraphDataset
from utils.read_nodes import read_node_matrix


def get_colors(type_info: Dict[str, Tuple[str, List[int]]]) -> List[str]:
    """
    Retrieves a list of hexadecimal colors from a dictionary of RGB tuples.

    :param type_info: A dictionary containing type information as keys and RGB tuples as values.
    :type type_info: Dict[str, Tuple[str, List[int]]]
    
    :return: A list of hexadecimal colors converted from the RGB tuples.
    :rtype: List[str]
    """
    def to_hex(rgb_tuple: Tuple[int, int, int]) -> str:
        """
        Converts an RGB tuple to a hexadecimal color representation.

        :param rgb_tuple: A tuple containing three integer values representing RGB channels.
        :type rgb_tuple: Tuple[int, int, int]

        :return: A string representing the hexadecimal color code.
        :rtype: str
        """
        hex_values = [format(value, '02x') for value in rgb_tuple]
        return '#' + ''.join(hex_values)
    return [to_hex(rgb_tuple) for name, rgb_tuple in type_info.values()]


def draw_graph(
        orig: np.ndarray,
        graph: DGLHeteroGraph,
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        attention_edges: List[np.ndarray],
        attention_weights: List[np.ndarray],
        type_info: Dict[str, Tuple[str, List[int]]],
        save_path: str
        ) -> np.ndarray:
    """
    Draws a graph into an image using Matplotlib and NetworkX.

    :param orig: The original data used to construct the graph.
    :type orig: np.ndarray

    :param graph: The graph object to be visualized.
    :type graph: DGLHeteroGraph

    :param x: Node features used for plotting.
    :type x: np.ndarray

    :param y: Node coordinates used for plotting.
    :type y: np.ndarray

    :param labels: Node labels used for visualization.
    :type labels: np.ndarray

    :param type_info: A dictionary containing type information as keys and RGB tuples as values.
    :type type_info: Dict[str, Tuple[str, List[int]]]

    :param save_path: The file path to save the resulting graph visualization.
    :type save_path: str

    :return: The image of the graph.
    :rtype: np.ndarray
    """

    # Convert to NetworkX graph
    # gx = dgl.to_networkx(graph)
    gx = torch_geometric.utils.to_networkx(graph, to_undirected=False)

    # Add attention weights
    list_weighted_edges = list((e[0],e[1],w) for e, w in zip(attention_edges, attention_weights))
    gx.add_weighted_edges_from(list_weighted_edges)
    gx.remove_edges_from(nx.selfloop_edges(gx))

    # Center position of the cells
    pos = {k: (y[k], x[k]) for k in range(len(x))}

    # Colors
    colors = get_colors(type_info)
    cols = [colors[label + 1] for label in labels]

    # Figure
    fig = plt.figure(frameon=True, figsize=(10,10))
    fig.tight_layout()

    ax = plt.Axes(fig, [0., 0., 1., 1.]); ax.set_axis_off(); fig.add_axes(ax)

    _, weights = zip(*nx.get_edge_attributes(gx, 'weight').items())
    weights = np.array(weights)[:,0]
    nx.draw(gx, pos=pos, node_color=cols, edge_color=weights, connectionstyle='arc3, rad=0.1', edge_cmap=plt.cm.coolwarm,
            arrows=True, arrowstyle='->', arrowsize=5, with_labels=False, node_size=20, width=1.25, ax=ax)  # 1.5

    norm = mpl.colors.Normalize(vmin=np.min(weights), vmax=np.max(weights))
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.coolwarm), ax=ax)

    ax.imshow(orig, aspect='auto')
    fig.savefig(save_path, bbox_inches='tight')

    return orig


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-dir', type=str, help='Path to base images folder. Must be in .png format.')
    parser.add_argument('--png-dir', type=str, help='Path to folder with png of the labels.')
    parser.add_argument('--csv-dir', type=str, help='Path to folder with csv of the labels.')
    parser.add_argument('--output-dir', type=str, help='Path to folder where to save results.')
    parser.add_argument('--attention-dir', type=str, help='Path to folder with the attention weights of the images.')
    parser.add_argument('--type-info', type=str, help='Path to type_info.json.')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-degree', type=int, default=10, help='Maximum degree allowed for each node.')
    parser.add_argument('--max-distance', type=int, default=200, help='Maximum allowed distance between nodes, in pixels.')
    return parser


def main_subthread(
        args: Namespace,
        name: str,
        graph_dataset: GraphDataset,
        type_info: Dict[str, Tuple[str, List[int]]],
        k: int,
        pbar: tqdm,
        ):

    # Original image
    orig = cv2.imread(os.path.join(args.orig_dir, name + '.png'), cv2.IMREAD_COLOR)[:, :, ::-1]

    # Graph information
    X, y, xx, yy = read_node_matrix(
        os.path.join(args.output_dir, 'graphs', name + '.nodes.csv'), return_coordinates=True, return_class=True,
        remove_prior=False, remove_morph=False
        )
    graph = graph_dataset[k]
    graph.edge_index, _ = remove_self_loops(graph.edge_index)

    # Weights information of each layer
    nums_layers = len([att_name for att_name in os.listdir(args.attention_dir) if name in att_name])
    for n in range(nums_layers):
        attention_info = np.load(os.path.join(args.attention_dir, name+'_'+str(n)+'.npz'))
        attention_edges, attention_weights = attention_info['edges'], attention_info['weights']
        attention_edges = np.transpose(attention_edges)

        # Draw graph
        draw_graph(orig, graph, xx, yy, y, attention_edges, attention_weights,
                   type_info, os.path.join(args.output_dir, name+'_'+str(n)+'.graph-overlay.png'))

    """
    # Convert to NetworkX graph
    nx_graph = dgl.to_networkx(graph)
    nodes = [(i, {'target': str(y[i])}) for i in nx_graph.nodes()]
    nx_graph.add_nodes_from(nodes)
    nx.write_gml(nx_graph, os.path.join(args.output_dir, 'graphs', name + '.gml'))
    """

    pbar.update(1)


def main_with_args(args: Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    # Obtain graph information
    newargs = Namespace(
        png_dir=args.png_dir,
        csv_dir=args.csv_dir,
        orig_dir=args.orig_dir,
        output_path=os.path.join(args.output_dir, 'graphs'),
        num_workers=args.num_workers
    )
    pngcsv2graph_main(newargs)

    # Dataset and general information
    graph_dataset = GraphDataset(
        node_dir = os.path.join(args.output_dir, 'graphs'),
        max_degree = args.max_degree,
        max_dist = args.max_distance
    )
    names = sorted(get_names(args.orig_dir, '.png'))
    with open(args.type_info, 'r') as f:
        type_info = json.load(f)

    # Draw graph
    pbar = tqdm(total=len(names))
    plt.switch_backend('Agg')
    if args.num_workers > 0:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for k, name in enumerate(names):
                executor.submit(main_subthread, args, name, graph_dataset, type_info, k, pbar)
    else:
        for k, name in enumerate(names):
            main_subthread(args, name, graph_dataset, type_info, k, pbar)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)

if __name__ == '__main__':
    main()
