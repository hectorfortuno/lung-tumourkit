"""
Module to create graph from nodes.

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
from itertools import tee
from typing import Tuple, List, Optional, Callable, Any
import os
import numpy as np

from torch_geometric.utils import add_self_loops

# import sys
# sys.path.append(r"../")
from utils.preprocessing import get_names
from utils.nearest import generate_tree
from utils.classification import fit_column_normalizer
from utils.read_nodes import read_node_matrix

import torch
from torch.utils.data import Dataset
import dgl
from sklearn.preprocessing import Normalizer

from torch_geometric.data import Data


class GraphDataset(Dataset):
    """
    Torch Dataset to load graphs from .nodes.csv files.

    Generated graph is in DGL format, with node attributes in .ndata
    and edge attributes in .edata.

    Graph edges are generated on the fly.
    """
    def __init__(
            self, node_dir: str, max_dist: float, max_degree: int,
            files: Optional[List[str]] = None,
            transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            column_normalize: Optional[bool] = False,
            row_normalize: Optional[bool] = False,
            normalizers: Optional[Tuple[Any]] = None,
            return_names: Optional[bool] = False,
            is_inference: Optional[bool] = False,
            remove_prior: Optional[bool] = False,
            remove_area: Optional[bool] = False,
            remove_perimeter: Optional[bool] = False,
            remove_std: Optional[bool] = False,
            remove_hist: Optional[bool] = False,
            remove_coords: Optional[bool] = False,
            enable_background: Optional[bool] = False,
            stroma_mask: Optional[bool] = False,
            stroma_label: Optional[int] = -1,
            ):
        """
        node_dir: Path to .nodes.csv files.
        max_dist: Maximum distance to consider two nodes as neighbours.
        max_degree: Maximum degree for each node.
        files: List of names to include in the dataset. If None all names are included.
        column_normalize: Whether to subtract mean and divide by standard deviation each feature.
        row_normalize: Whether to apply row normalization. Reference: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/normalize_features.html#NormalizeFeatures
        normalizers: sklearn tuple of objects with transform method.
        """

        super().__init__()

        self.node_dir = node_dir
        if files is not None:
            self.node_names = set(get_names(node_dir, '.nodes.csv'))
            self.node_names = sorted(list(self.node_names.intersection(set(files))))
        else:
            self.node_names = sorted(get_names(node_dir, '.nodes.csv'))
        self.max_dist = max_dist
        self.max_degree = max_degree
        self.transform = transform
        self.column_normalize = column_normalize
        self.row_normalize = row_normalize
        self.normalizers = normalizers
        self.remove_prior = remove_prior
        self.remove_area = remove_area
        self.remove_perimeter = remove_perimeter
        self.remove_std = remove_std
        self.remove_hist = remove_hist
        self.remove_coords = remove_coords
        self.return_names = return_names
        self.is_inference = is_inference
        self.enable_background = enable_background
        self.stroma_mask = stroma_mask
        self.stroma_label = stroma_label
        self.initialize_normalizers()


    def __getitem__(self, idx):

        # Get file information
        file_name = self.node_names[idx] + '.nodes.csv'
        tmp = read_node_matrix(
            os.path.join(self.node_dir, file_name), return_coordinates=True, return_class=not self.is_inference,
            remove_prior=self.remove_prior, remove_area=self.remove_area, remove_perimeter=self.remove_perimeter, 
            remove_std=self.remove_std, remove_hist=self.remove_hist, remove_coords=self.remove_coords,
            enable_background=self.enable_background, stroma_mask=self.stroma_mask, stroma_label=self.stroma_label
            )

        X, y, xx, yy = tmp

        # Transform values
        if self.column_normalize:
            X = self.col_sc.transform(X)
        if self.row_normalize:
            X = self.row_sc.transform(X)
        if self.normalizers is not None:
            for normalizer in self.normalizers:
                X = normalizer.transform(X)
        if self.transform is not None:
            X = self.transform(X)

        X = torch.tensor(X, dtype=torch.float32)

        # Create graph (features of nodes and edges)
        source, dest, dists = GraphDataset.create_edges(xx, yy, self.max_degree, self.max_dist)
        edge_index = torch.tensor([source, dest])
        g = Data(X, edge_index=edge_index)
        if not self.is_inference:
            g.y = torch.tensor(y, dtype=torch.long)
        g.dist = torch.tensor(dists, dtype=torch.float32).reshape((-1, 1))

        # Add self loop
        g.edge_index, _ = add_self_loops(g.edge_index)

        if self.return_names:
            return g, file_name

        return g


    def __len__(self):
        return len(self.node_names)


    @staticmethod
    def create_edges(
            xx: List[float], yy: List[float],
            max_degree: int, threshold: float
            ) -> Tuple[List[int], List, List]:
        """
        Creates edges between nearby nodes.

        xx: X coordinates of nodes.
        yy: Y coordinates of nodes.
        max_degree: Maximum degree for each node.
        threshold: Maximum distance to look at.

        Returns
        source: List of source nodes id.
        dest: List of destination nodes id.
        distances: Distances between nodes in edges.
        """

        # Create tree
        tree = generate_tree(zip(xx, yy))

        # Add edges to the tree
        source, dest, distances = [], [], []
        for i, (x, y) in enumerate(zip(xx, yy)):

            # Related nodes
            dists, idx = tree.query((x, y), k=max_degree, distance_upper_bound=threshold)
            tmp = list(filter(lambda x: x[0] > 1e-10 and x[1] < len(xx), zip(dists, idx)))
            tmp1, tmp2 = tee(tmp)
            dists, idx = list(x[0] for x in tmp1), list(x[1] for x in tmp2)

            # Edges information
            source.extend([i for _ in range(len(idx))])         # node with index "i" in all the cases
            dest.extend(idx)
            distances.extend(dists)

        return source, dest, distances


    def initialize_normalizers(self):
        """
        Fits normalizers for later use and also checks they contain transform method.
        """
        if self.normalizers is not None:
            for normalizer in self.normalizers:
                assert callable(getattr(normalizer, "transform", None)), \
                    'Normalizers provided must have transform method.'
        if self.column_normalize:
            self.col_sc = fit_column_normalizer(self.node_dir, self.node_names, remove_area=self.remove_area, remove_perimeter=self.remove_perimeter, 
                                                remove_std=self.remove_std, remove_hist=self.remove_hist, remove_prior=self.remove_prior, 
                                                remove_coords=self.remove_coords, enable_background=self.enable_background)
            assert callable(getattr(self.col_sc, "transform", None)), \
                'Error loading column normalizer.'
        if self.row_normalize:
            self.row_sc = Normalizer(norm='l1')
            assert callable(getattr(self.row_sc, "transform", None)), \
                'Error loading row normalizer.'


    def get_normalizers(self) -> Tuple[Any]:
        """
        Returns a tuple with all the normalizers in the order they are used.
        """
        if self.column_normalize and self.row_normalize and self.normalizers is not None:
            return [self.col_sc, self.row_sc, *self.normalizers]
        if self.column_normalize and self.row_normalize:
            return [self.col_sc, self.row_sc]
        if self.column_normalize and self.normalizers is not None:
            return [self.col_sc, *self.normalizers]
        if self.row_normalize and self.normalizers is not None:
            return [self.row_sc, *self.normalizers]
        if self.column_normalize:
            return [self.col_sc]
        if self.row_normalize and self.normalizers is not None:
            return [self.row_sc, *self.normalizers]
        if self.normalizers is not None:
            return self.normalizers
