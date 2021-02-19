import torch
import torch.nn.functional as F
import numpy as np

from collections import Counter

import torch_geometric
from torch_geometric.utils import to_dense_batch, to_dense_adj
from rdkit import Chem as Chem
from rdkit.Chem import Draw

import os
import os.path as osp
import shutil
import pickle

from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)

def to_one_hot(data):
    data.x = F.one_hot(data.x, num_classes=28).squeeze(1).to(torch.float)
    return data

def dataset_statistic(dataset):
    total_num_nodes = 0
    molecule_counter = Counter()
    for data in dataset:
        molecule_counter.update(data.x.argmax(-1).tolist())
        total_num_nodes += data.x.shape[0]
    avg_num_nodes = int(np.ceil(total_num_nodes / len(dataset)))
    return avg_num_nodes


idx_to_atom = {0:6, 1:8, 2:7, 3:9, 4:6, 5:16, 6:17, 7:8, 8:7, 
               9:35, 10:7, 11:7, 12:7, 13:7,
               14:16, 15:53, 16:15, 17:8, 18:7, 19:8, 20:16, 21:15, 22:15, 23:6, 24:15, 25:16, 26:6, 27:15}

def mol_from_graphs(node_list, adjacency_matrix):
    # Create empty but editable mol object
    mol = Chem.RWMol()
    
    node_to_idx = {}
    for i in range(len(node_list)):
        atom_num = idx_to_atom[int(node_list[i])]
        a = Chem.Atom(atom_num)
        mol_idx = mol.AddAtom(a)
        node_to_idx[i] = mol_idx
    
    adjacency_matrix = adjacency_matrix.numpy()
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            if iy <= ix:
                continue
            if bond == 0:
                continue
            if bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
            elif bond == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
            elif bond == 4:
                bond_type = Chem.rdchem.BondType.AROMATIC

            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
    # Convert to Mol object
    mol = mol.GetMol()
    # Sanitize the molecules
    try:
        Chem.SanitizeMol(mol)
    except:
        # print(node_list)
        # print(Chem.MolToSmiles(mol))
        return None
    
    return mol

class ZINC(InMemoryDataset):
    r"""The ZINC dataset from the `"Grammar Variational Autoencoder"
    <https://arxiv.org/abs/1703.01925>`_ paper, containing about 250,000
    molecular graphs with up to 38 heavy atoms.
    The task is to regress a molecular property known as the constrained
    solubility.

    Args:
        root (string): Root directory where the dataset should be saved.
        subset (boolean, optional): If set to :obj:`True`, will only load a
            subset of the dataset (13,000 molecular graphs), following the
            `"Benchmarking Graph Neural Networks"
            <https://arxiv.org/abs/2003.00982>`_ paper. (default: :obj:`False`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')

    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        assert split in ['train', 'val', 'test']
        super(ZINC, self).__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
        os.unlink(path)

        for split in ['train', 'val', 'test']:
            download_url(self.split_url.format(split), self.raw_dir)

    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = range(len(mols))

            if self.subset:
                with open(osp.join(self.raw_dir, f'{split}.index'), 'r') as f:
                    indices = [int(x) for x in f.read()[:-1].split(',')]

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                mol = mols[idx]

                x = mol['atom_type'].to(torch.long).view(-1, 1)
                y = mol['logP_SA_cycle_normalized'].to(torch.float)

                adj = mol['bond_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

