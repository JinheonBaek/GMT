import os
import time

from tqdm import tqdm, trange

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch

from utils.molecule_utils import ZINC, dataset_statistic, to_one_hot, mol_from_graphs
from models.ae import GraphMultisetTransformer

class Trainer(object):

    def __init__(self, args):

        super(Trainer, self).__init__()

        # Random Seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.args = args
        self.exp_name = self.experiment_name()

        self.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(args.gpu)
            self.args.device = 'cuda:{}'.format(args.gpu)
        else:
            self.args.device = 'cpu'

        self.load_data()

        self.logdir = "./logs"
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir, exist_ok="True")
        self.ckpt = os.path.join(self.logdir, "best_molecule.pth")

    def load_data(self):
        print("Load Dataset...")
        start = time.time()
        subset = True
        train_dataset = ZINC("./zinc", subset=subset, split='train', transform=to_one_hot)
        valid_dataset = ZINC("./zinc", subset=subset, split='val', transform=to_one_hot)
        test_dataset = ZINC("./zinc", subset=subset, split="test", transform=to_one_hot)
        n_nodes = dataset_statistic(train_dataset)
        print(n_nodes)
        print("Done. Elapsed Time: {}".format(time.time() - start))

        batch_size = 128
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        self.test_dataset = test_dataset
        # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        self.args.num_features = 28
        self.args.num_classes = 28
        self.n_nodes = n_nodes


    def load_model(self):

        if self.args.model == 'GMT':
            model = GraphMultisetTransformer(self.args, self.n_nodes)
        else:
            raise ValueError("Model Name <{}> is Unknown".format(self.args.model))

        return model

    def train(self):

        self.model = self.load_model()
        device = self.args.device
        model = self.model

        print(model)
        model = model.to(device)

        ### Hyperparameters ###
        ITER = 500 # 500
        es_patience = 50
        tol = 1e-5
        best_loss = 1000000

        optim = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()
        losses = []
        patience = es_patience
        epoch_iter = trange(0, ITER, desc='[EPOCH]', position=1)

        for epoch in epoch_iter:
            model.train()
            total_loss = 0
            
            for i, data in enumerate(tqdm(self.train_loader, desc='[Train]', position=0)):
                optim.zero_grad()
                data = data.to(device)
                out = model(data)
                target = data.x.argmax(-1) # To one-hot encoding
                loss = criterion(out, target)
                loss.backward()
                optim.step()
                losses.append(loss.item())
                desc = f"[Train] Train Loss {loss.item()}"
                epoch_iter.set_description(desc)
                epoch_iter.refresh()

            valid_acc, valid_loss = self.eval(self.val_loader)
            tqdm.write(f"[Epoch {epoch}] (Loss) Loss: {valid_loss:.2f}, Acc: {valid_acc:.2f}")

            if valid_loss < best_loss:
                torch.save(self.model.state_dict(), self.ckpt)
                patience = es_patience
                best_loss = valid_loss
            else:
                patience -= 1
                if patience == 0: break

        # Load Best Model
        self.model = self.load_model()
        self.model.load_state_dict(torch.load(self.ckpt))
        self.model = self.model.to(device)
        
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=128, shuffle=False)
        self.test(test_loader)

    def eval(self, loader):
        device = self.args.device
        valid_loss = 0
        valid_acc = 0
        total_node_num = 0
        criterion = nn.CrossEntropyLoss()

        for i, data in enumerate(tqdm(loader, desc='[Eval]', position=0)):
            self.model.eval()
            data = data.to(device)
            out = self.model(data)
            pred_logit = torch.softmax(out, dim=-1)
            target = data.x.argmax(-1)
            loss = criterion(pred_logit, target)
            pred = pred_logit.argmax(-1)
            valid_loss += loss
            valid_acc += (pred == target).sum().item()
            total_node_num += float(pred.shape[0])

        valid_loss = valid_loss / float(len(loader))
        valid_acc = valid_acc / total_node_num * 100

        return valid_acc, valid_loss.item()

    def test(self, loader):
        test_acc, test_loss = self.eval(loader)
        
        # Execute below if the rdkit is available
        device = self.args.device
        try:
            import rdkit
        except:
            return

        exact_match = 0
        validity = 0
        num_valid = 0
        num_invalid = 0

        # Reinit the dataloader
        loader = DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)

        for i, data in enumerate(tqdm(loader, desc='[Test]')):
            nodes = data.x.argmax(-1)
            a = to_dense_adj(data.edge_index, edge_attr=data.edge_attr)[0]
            mol = mol_from_graphs(nodes, a)
            if mol is None:
                num_invalid += 1
                continue
            else: num_valid += 1
            smiles = rdkit.Chem.MolToSmiles(mol)
            data = data.to(device)
            out = self.model(data)
            pred = torch.softmax(out, dim=-1).argmax(-1)
            pred_mol = mol_from_graphs(pred, a)
            if pred_mol is not None:
                pred_smiles = rdkit.Chem.MolToSmiles(pred_mol)
            else: pred_smiles = ''

            _exact_match = smiles == pred_smiles
            _validity = pred_smiles != ''

            exact_match += exact_match
            validity += _validity

        print(num_valid)
        exact_match = exact_match / float(num_valid) * 100
        validity = validity / float(num_valid) * 100

        print(f"GT Valid Molecules: {num_valid}, Invalid Molecules: {num_invalid}")
        print(f"Pred EM: {exact_match:.2f}, Validity: {validity:.2f}, Acc: {test_acc}")


    def experiment_name(self):

        ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

        exp_name = str()

        return exp_name
