import os
import time

import rdkit

from tqdm import tqdm, trange

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj

from utils.molecule_utils import ZINC, dataset_statistic, to_one_hot, mol_from_graphs
from utils.logger import Logger
from models.nets import GraphMultisetTransformer_for_Recon

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
        self.ckpt = os.path.join('./checkpoints/{}'.format(self.log_folder_name), "best_molecule.pth")

        self.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(args.gpu)
            self.args.device = 'cuda:{}'.format(args.gpu)
        else:
            self.args.device = 'cpu'

        self.train_dataset, self.val_dataset, self.test_dataset = self.load_data()

    def load_data(self):
        
        subset = True
        
        train_dataset = ZINC("./zinc", subset=subset, split='train', transform=to_one_hot)
        valid_dataset = ZINC("./zinc", subset=subset, split='val', transform=to_one_hot)
        test_dataset = ZINC("./zinc", subset=subset, split="test", transform=to_one_hot)
        avg_num_nodes = dataset_statistic(train_dataset)
        
        self.args.num_features = 28
        self.args.num_classes = 28
        self.args.avg_num_nodes = avg_num_nodes

        return train_dataset, valid_dataset, test_dataset

    def load_dataloader(self):
        
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)

        return train_loader, val_loader, test_loader

    def load_model(self):

        if self.args.model == 'GMT':
            model = GraphMultisetTransformer_for_Recon(self.args)

        else:
            raise ValueError("Model Name <{}> is Unknown".format(self.args.model))

        if self.use_cuda:
            model.to(self.args.device)

        return model

    def set_log(self):

        self.best_loss = 1000000
        self.patience = self.args.patience

        logger = Logger(str(os.path.join('./logs/{}/'.format(self.log_folder_name), 'experiment-{}_seed-{}.log'.format(self.exp_name, self.args.seed))), mode='a')

        t_start = time.perf_counter()

        return logger, t_start

    def train(self):

        train_loader, val_loader, test_loader = self.load_dataloader()

        self.model = self.load_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()

        logger, t_start = self.set_log()

        epoch_iter = trange(0, self.args.num_epochs, desc='[EPOCH]', position=1)

        for epoch in epoch_iter:
            self.model.train()
            
            for _, data in enumerate(tqdm(train_loader, desc='[Train]', position=0)):
                self.optimizer.zero_grad()
                data = data.to(self.args.device)
                out = self.model(data)
                target = data.x.argmax(-1) # To one-hot encoding
                
                loss = self.criterion(out, target)
                loss.backward()
                self.optimizer.step()

                desc = f"[Train] Train Loss {loss.item()}"
                epoch_iter.set_description(desc)
                epoch_iter.refresh()

            valid_acc, valid_loss = self.eval(val_loader)
            logger.log(f"[Epoch {epoch}] Loss: {valid_loss:.2f}, Acc: {valid_acc:.2f}")

            if valid_loss < self.best_loss:
                torch.save(self.model.state_dict(), self.ckpt)
                self.patience = self.args.patience
                self.best_loss = valid_loss
            else:
                self.patience -= 1
                if self.patience == 0: break

        t_end = time.perf_counter()

        # Load Best Model
        self.model = self.load_model()
        self.model.load_state_dict(torch.load(self.ckpt))
        self.model = self.model.to(self.args.device)

        num_valid, num_invalid, exact_match, validity, test_acc = self.test(test_loader)

        logger.log(f"GT Valid Molecules: {num_valid}, Invalid Molecules: {num_invalid}")
        logger.log(f"Pred EM: {exact_match:.2f}, Validity: {validity:.2f}, Acc: {test_acc} with Time: {t_end - t_start}")

        result_file = "./results/{}/{}-results.txt".format(self.log_folder_name, self.exp_name)
        with open(result_file, 'a+') as f:
            f.write("{}: {} {} {}\n".format(self.args.seed, exact_match, validity, test_acc))

    def eval(self, loader):

        self.model.eval()

        valid_loss = 0
        valid_acc = 0
        total_node_num = 0

        for _, data in enumerate(tqdm(loader, desc='[Eval]', position=0)):
            data = data.to(self.args.device)
            out = self.model(data)
            pred_logit = torch.softmax(out, dim=-1)
            target = data.x.argmax(-1)
            loss = self.criterion(pred_logit, target)
            pred = pred_logit.argmax(-1)
            valid_loss += loss
            valid_acc += (pred == target).sum().item()
            total_node_num += float(pred.shape[0])

        valid_loss = valid_loss / float(len(loader))
        valid_acc = valid_acc / total_node_num * 100

        return valid_acc, valid_loss.item()

    def test(self, loader):

        test_acc, _ = self.eval(loader)

        exact_match = 0
        validity = 0
        num_valid = 0
        num_invalid = 0

        for _, data in enumerate(tqdm(loader, desc='[Test]')):
            nodes = data.x.argmax(-1)
            a = to_dense_adj(data.edge_index, edge_attr=data.edge_attr)[0]
            mol = mol_from_graphs(nodes, a)
            
            if mol is None:
                num_invalid += 1
                continue
            else: num_valid += 1
            
            smiles = rdkit.Chem.MolToSmiles(mol)
            data = data.to(self.args.device)
            out = self.model(data)
            pred = torch.softmax(out, dim=-1).argmax(-1)
            pred_mol = mol_from_graphs(pred, a)
            
            if pred_mol is not None:
                pred_smiles = rdkit.Chem.MolToSmiles(pred_mol)
            else: pred_smiles = ''

            _exact_match = smiles == pred_smiles
            _validity = pred_smiles != ''

            exact_match += _exact_match
            validity += _validity

        exact_match = exact_match / float(num_valid) * 100
        validity = validity / float(num_valid) * 100

        return num_valid, num_invalid, exact_match, validity, test_acc

    def experiment_name(self):

        ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

        self.log_folder_name = os.path.join(*[self.args.data, self.args.model, self.args.experiment_number])

        if not(os.path.isdir('./checkpoints/{}'.format(self.log_folder_name))):
            os.makedirs(os.path.join('./checkpoints/{}'.format(self.log_folder_name)))

        if not(os.path.isdir('./results/{}'.format(self.log_folder_name))):
            os.makedirs(os.path.join('./results/{}'.format(self.log_folder_name)))

        if not(os.path.isdir('./logs/{}'.format(self.log_folder_name))):
            os.makedirs(os.path.join('./logs/{}'.format(self.log_folder_name)))

        print("Make Directory {} in Logs, Checkpoints and Results Folders".format(self.log_folder_name))

        exp_name = str()
        exp_name += "CV={}_".format(self.args.conv)
        exp_name += "NC={}_".format(self.args.num_convs)
        exp_name += "NU={}_".format(self.args.num_unconvs)
        exp_name += "MC={}_".format(self.args.mab_conv)
        exp_name += "MS={}_".format(self.args.model_string)
        exp_name += "BS={}_".format(self.args.batch_size)
        exp_name += "LR={}_".format(self.args.lr)
        exp_name += "DO={}_".format(self.args.dropout)
        exp_name += "HD={}_".format(self.args.num_hidden)
        exp_name += "NH={}_".format(self.args.num_heads)
        exp_name += "PL={}_".format(self.args.pooling_ratio)
        exp_name += "LN={}_".format(self.args.ln)
        exp_name += "CS={}_".format(self.args.cluster)
        exp_name += "TS={}".format(ts)

        # Save training arguments for reproduction
        torch.save(self.args, os.path.join('./checkpoints/{}'.format(self.log_folder_name), 'training_args.bin'))

        return exp_name
