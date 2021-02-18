import os
import time

from tqdm import tqdm, trange

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader

from utils.molecule_utils import ZINC, dataset_statistic, to_one_hot
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

        self.train_loader, self.val_loader, self.test_loader, self.n_nodes = self.load_data()

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
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        self.args.num_features = 28
        self.args.num_classes = 28

        return train_loader, val_loader, test_loader, n_nodes


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
        ITER = 500
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

    def eval(self, loader):

        pass

    def experiment_name(self):

        ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

        exp_name = str()

        return exp_name
