import os
import time

from tqdm import tqdm, trange

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pygsp import graphs
from torch_geometric.data import Data

from utils.logger import Logger
from models.nets import GraphMultisetTransformer_for_Recon

import matplotlib.pyplot as plt

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
        self.ckpt = os.path.join('./checkpoints/{}'.format(self.log_folder_name), "best_model.pth")

        self.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(args.gpu)
            self.args.device = 'cuda:{}'.format(args.gpu)
        else:
            self.args.device = 'cpu'

        self.data = self.load_data()

    def load_data(self):
        
        if self.args.data == 'ring':
            G = graphs.Ring(N=200)
        elif self.args.data == 'grid':
            G = graphs.Grid2d(N1=30, N2=30)

        X = G.coords.astype(np.float32)
        A = G.W

        coo = A.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        X = torch.FloatTensor(X)
        edge_index = torch.LongTensor(indices)
        edge_weight = torch.FloatTensor(values)
        batch = torch.LongTensor([0 for _ in range(X.shape[0])])

        data = Data(
            x = X,
            edge_index = edge_index,
            edge_attr = edge_weight,
            batch = batch
        )

        self.args.num_features = 2
        self.args.num_classes = 2
        self.args.avg_num_nodes = A.shape[0]

        if self.use_cuda:
            data.to(self.args.device)

        return data

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

        self.model = self.load_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.MSELoss()

        logger, t_start = self.set_log()

        epoch_iter = trange(0, self.args.num_epochs, desc='[EPOCH]', position=0)

        for epoch in epoch_iter:
            self.model.train()
            
            self.optimizer.zero_grad()
            out = self.model(self.data)
            target = self.data.x
            
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()

            desc = f"[Train] Train Loss {loss.item()}"
            epoch_iter.set_description(desc)
            epoch_iter.refresh()

            logger.log(f"[Epoch {epoch}] Loss: {loss.item():.10f}")

            if loss.item() < self.best_loss:
                torch.save(self.model.state_dict(), self.ckpt)
                self.patience = self.args.patience
                self.best_loss = loss.item()
            else:
                self.patience -= 1
                if self.patience == 0: break

        t_end = time.perf_counter()

        # Load Best Model
        self.model = self.load_model()
        self.model.load_state_dict(torch.load(self.ckpt))
        self.model = self.model.to(self.args.device)

        loss, target, out = self.eval()
        self.draw(target, out)

        logger.log(f"Loss: {loss:.10f} with Time: {t_end - t_start}")

        result_file = "./results/{}/{}-results.txt".format(self.log_folder_name, self.exp_name)
        with open(result_file, 'a+') as f:
            f.write("{}: {}\n".format(self.args.seed, loss))

    def eval(self):

        self.model.eval()

        out = self.model(self.data)
        target = self.data.x
        loss = self.criterion(out, target)

        return loss.item(), target.detach().cpu(), out.detach().cpu()

    def draw(self, target, out):

        plt.figure(figsize=(8, 4))
        pad = 0.1
        x_min, x_max = target[:, 0].min() - pad, target[:, 0].max() + pad
        y_min, y_max = target[:, 1].min() - pad, target[:, 1].max() + pad
        colors = target[:, 0] + target[:, 1]
        plt.subplot(1, 2, 1)
        plt.scatter(*target[:, :2].T, c=colors, s=8, zorder=2)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('Original')
        plt.subplot(1, 2, 2)
        plt.scatter(*out[:, :2].T, c=colors, s=8, zorder=2)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('Reconstructed')
        plt.tight_layout()
        plt.savefig("./results/{}/{}-plot.jpg".format(self.log_folder_name, self.exp_name))

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
