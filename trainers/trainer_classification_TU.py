import os
import time

from tqdm import tqdm, trange

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.optimization import get_cosine_schedule_with_warmup

from utils.data_loader import get_dataset
from models.nets import GraphMultisetTransformer

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

        self.dataset = self.load_data()

    def load_data(self):

        dataset = get_dataset(self.args.data, normalize=self.args.normalize)
        self.args.num_features, self.args.num_classes, self.args.avg_num_nodes = dataset.num_features, dataset.num_classes, np.ceil(np.mean([data.num_nodes for data in dataset]))
        print('# %s: [FEATURES]-%d [NUM_CLASSES]-%d [AVG_NODES]-%d' % (dataset, self.args.num_features, self.args.num_classes, self.args.avg_num_nodes))

        return dataset

    def load_model(self):

        if self.args.model == 'GMT':

            model = GraphMultisetTransformer(self.args)

        else:

            raise ValueError("Model Name <{}> is Unknown".format(self.args.model))

        return model

    def train(self):

        self.model = self.load_model()

        print(self.model)

        pass

    def eval(self, loader):

        pass

    def experiment_name(self):

        ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

        exp_name = str()

        return exp_name
