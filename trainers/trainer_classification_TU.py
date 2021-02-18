import os
import time

from tqdm import tqdm, trange

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.optimization import get_cosine_schedule_with_warmup

from torch_geometric.data import DataLoader

from functools import reduce

from utils.data import get_dataset, num_graphs
from utils.logger import Logger
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
        self.exp_name = self.set_experiment_name()

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

    def load_dataloader(self, fold_number, val_fold_number):

        train_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/train_idx-%d.txt' % (self.args.data, fold_number),
                                                dtype=np.int32), dtype=torch.long)
        val_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/test_idx-%d.txt' % (self.args.data, val_fold_number),
                                                dtype=np.int32), dtype=torch.long)     
        test_idxes = torch.as_tensor(np.loadtxt('./datasets/%s/10fold_idx/test_idx-%d.txt' % (self.args.data, fold_number),
                                                dtype=np.int32), dtype=torch.long)

        all_idxes = reduce(np.union1d, (train_idxes, val_idxes, test_idxes))
        assert len(all_idxes) == len(self.dataset)

        train_idxes = torch.as_tensor(np.setdiff1d(train_idxes, val_idxes))

        train_set, val_set, test_set = self.dataset[train_idxes], self.dataset[val_idxes], self.dataset[test_idxes]

        train_loader = DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=self.args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_set, batch_size=self.args.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def load_model(self):

        if self.args.model == 'GMT':

            model = GraphMultisetTransformer(self.args)

        else:

            raise ValueError("Model Name <{}> is Unknown".format(self.args.model))

        if self.use_cuda:

            model.to(self.args.device)

        return model

    def set_log(self, fold_number):

        self.patience = 0
        self.best_loss_epoch = 0
        self.best_acc_epoch = 0
        self.best_loss = 1e9
        self.best_loss_acc = -1e9
        self.best_acc = -1e9
        self.best_acc_loss = 1e9

        logger = Logger(str(os.path.join('./logs/{}/'.format(self.log_folder_name), 'experiment-{}_fold-{}_seed-{}.log'.format(self.exp_name, fold_number, self.args.seed))), mode='a')

        t_start = time.perf_counter()

        return logger, t_start

    def organize_val_log(self, logger, train_loss, val_loss, val_acc, fold_number, epoch):

        if val_loss < self.best_loss:
            torch.save(
                self.model.state_dict(), 
                './checkpoints/{}/experiment-{}_fold-{}_seed-{}_best-model.pth'.format(self.log_folder_name, self.exp_name, fold_number, self.args.seed)
            )
            
            self.best_loss_acc = val_acc
            self.best_loss = val_loss
            self.best_loss_epoch = epoch
            self.patience = 0
        else:
            self.patience += 1

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_acc_loss = val_loss
            self.best_acc_epoch = epoch

        logger.log('[Val: Fold %d-Epoch %d] TrL: %.2f VaL: %.2f VaAcc: %.2f%%' % (
            fold_number, epoch, train_loss, val_loss, val_acc))

        logger.log("[Val: Fold %d-Epoch %d] (Loss) Loss: %.2f Acc: %.2f%% at Epoch: %d / (Acc) Loss: %.2f Acc: %.2f%% at Epoch: %d" % (
            fold_number, epoch, self.best_loss, self.best_loss_acc, self.best_loss_epoch, self.best_acc_loss, self.best_acc, self.best_acc_epoch))

    def organize_test_log(self, logger, test_loss, test_acc, t_start, t_end, fold_number):

        self.overall_results['durations'].append(t_end - t_start)
        self.overall_results['val_loss'].append(self.best_loss)
        self.overall_results['val_acc'].append(self.best_acc)
        self.overall_results['test_loss'].append(test_loss)
        self.overall_results['test_acc'].append(test_acc)

        logger.log("[Test: Fold %d] (Loss) Loss: %.2f Acc: %.2f%% at Epoch: %d / (Acc) Loss: %.2f Acc: %.2f%% at Epoch: %d" % (
            fold_number, self.best_loss, self.best_loss_acc, self.best_loss_epoch, self.best_acc_loss, self.best_acc, self.best_acc_epoch))
        logger.log("[Test: Fold {}] Test Acc: {} with Time: {}".format(fold_number, test_acc, (t_end - t_start)))

        test_result_file = "./results/{}/results.txt".format(self.log_folder_name)
        with open(test_result_file, 'a+') as f:
            f.write("[FOLD {}] {}: {} {} {} {}\n".format(fold_number, self.args.seed, self.best_loss, self.best_acc, test_loss, test_acc))

    def train(self):

        self.overall_results = {
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'durations': []
        }

        train_fold_iter = tqdm(range(1, 11), desc='Training')
        val_fold_iter = [i for i in range(1, 11)]

        for fold_number in train_fold_iter:

            val_fold_number = val_fold_iter[fold_number - 2]

            train_loader, val_loader, test_loader = self.load_dataloader(fold_number, val_fold_number)

            # Load Model & Optimizer
            self.model = self.load_model()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)

            if self.args.lr_schedule:
                self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.args.patience * len(train_loader), self.args.num_epochs * len(train_loader))

            logger, t_start = self.set_log(fold_number)

            # K-Fold Training
            for epoch in trange(0, (self.args.num_epochs), desc = '[Epoch]', position = 1):

                self.model.train()
                total_loss = 0

                for _, data in enumerate(train_loader):

                    self.optimizer.zero_grad()
                    data = data.to(self.args.device)
                    out = self.model(data)
                    loss = F.nll_loss(out, data.y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                    total_loss += loss.item() * num_graphs(data)
                    self.optimizer.step()

                    if self.args.lr_schedule:
                        self.scheduler.step()

                total_loss = total_loss / len(train_loader.dataset)

                # Validation
                val_acc, val_loss = self.eval(val_loader)
                self.organize_val_log(logger, total_loss, val_loss, val_acc, fold_number, epoch)

                train_fold_iter.set_description('[Fold %d] TrL: %.2f VaL: %.2f VaAcc: %.2f%%' % (
                    fold_number, total_loss, val_loss, val_acc))
                train_fold_iter.refresh()

            t_end = time.perf_counter()

            checkpoint = torch.load('./checkpoints/{}/experiment-{}_fold-{}_seed-{}_best-model.pth'.format(self.log_folder_name, self.exp_name, fold_number, self.args.seed))
            self.model.load_state_dict(checkpoint)
            
            test_acc, test_loss = self.eval(test_loader)
            self.organize_test_log(logger, test_loss, test_acc, t_start, t_end, fold_number)

        final_result_file = "./results/{}/total_results.txt".format(self.log_folder_name)
        with open(final_result_file, 'a+') as f:
            f.write("{}: {} {} {} {}\n".format(
                self.args.seed, 
                np.array(self.overall_results['val_acc']).mean(), 
                np.array(self.overall_results['val_acc']).std(), 
                np.array(self.overall_results['test_acc']).mean(), 
                np.array(self.overall_results['test_acc']).std()
            ))

    def eval(self, loader):

        self.model.eval()

        correct = 0.
        loss = 0.

        for data in loader:

            data = data.to(self.args.device)
            with torch.no_grad():
                out = self.model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss += F.nll_loss(out, data.y, reduction='sum').item()
        
        return correct / len(loader.dataset), loss / len(loader.dataset)

    def set_experiment_name(self):

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
        exp_name += "MC={}_".format(self.args.mab_conv)
        exp_name += "MS={}_".format(self.args.model_string)
        exp_name += "BS={}_".format(self.args.batch_size)
        exp_name += "LR={}_".format(self.args.lr)
        exp_name += "WD={}_".format(self.args.weight_decay)
        exp_name += "GN={}_".format(self.args.grad_norm)
        exp_name += "DO={}_".format(self.args.dropout)
        exp_name += "HD={}_".format(self.args.num_hidden)
        exp_name += "NH={}_".format(self.args.num_heads)
        exp_name += "PL={}_".format(self.args.pooling_ratio)
        exp_name += "LN={}_".format(self.args.ln)
        exp_name += "LS={}_".format(self.args.lr_schedule)
        exp_name += "CS={}_".format(self.args.cluster)
        exp_name += "NM={}_".format(self.args.normalize)
        exp_name += "TS={}".format(ts)

        # Save training arguments for reproduction
        torch.save(self.args, os.path.join('./checkpoints/{}'.format(self.log_folder_name), 'training_args.bin'))

        return exp_name
