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
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from utils.data import num_graphs
from utils.logger import Logger
from models.nets import GraphMultisetTransformer_for_OGB

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

        self.evaluator = Evaluator(args.data)

    def load_data(self):

        dataset = PygGraphPropPredDataset(name = self.args.data)
        self.args.task_type, self.args.num_features, self.args.num_classes, self.args.avg_num_nodes \
            = dataset.task_type, dataset.num_features, dataset.num_tasks, np.ceil(np.mean([data.num_nodes for data in dataset]))
        print('# %s: [Task]-%s [FEATURES]-%d [NUM_CLASSES]-%d [AVG_NODES]-%d' % (dataset, self.args.task_type, self.args.num_features, self.args.num_classes, self.args.avg_num_nodes))

        return dataset

    def load_dataloader(self):

        split_idx = self.dataset.get_idx_split()

        train_loader = DataLoader(self.dataset[split_idx["train"]], batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(self.dataset[split_idx["valid"]], batch_size=self.args.batch_size, shuffle=False)
        test_loader = DataLoader(self.dataset[split_idx["test"]], batch_size=self.args.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def load_model(self):

        if self.args.model == 'GMT':

            model = GraphMultisetTransformer_for_OGB(self.args)

        else:

            raise ValueError("Model Name <{}> is Unknown".format(self.args.model))

        if self.use_cuda:

            model.to(self.args.device)

        return model

    def set_log(self):

        self.train_curve = []
        self.valid_curve = []
        self.test_curve = []

        logger = Logger(str(os.path.join('./logs/{}/'.format(self.log_folder_name), 'experiment-{}_seed-{}.log'.format(self.exp_name, self.args.seed))), mode='a')

        t_start = time.perf_counter()

        return logger, t_start

    def organize_log(self, logger, train_perf, valid_perf, test_perf, train_loss, epoch):

        self.train_curve.append(train_perf[self.dataset.eval_metric])
        self.valid_curve.append(valid_perf[self.dataset.eval_metric])
        self.test_curve.append(test_perf[self.dataset.eval_metric])

        logger.log("[Val: Epoch %d] (Loss) Loss: %.4f Train: %.4f%% Valid: %.4f%% Test: %.4f%% " % (
            epoch, train_loss, self.train_curve[-1], self.valid_curve[-1], self.test_curve[-1]))

    def train(self):

        train_loader, val_loader, test_loader = self.load_dataloader()

        # Load Model & Optimizer
        self.model = self.load_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)

        self.cls_criterion = torch.nn.BCEWithLogitsLoss()
        self.reg_criterion = torch.nn.MSELoss()

        if self.args.lr_schedule:
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.args.patience * len(train_loader), self.args.num_epochs * len(train_loader))

        logger, t_start = self.set_log()

        for epoch in trange(0, (self.args.num_epochs), desc = '[Epoch]', position = 1):

            self.model.train()
            total_loss = 0

            for _, data in enumerate(tqdm(train_loader, desc="[Iteration]")):

                if data.x.shape[0] == 1 or data.batch[-1] == 0: pass

                self.optimizer.zero_grad()
                data = data.to(self.args.device)
                out = self.model(data)

                is_labeled = data.y == data.y

                if "classification" in self.args.task_type: 
                    loss = self.cls_criterion(out.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
                else:
                    loss = self.reg_criterion(out.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                total_loss += loss.item() * num_graphs(data)
                self.optimizer.step()

                if self.args.lr_schedule:
                    self.scheduler.step()

            total_loss = total_loss / len(train_loader.dataset)

            train_perf, valid_perf, test_perf = self.eval(train_loader), self.eval(val_loader), self.eval(test_loader)
            self.organize_log(logger, train_perf, valid_perf, test_perf, total_loss, epoch)

        t_end = time.perf_counter()

        if 'classification' in self.dataset.task_type:
            best_val_epoch = np.argmax(np.array(self.valid_curve))
            best_train = max(self.train_curve)
        else:
            best_val_epoch = np.argmin(np.array(self.valid_curve))
            best_train = min(self.train_curve)

        best_val = self.valid_curve[best_val_epoch]
        test_score = self.test_curve[best_val_epoch]

        logger.log("Train: {} Valid: {} Test: {} with Time: {}".format(best_train, best_val, test_score, (t_end - t_start)))

        result_file = "./results/{}/results.txt".format(self.log_folder_name)
        with open(result_file, 'a+') as f:
            f.write("{}: {} {} {} {}\n".format(self.args.seed, best_train, self.train_curve[best_val_epoch], best_val, test_score))

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'Val': best_val,
            'Train': self.train_curve[best_val_epoch],
            'Test': test_score,
            'BestTrain': best_train
            }, './checkpoints/{}/best-model_{}.pth'.format(self.log_folder_name, self.args.seed))

    def eval(self, loader):

        self.model.eval()

        y_true = []
        y_pred = []

        for _, batch in enumerate(tqdm(loader, desc="[Iteration]")):
            batch = batch.to(self.args.device)

            if batch.x.shape[0] == 1: pass

            with torch.no_grad():
                pred = self.model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return self.evaluator.eval(input_dict)

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
        exp_name += "TS={}".format(ts)

        # Save training arguments for reproduction
        torch.save(self.args, os.path.join('./checkpoints/{}'.format(self.log_folder_name), 'training_args.bin'))

        return exp_name