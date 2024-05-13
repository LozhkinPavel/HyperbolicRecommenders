import argparse

import numpy as np

import torch
import torch.nn as nn
import geoopt
import json

from src.batchmodels import EucMLREuclidianAutoEncoder, EucMLRHyperbolicAutoEncoder, HypMLREuclidianAutoEncoder, \
    HypMLRHyperbolicAutoEncoder, HypLinearAE, MobiusAutoEncoder, MobiusLinear, PoincareLinear, \
    UnidirectionalPoincareMLR, ExpMap0, MultipleOptimizer, HypLinear
from src.batchrunner import train, report_metrics, eval_model, add_scores
from src.datareader import read_data
from src.datasets import make_loaders_strong, make_loaders_weak
from src.geoopt_plusplus.manifolds import PoincareBall
from src.random import fix_torch_seed

from copy import deepcopy
from tqdm import tqdm

assert torch.cuda.is_available()

parser = argparse.ArgumentParser()

parser.add_argument("--dataname", type=str, required=True)
parser.add_argument("--bias", type=str, default='True')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--c", type=float, default=0.5)
parser.add_argument("--threshold", type=float, default=3.5)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--show-progress", default=False, action='store_true')
parser.add_argument("--data_dir", default="./data/")

args = parser.parse_args()

bias = (args.bias == 'True')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fix_torch_seed(args.seed)

criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

data_dir, data_name = args.data_dir, args.dataname

train_data, valid_in_data, valid_out_data, test_in_data, test_out_data, valid_unbias, test_unbias = read_data(data_dir, data_name)

train_loader, valid_loader, test_loader, train_val_loader = make_loaders_weak(train_data, valid_in_data, valid_out_data,
                                                                              test_in_data, test_out_data, args.batch_size, device)

if data_name == 'ml1m':
    usr_data = torch.tensor((train_data > args.threshold).to_array(), dtype=torch.float64).to(device)
    item_data = torch.eye(usr_data.shape[1], dtype=torch.float64).to(device)

layer_factories = {'HypLinear': HypLinear, 'Mobius': MobiusLinear, 'PoincareLinear': PoincareLinear,
                   'HypMLR': UnidirectionalPoincareMLR}

optimizers = {'Mobius': geoopt.optim.RiemannianAdam, 'HypLinear': torch.optim.Adam, 'PoincareLinear': torch.optim.Adam,
              'HypMLR': torch.optim.Adam}

models_scores = {}

with open('ae_model_best_args.txt') as f:
    args_dict = json.load(f)

for encoder in ['HypLinear', 'Mobius', 'PoincareLinear']:
    for decoder in ['HypLinear', 'Mobius', 'PoincareLinear', 'HypMLR']:
# for encoder in ['HypLinear']:
#     for decoder in ['HypLinear']:
        args.learning_rate = args_dict[encoder + '+' + decoder]['learning_rate']
        args.embedding_dim = int(args_dict[encoder + '+' + decoder]['embedding_dim'])
        args.c = args_dict[encoder + '+' + decoder]['c']

        ball = PoincareBall(args.c)
        encoder_layer = layer_factories[encoder](train_data.shape[1], args.embedding_dim,
                                                 bias=bias, ball=ball)
        decoder_layer = layer_factories[decoder](args.embedding_dim, train_data.shape[1],
                                                 bias=bias, ball=ball)
        model = nn.Sequential(ExpMap0(ball), encoder_layer, decoder_layer).cuda()

        encoder_optimizer = optimizers[encoder](model[1].parameters(), lr=args.learning_rate)
        decoder_optimizer = optimizers[decoder](model[2].parameters(), lr=args.learning_rate)

        optimizer = MultipleOptimizer(encoder_optimizer, decoder_optimizer)

        scheduler = None
        show_progress = args.show_progress

        best_ndcg = -np.inf
        for epoch in range(args.epochs):
            losses = train(train_val_loader, model, optimizer, criterion, show_progress=show_progress,
                           threshold=args.threshold)
            scores = eval_model(model, criterion, test_loader, test_out_data, test_unbias,
                                topk=[10], show_progress=args.show_progress, threshold=args.threshold, only_ndcg=True)
            scores.update({'train loss': np.mean(losses)})
            if scores['ndcg@10'] > best_ndcg:
                best_model = deepcopy(model)
                best_ndcg = scores['ndcg@10']

        scores = eval_model(best_model, criterion, test_loader, test_out_data, test_unbias,
                            topk=[1, 5, 10, 20, 50, 100], show_progress=args.show_progress, threshold=args.threshold)

        if data_name == 'ml1m':
            add_scores(best_model[:2], usr_data, item_data, ball, scores)

        models_scores[encoder + '+' + decoder] = scores

        print(f'{encoder}+{decoder}:', scores)

for _, metrics in models_scores.items():
    for key, value in metrics.items():
        metrics[key] = np.float64(value)

with open("ae_model_scores.txt", "w") as fp:
    json.dump(models_scores, fp)
