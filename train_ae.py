import argparse
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import geoopt


from src.batchmodels import EucMLREuclidianAutoEncoder, EucMLRHyperbolicAutoEncoder, HypMLREuclidianAutoEncoder, \
    HypMLRHyperbolicAutoEncoder, ExpMap0, HypLinearAE, MobiusAutoEncoder, HypLinear, MultipleOptimizer
from src.batchrunner import train, report_metrics, eval_model, add_scores
from src.datareader import read_data
from src.datasets import observations_loader, UserBatchDataset, make_loaders_strong, make_loaders_weak
from src.mobius_linear import MobiusLinear
from src.random import fix_torch_seed
from src.geoopt_plusplus.manifolds import PoincareBall
import matplotlib.pyplot as plt
from src.geoopt_plusplus.modules import PoincareLinear, UnidirectionalPoincareMLR
from matplotlib import collections as mc
import matplotlib as mpl
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--dataname", type=str, required=True)
parser.add_argument("--encoder", type=str, required=True, choices=['HypLinear', 'Mobius', 'PoincareLinear'])
parser.add_argument("--decoder", type=str, required=True, choices=['HypLinear', 'Mobius', 'PoincareLinear', 'HypMLR'])
parser.add_argument("--bias", type=str, default='True')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--threshold", type=float, default=3.5)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--c", type=float, default=0.5)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--show-progress", default=False, action='store_true')
parser.add_argument("--data_dir", default="./data/")
args = parser.parse_args()

fix_torch_seed(args.seed)

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

ball = PoincareBall(args.c)
encoder_layer = layer_factories[args.encoder](train_data.shape[1], args.embedding_dim, bias=bias, ball=ball)
decoder_layer = layer_factories[args.decoder](args.embedding_dim, train_data.shape[1], bias=bias, ball=ball)
model = nn.Sequential(ExpMap0(ball), encoder_layer, decoder_layer).cuda()

encoder_optimizer = optimizers[args.encoder](model[1].parameters(), lr=args.learning_rate)
decoder_optimizer = optimizers[args.decoder](model[2].parameters(), lr=args.learning_rate)

optimizer = MultipleOptimizer(encoder_optimizer, decoder_optimizer)

scheduler = None

show_progress = args.show_progress

best_ndcg = -np.inf
for epoch in range(args.epochs):
    losses = train(train_val_loader, model, optimizer, criterion, show_progress=show_progress, threshold=args.threshold)
    scores = eval_model(model, criterion, test_loader, test_out_data, test_unbias, topk=[1, 5, 10, 20],
                        show_progress=args.show_progress, threshold=args.threshold, only_ndcg=True)
    scores.update({'train loss': np.mean(losses)})
    if scores['ndcg@10'] > best_ndcg:
        best_ndcg = scores['ndcg@10']
        best_model = deepcopy(model)
    report_metrics(scores, epoch)


scores = eval_model(best_model, criterion, test_loader, test_out_data, test_unbias, topk=[1, 5, 10, 20],
                    show_progress=args.show_progress, threshold=args.threshold)

if data_name == 'ml1m':
    add_scores(best_model[:2], usr_data, item_data, ball, scores)

print(scores)