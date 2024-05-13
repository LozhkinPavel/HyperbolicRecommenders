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
parser.add_argument("--num_setups", type=int, default=5)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--c", type=float, default=0.5)
parser.add_argument("--threshold", type=float, default=3.5)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--show-progress", default=False, action='store_true')
parser.add_argument("--data_dir", default="./data/")

args = parser.parse_args()

bias = (args.bias == 'True')

fix_torch_seed(args.seed)

learning_rates = np.logspace(-4, -2, 3)
embedding_dims = [16, 32, 64, 128, 256]
curvatures = [0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]

# learning_rates = [1e-2]
# embedding_dims = [256]
# curvatures = [1.0]

lr_grid, curvature_grid = np.meshgrid(learning_rates, curvatures)

lr_grid = lr_grid.flatten()
curvature_grid = curvature_grid.flatten()

time_to_converge = []
ndcgs = []

criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
# criterion = nn.BCEWithLogitsLoss()
data_dir, data_name = args.data_dir, args.dataname

train_data, valid_in_data, valid_out_data, test_in_data, test_out_data, valid_unbias, test_unbias = read_data(data_dir, data_name)

train_loader, valid_loader, test_loader, train_val_loader = make_loaders_weak(train_data, valid_in_data, valid_out_data,
                                                                         test_in_data, test_out_data, args.batch_size)
total_size = train_data.shape[0] + valid_in_data.shape[0] + test_in_data.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# usr_data = (torch.sparse_csr_tensor(train_data.indptr, train_data.indices, train_data.data, train_data.shape,
#                                     dtype=torch.float64).to(device).to_dense() > args.threshold).double()
# item_data = torch.eye(usr_data.shape[1], dtype=torch.float64).to(device)

layer_factories = {'HypLinear': HypLinear, 'Mobius': MobiusLinear, 'PoincareLinear': PoincareLinear,
                   'HypMLR': UnidirectionalPoincareMLR}

optimizers = {'Mobius': geoopt.optim.RiemannianAdam, 'HypLinear': torch.optim.Adam, 'PoincareLinear': torch.optim.Adam,
              'HypMLR': torch.optim.Adam}

models_scores = {}
models_best_args = {}

for encoder in tqdm(['HypLinear', 'Mobius', 'PoincareLinear']):
    for decoder in ['HypLinear', 'Mobius', 'PoincareLinear', 'HypMLR']:
# for encoder in ['HypLinear']:
#     for decoder in ['HypLinear']:
        for embedding_dim in tqdm(embedding_dims):
            best_ndcg = -np.inf
            for i in np.random.choice(np.arange(lr_grid.shape[0]), size=args.num_setups, replace=False):
                args.learning_rate = lr_grid[i]
                args.embedding_dim = embedding_dim
                args.c = curvature_grid[i]

                ball = PoincareBall(args.c)
                encoder_layer = layer_factories[encoder](train_data.shape[1], args.embedding_dim, bias=bias,
                                                         ball=ball)
                decoder_layer = layer_factories[decoder](args.embedding_dim, train_data.shape[1], bias=bias,
                                                         ball=ball)
                model = nn.Sequential(ExpMap0(ball), encoder_layer, decoder_layer).cuda()

                encoder_optimizer = optimizers[encoder](model[1].parameters(), lr=args.learning_rate)
                decoder_optimizer = optimizers[decoder](model[2].parameters(), lr=args.learning_rate)

                optimizer = MultipleOptimizer(encoder_optimizer, decoder_optimizer)

                scheduler = None

                show_progress = args.show_progress

                cur_best_ndcg = -np.inf
                for epoch in range(args.epochs):
                    losses = train(train_loader, model, optimizer, criterion, show_progress=show_progress,
                                   threshold=args.threshold)
                    scores = eval_model(model, criterion, valid_loader, valid_out_data, valid_unbias, topk=[10],
                                        show_progress=args.show_progress, threshold=args.threshold, only_ndcg=True)
                    scores.update({'train loss': np.mean(losses)})
                    if scores['ndcg@10'] > cur_best_ndcg:
                        cur_best_ndcg = scores['ndcg@10']
                        cur_val_scores = deepcopy(scores)
                #     report_metrics(scores, epoch)
                if cur_best_ndcg > best_ndcg:
                    best_ndcg = cur_best_ndcg
                    best_args = deepcopy(args)
                    val_scores = deepcopy(cur_val_scores)
                    # print(best_args)
                # print(best_ndcg)
            ball = PoincareBall(best_args.c)
            encoder_layer = layer_factories[encoder](train_data.shape[1], best_args.embedding_dim,
                                                     bias=bias, ball=ball)
            decoder_layer = layer_factories[decoder](best_args.embedding_dim, train_data.shape[1],
                                                     bias=bias, ball=ball)
            best_model = nn.Sequential(ExpMap0(ball), encoder_layer, decoder_layer).cuda()

            encoder_optimizer = optimizers[encoder](best_model[1].parameters(), lr=best_args.learning_rate)
            decoder_optimizer = optimizers[decoder](best_model[2].parameters(), lr=best_args.learning_rate)

            optimizer = MultipleOptimizer(encoder_optimizer, decoder_optimizer)

            scheduler = None
            show_progress = args.show_progress

            cur_best_ndcg = -np.inf
            for epoch in range(args.epochs):
                losses = train(train_val_loader, best_model, optimizer, criterion,
                               masked_loss=False, show_progress=show_progress, threshold=args.threshold)
                scores = eval_model(best_model, criterion, test_loader, test_out_data, test_unbias,
                                    topk=[10], show_progress=args.show_progress, threshold=args.threshold, only_ndcg=True)
                scores.update({'train loss': np.mean(losses)})
                if scores['ndcg@10'] > cur_best_ndcg:
                    cur_best_model = deepcopy(best_model)
                    cur_best_ndcg = scores['ndcg@10']

            # print(best_scores)
            best_model = deepcopy(cur_best_model)
            best_scores = eval_model(best_model, criterion, test_loader, test_out_data, test_unbias,
                                     topk=[1, 5, 10, 20, 50, 100], show_progress=args.show_progress, threshold=args.threshold)

            # add_scores(best_model[:2], usr_data, item_data, ball, best_scores)

            for key, value in val_scores.items():
                best_scores['val_' + key] = value

            models_scores[encoder + '+' + decoder + f'@{embedding_dim}'] = best_scores
            models_best_args[encoder + '+' + decoder + f'@{embedding_dim}'] = vars(best_args)

            print(encoder + '+' + decoder, 'best ndcg@10 is', best_scores['ndcg@10'])

for _, metrics in models_scores.items():
    for key, value in metrics.items():
        metrics[key] = np.float64(value)

for key, value in models_best_args.items():
    new_args = {}
    new_args['c'] = np.float64(value['c'])
    new_args['learning_rate'] = np.float64(value['learning_rate'])
    new_args['embedding_dim'] = np.float64(value['embedding_dim'])
    models_best_args[key] = new_args

with open("model_scores.txt", "w") as fp:
    json.dump(models_scores, fp)

with open("model_best_args.txt", "w") as fp:
    json.dump(models_best_args, fp)
