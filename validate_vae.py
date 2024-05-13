import argparse

import numpy as np

import torch
import torch.nn as nn
import geoopt
import json

from src.batchrunner import eval_model, add_scores
from src.datareader import read_data
from src.geoopt_plusplus import UnidirectionalPoincareMLR, PoincareLinear
from src.models import HypLinear, MobiusLinear
from src.vae.vae_models import SampleLayer
from src.vae.vae_runner import Trainer
from src.datasets import make_loaders_weak
from src.geoopt_plusplus.manifolds import PoincareBall
from src.random import fix_torch_seed

from copy import deepcopy

from src.vae.vae_models import HyperbolicVAE

assert torch.cuda.is_available()

parser = argparse.ArgumentParser()

parser.add_argument("--dataname", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data/")
parser.add_argument("--threshold", type=float, default=3.5)
parser.add_argument("--num_setups", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--embedding_dim", type=int, default=600)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--c", type=float, default=0.005)
parser.add_argument('--total_anneal_steps', type=int, default=10)
parser.add_argument('--anneal_cap', type=float, default=1.0)
parser.add_argument("--show_progress", default=False, action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fix_torch_seed(args.seed)

criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

data_dir, data_name = args.data_dir, args.dataname

train_data, valid_in_data, valid_out_data, test_in_data, test_out_data, valid_unbias, test_unbias = read_data(data_dir, data_name)

train_loader, valid_loader, test_loader, train_val_loader = make_loaders_weak(train_data, valid_in_data, valid_out_data,
                                                                              test_in_data, test_out_data, args.batch_size, device)
total_size = train_data.shape[0] + valid_in_data.shape[0] + test_in_data.shape[0]

if data_name == 'ml1m':
    usr_data = torch.tensor((train_data > args.threshold).to_array(), dtype=torch.float64).to(device)
    item_data = torch.eye(usr_data.shape[1], dtype=torch.float64).to(device)

layer_factories = {'HypLinear': HypLinear, 'Mobius': MobiusLinear, 'PoincareLinear': PoincareLinear,
                   'HypMLR': UnidirectionalPoincareMLR}

optimizers = {'Mobius': geoopt.optim.RiemannianAdam, 'HypLinear': torch.optim.Adam, 'PoincareLinear': torch.optim.Adam,
              'HypMLR': torch.optim.Adam}

models_scores = {}

with open('vae_model_best_args.txt') as f:
    args_dict = json.load(f)

for encoder in ['HypLinear', 'Mobius', 'PoincareLinear']:
    for decoder in ['HypLinear', 'Mobius', 'PoincareLinear', 'HypMLR']:
# for encoder in ['HypLinear']:
#     for decoder in ['HypLinear']:
        args.learning_rate = args_dict[encoder + '+' + decoder]['learning_rate']
        args.embedding_dim = int(args_dict[encoder + '+' + decoder]['embedding_dim'])
        args.c = args_dict[encoder + '+' + decoder]['c']
        args.anneal_cap = args_dict[encoder + '+' + decoder]['anneal_cap']

        ball = PoincareBall(args.c)
        model = HyperbolicVAE(train_data.shape[1], args.embedding_dim, ball, layer_factories[encoder],
                              layer_factories[decoder], False).to(device)

        trainer = Trainer(model, total_anneal_steps=args.total_anneal_steps, anneal_cap=args.anneal_cap)

        encoder_optimizer = optimizers[encoder](model.component.fc_mean.parameters(), lr=args.learning_rate)
        decoder_optimizer = optimizers[decoder](model.decoder.parameters(), lr=args.learning_rate)
        var_optimizer = torch.optim.Adam(model.component.fc_logvar.parameters(), lr=args.learning_rate)

        optimizer = MultipleOptimizer(encoder_optimizer, decoder_optimizer, var_optimizer)

        scheduler = None
        best_ndcg = -np.inf
        for epoch in range(1, args.epochs + 1):
            trainer.train(optimizer=optimizer, train_loader=train_val_loader, threshold=args.threshold)
            scores = eval_model(trainer.model, criterion, test_loader, test_out_data, test_unbias, topk=[10],
                                show_progress=args.show_progress, variational=True, threshold=args.threshold, only_ndcg=True)
            if scores['ndcg@10'] > best_ndcg:
                best_ndcg = scores['ndcg@10']
                best_model = deepcopy(trainer.model)

        scores = eval_model(best_model, criterion, test_loader, test_out_data, test_unbias, topk=[1, 5, 10, 20, 50, 100],
                            show_progress=args.show_progress, variational=True, threshold=args.threshold)

        if data_name == 'ml1m':
            sample_model = nn.Sequential(best_model.component, SampleLayer(best_model.ball))
            add_scores(sample_model, usr_data, item_data, best_model.ball, scores)

        models_scores[encoder + '+' + decoder] = scores

        print(f'{encoder}+{decoder}:', scores)

for _, metrics in models_scores.items():
    for key, value in metrics.items():
        metrics[key] = np.float64(value)

with open("vae_model_scores.txt", "w") as fp:
    json.dump(models_scores, fp)
