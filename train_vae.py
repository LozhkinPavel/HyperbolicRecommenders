import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import geoopt
from tqdm import tqdm

from src.batchmodels import HypLinear, MultipleOptimizer
from src.datareader import read_data
from src.datasets import make_loaders_weak
from src.geoopt_plusplus import UnidirectionalPoincareMLR, PoincareLinear
from src.mobius_linear import MobiusLinear
from src.vae.vae_models import HyperbolicVAE
from src.vae.rsvae import SampleLayer
from src.vae.vae_runner import Trainer
from src.batchrunner import report_metrics, add_scores, eval_model
from src.random import fix_torch_seed
from src.geoopt_plusplus.manifolds import PoincareBall

parser = argparse.ArgumentParser()
parser.add_argument("--dataname", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data/")
parser.add_argument("--encoder", type=str, required=True, choices=['HypLinear', 'Mobius', 'PoincareLinear'])
parser.add_argument("--decoder", type=str, required=True, choices=['HypLinear', 'Mobius', 'PoincareLinear', 'HypMLR'])
parser.add_argument("--threshold", type=float, default=3.5)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--embedding_dim", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--c", type=float, default=0.5)
parser.add_argument('--total_anneal_steps', type=int, default=20)
parser.add_argument('--anneal_cap', type=float, default=0.2)
parser.add_argument("--show_progress", default=False, action='store_true')
args = parser.parse_args()

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

criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

ball = PoincareBall(args.c)

model = HyperbolicVAE(train_data.shape[1], args.embedding_dim, ball, layer_factories[args.encoder],
                      layer_factories[args.decoder], False).to(device)

trainer = Trainer(model, total_anneal_steps=args.total_anneal_steps, anneal_cap=args.anneal_cap)

encoder_optimizer = optimizers[args.encoder](model.component.fc_mean.parameters(), lr=args.learning_rate)
decoder_optimizer = optimizers[args.decoder](model.decoder.parameters(), lr=args.learning_rate)
var_optimizer = torch.optim.Adam(model.component.fc_logvar.parameters(), lr=args.learning_rate)

optimizer = MultipleOptimizer(encoder_optimizer, decoder_optimizer, var_optimizer)

scheduler = None
best_ndcg = -np.inf
for epoch in tqdm(range(1, args.epochs + 1)):
    trainer.train(optimizer=optimizer, train_loader=train_val_loader,
                  threshold=args.threshold)
    scores = eval_model(trainer.model, criterion, test_loader, test_out_data, test_unbias, topk=[10],
                        show_progress=args.show_progress, variational=True, threshold=args.threshold, only_ndcg=True)
    if scores['ndcg@10'] > best_ndcg:
        best_ndcg = scores['ndcg@10']
        best_model = deepcopy(trainer.model)
    report_metrics(scores, epoch)

scores = eval_model(best_model, criterion, test_loader, test_out_data, test_unbias, topk=[1, 5, 10, 20, 50, 100],
                    show_progress=args.show_progress, variational=True, threshold=args.threshold)

if data_name == 'ml1m':
    sample_model = nn.Sequential(best_model.component, SampleLayer(best_model.ball))
    add_scores(sample_model, usr_data, item_data, best_model.ball, scores)
print("Final scores:", scores)

