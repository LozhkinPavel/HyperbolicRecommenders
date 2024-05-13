import json
import os
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import geoopt


from src.batchmodels import EucMLREuclidianAutoEncoder, EucMLRHyperbolicAutoEncoder, HypMLREuclidianAutoEncoder,\
    HypMLRHyperbolicAutoEncoder, ExpMap0, HypLinearAE, MobiusAutoEncoder, HypLinear, EdgeAutoEncoder
from src.batchrunner import train, report_metrics, eval_model, train_edge, eval_edge_model
from src.datareader import read_data
from src.datasets import observations_loader, UserBatchDataset, make_loaders_strong, make_loaders_weak
from src.random import fix_torch_seed
from src.vae.vae_runner import HypVaeDataset
import matplotlib.pyplot as plt
from src.geoopt_plusplus.modules import PoincareLinear, UnidirectionalPoincareMLR
from matplotlib import collections as mc
import matplotlib as mpl
from copy import deepcopy

parser = argparse.ArgumentParser()

parser.add_argument("--datapack", type=str, required=True, choices=["persdiff", "urm"])
parser.add_argument("--dataname", type=str, required=True) # depends on choice of data pack
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--c", type=float, default=0.5)
parser.add_argument("--gamma", type=float, default=0.7)
parser.add_argument("--threshold", type=float, default=3.5)
parser.add_argument("--step_size", type=int, default=7)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--show-progress", default=False, action='store_true')
parser.add_argument("--data_dir", default="./data/")
# wandb compatibility
parser.add_argument("--scheduler_on", type=str, default="True")
args = parser.parse_args()

scheduler_on = (args.scheduler_on == "True")

fix_torch_seed(args.seed)

learning_rates = np.logspace(-5, -2, 4)
embedding_dims = [32, 64, 128, 256]
curvatures = [0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]

# learning_rates = [1e-3]
# embedding_dims = [32]
# curvatures = [1.0]

criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

data_dir, data_pack, data_name = args.data_dir, args.datapack, args.dataname

train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = read_data(data_dir, data_pack, data_name)

train_loader, valid_loader, test_loader, train_val_loader = make_loaders_weak(train_data, valid_in_data, valid_out_data,
                                                                         test_in_data, test_out_data, args.batch_size)
total_size = train_data.shape[0] + valid_in_data.shape[0] + test_in_data.shape[0]

best_ndcg = -np.inf
for learning_rate in learning_rates:
    for embedding_dim in embedding_dims:
        for c in curvatures:
            args.learning_rate = learning_rate
            args.embedding_dim = embedding_dim
            args.c = c

            model = EdgeAutoEncoder(train_data.shape[1], total_size, args.embedding_dim, c=args.c).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            show_progress = args.show_progress

            cur_best_ndcg = -np.inf
            for epoch in range(args.epochs):
                losses = train_edge(train_loader, model, optimizer, criterion, show_progress=show_progress, threshold=args.threshold)
                scores = eval_edge_model(model, criterion, valid_loader, valid_out_data, topk=[10],
                                    show_progress=args.show_progress, threshold=args.threshold)
                scores.update({'train loss': np.mean(losses)})
                if scores['ndcg@10'] > cur_best_ndcg:
                    cur_best_ndcg = scores['ndcg@10']
                # report_metrics(scores, epoch)
            if cur_best_ndcg > best_ndcg:
                best_ndcg = cur_best_ndcg
                best_args = deepcopy(args)
                # print(best_args)


best_model = EdgeAutoEncoder(train_data.shape[1], total_size + train_data.shape[1], best_args.embedding_dim,
                             c=best_args.c).cuda()

optimizer = torch.optim.Adam(best_model.parameters(), lr=best_args.learning_rate)

scheduler = None

cur_best_ndcg = -np.inf
for epoch in range(args.epochs):
    losses = train_edge(train_val_loader, best_model, optimizer, criterion,
                        masked_loss=False, show_progress=show_progress, threshold=args.threshold)
    scores = eval_edge_model(best_model, criterion, test_loader, test_out_data, topk=[1, 5, 10, 20],
                             show_progress=args.show_progress, threshold=args.threshold)
    scores.update({'train loss': np.mean(losses)})
    if scores['ndcg@10'] > cur_best_ndcg:
        cur_best_model = deepcopy(best_model)
        cur_best_ndcg = scores['ndcg@10']
        best_scores = deepcopy(scores)

print(best_scores)
best_model = deepcopy(cur_best_model)
new_model = nn.Sequential(best_model.edge_enc, best_model.exp_map, best_model.encode,
                          PoincareLinear(best_args.embedding_dim, 2, bias=True, ball=best_model.ball),
                          PoincareLinear(2, best_args.embedding_dim, bias=True, ball=best_model.ball),
                          best_model.decode).cuda()

for param in new_model[0].parameters():
    param.requires_grad = False
for param in new_model[1].parameters():
    param.requires_grad = False
for param in new_model[2].parameters():
    param.requires_grad = False
for param in new_model[5].parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=best_args.step_size, gamma=best_args.gamma)

for epoch in range(best_args.epochs):
    losses = train_edge(train_loader, new_model, optimizer, criterion, show_progress=show_progress, threshold=args.threshold)

usr_data = (torch.sparse_csr_tensor(train_data.indptr, train_data.indices, train_data.data, train_data.shape,
                                   dtype=torch.float64).to_dense().cuda() > args.threshold).type(torch.DoubleTensor)
item_data = torch.eye(usr_data.shape[1], dtype=torch.float64).cuda()

usr_embeddings = new_model[:4]([usr_data, torch.arange(usr_data.shape[0]).cuda()]).cpu().detach().numpy()
item_embeddings = new_model[:4]([item_data, torch.arange(total_size, total_size + item_data.shape[0]).cuda()]).cpu().detach().numpy()

usr_euc_dist = np.sqrt((usr_embeddings**2)[:, 0] + (usr_embeddings**2)[:, 1])
usr_hyp_dist = best_model.ball.dist0(torch.tensor(usr_embeddings, dtype=torch.float64).cuda(), dim=1).cpu().numpy()

item_euc_dist = np.sqrt((item_embeddings ** 2)[:, 0] + (item_embeddings ** 2)[:, 1])
item_hyp_dist = best_model.ball.dist0(torch.tensor(item_embeddings, dtype=torch.float64).cuda(), dim=1).cpu().numpy()

inds = usr_data.sum(dim=1).argsort().cpu().numpy()
usr_euc_dist = usr_euc_dist[inds]
usr_hyp_dist = usr_hyp_dist[inds]

inds = usr_data.sum(dim=0).argsort().cpu().numpy()
item_euc_dist = item_euc_dist[inds]
item_hyp_dist = item_hyp_dist[inds]

best_scores['usr_popquarter_mx'] = np.max(usr_euc_dist[-214:]) / best_model.ball.radius
best_scores['usr_popquarter_mean'] = np.mean(usr_euc_dist[-214:]) / best_model.ball.radius

best_scores['usr_modpophalf_mx'] = np.max(usr_euc_dist[-1432:-214]) / best_model.ball.radius
best_scores['usr_modpophalf_mean'] = np.mean(usr_euc_dist[-1432:-214]) / best_model.ball.radius

best_scores['usr_unpopquarter_mx'] = np.max(usr_euc_dist[:-1432]) / best_model.ball.radius
best_scores['usr_unpopquarter_mean'] = np.mean(usr_euc_dist[:-1432]) / best_model.ball.radius

best_scores['usr_pop_euc@100'] = np.max(usr_euc_dist[-100:]) / best_model.ball.radius
best_scores['usr_pop_euc@200'] = np.max(usr_euc_dist[-200:]) / best_model.ball.radius

best_scores['usr_pop_hyp@100'] = np.max(usr_hyp_dist[-100:])
best_scores['usr_pop_hyp@200'] = np.max(usr_hyp_dist[-200:])

best_scores['item_pop_euc@100'] = np.max(item_euc_dist[-100:]) / best_model.ball.radius
best_scores['item_pop_euc@200'] = np.max(item_euc_dist[-200:]) / best_model.ball.radius

best_scores['item_pop_hyp@100'] = np.max(item_hyp_dist[-100:])
best_scores['item_pop_hyp@200'] = np.max(item_hyp_dist[-200:])

best_scores['usr_antipop_euc@100'] = np.max(usr_euc_dist[:100]) / best_model.ball.radius
best_scores['usr_antipop_euc@200'] = np.max(usr_euc_dist[:200]) / best_model.ball.radius

best_scores['usr_antipop_hyp@100'] = np.max(usr_hyp_dist[:100])
best_scores['usr_antipop_hyp@200'] = np.max(usr_hyp_dist[:200])

best_scores['item_antipop_euc@100'] = np.max(item_euc_dist[:100]) / best_model.ball.radius
best_scores['item_antipop_euc@200'] = np.max(item_euc_dist[:200]) / best_model.ball.radius

best_scores['item_antipop_hyp@100'] = np.max(item_hyp_dist[:100])
best_scores['item_antipop_hyp@200'] = np.max(item_hyp_dist[:200])

for key, value in best_scores.items():
    best_scores[key] = np.float64(value)

with open("model_scores.txt", "w") as fp:
    json.dump(best_scores, fp)

best_args = vars(best_args)
new_args = {}
new_args['c'] = np.float64(best_args['c'])
new_args['learning_rate'] = np.float64(best_args['learning_rate'])
new_args['embedding_dim'] = np.float64(best_args['embedding_dim'])

with open("model_best_args.txt", "w") as fp:
    json.dump(new_args, fp)