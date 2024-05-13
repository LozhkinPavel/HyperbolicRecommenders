import json
import os
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import geoopt



from src.batchrunner import report_metrics, train_edge, eval_edge_model, add_scores
from src.datasets import make_loaders_weak
from src.models import EdgeAutoEncoder
from src.random import fix_torch_seed

train_edge, eval_edge_model
from src.datareader import read_data
from src.geoopt_plusplus.modules import PoincareLinear, UnidirectionalPoincareMLR
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fix_torch_seed(args.seed)

data_dir, data_name = args.data_dir, args.dataname

train_data, valid_in_data, valid_out_data, test_in_data, test_out_data, valid_unbias, test_unbias = read_data(data_dir, data_name)

train_loader, valid_loader, test_loader, train_val_loader = make_loaders_weak(train_data, valid_in_data, valid_out_data,
                                                                              test_in_data, test_out_data, args.batch_size)
criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

total_size = train_data.shape[0] + valid_in_data.shape[0] + test_in_data.shape[0]

if data_name == 'ml1m':
    usr_data = torch.tensor((train_data > args.threshold).to_array(), dtype=torch.float64).to(device)
    item_data = torch.eye(usr_data.shape[1], dtype=torch.float64).to(device)

learning_rates = np.logspace(-5, -2, 4)
embedding_dims = [32, 64, 128, 256]
curvatures = [0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]

# learning_rates = [1e-3]
# embedding_dims = [32]
# curvatures = [1.0]

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
            if cur_best_ndcg > best_ndcg:
                best_ndcg = cur_best_ndcg
                best_args = deepcopy(args)


best_model = EdgeAutoEncoder(train_data.shape[1], total_size + train_data.shape[1], best_args.embedding_dim,
                             c=best_args.c).cuda()

optimizer = torch.optim.Adam(best_model.parameters(), lr=best_args.learning_rate)

scheduler = None

best_ndcg = -np.inf
for epoch in range(args.epochs):
    losses = train_edge(train_val_loader, best_model, optimizer, criterion, show_progress=show_progress,
                        threshold=args.threshold)
    scores = eval_edge_model(best_model, criterion, test_loader, test_out_data, topk=[1, 5, 10, 20],
                             show_progress=args.show_progress, threshold=args.threshold)
    scores.update({'train loss': np.mean(losses)})
    if scores['ndcg@10'] > best_ndcg:
        cur_best_model = deepcopy(best_model)
        best_ndcg = scores['ndcg@10']
        best_scores = deepcopy(scores)

print(best_scores)
best_model = deepcopy(cur_best_model)
new_model = nn.Sequential(best_model.edge_enc, best_model.exp_map, best_model.encode,
                          PoincareLinear(best_args.embedding_dim, 2, bias=True, ball=best_model.ball),
                          PoincareLinear(2, best_args.embedding_dim, bias=True, ball=best_model.ball),
                          best_model.decode).to(device)

if data_name == 'ml1m':
    add_scores(best_model[:2], usr_data, item_data, ball, scores)

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