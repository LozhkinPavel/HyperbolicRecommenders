import json
import os
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import geoopt

from src.batchmodels import PureSVD
from src.batchrunner import train, report_metrics, eval_model
from src.datareader import read_data
from src.datasets import make_loaders_strong, make_loaders_weak
from src.random import fix_torch_seed
import scipy

from copy import deepcopy

assert torch.cuda.is_available()


parser = argparse.ArgumentParser()

parser.add_argument("--dataname", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--threshold", type=float, default=3.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data_dir", default="./data/")
parser.add_argument("--show-progress", default=False, action='store_true')
args = parser.parse_args()

fix_torch_seed(args.seed)

ranks = [10, 25, 50, 100, 200, 500]
ndcgs = []

data_dir, data_name = args.data_dir, args.dataname

train_data, valid_in_data, valid_out_data, test_in_data, test_out_data, valid_unbias, test_unbias = read_data(data_dir, data_name)

train_loader, valid_loader, test_loader, train_val_loader = make_loaders_weak(train_data, valid_in_data, valid_out_data,
                                                                         test_in_data, test_out_data, args.batch_size)
total_size = train_data.shape[0] + valid_in_data.shape[0] + test_in_data.shape[0]

best_rank = -1
best_ndcg = -np.inf
criterion = nn.CrossEntropyLoss(reduction='mean')

for rank in ranks:
    model = PureSVD(rank=rank)
    model.fit((train_data > args.threshold).astype(np.float64))
    scores = eval_model(model, criterion, valid_loader, valid_out_data, valid_unbias, topk=[1, 5, 10, 20],
                        show_progress=args.show_progress, threshold=args.threshold)
    if scores['ndcg@10'] > best_ndcg:
        best_ndcg = scores['ndcg@10']
        best_rank = rank
    print(f"Rank {rank}", scores)

best_model = PureSVD(rank=best_rank)
best_model.fit((scipy.sparse.vstack((train_data, valid_out_data, test_in_data)) > args.threshold).astype(np.float64))
scores = eval_model(best_model, criterion, test_loader, test_out_data, test_unbias, topk=[1, 5, 10, 20, 50, 100, 200],
                    show_progress=args.show_progress, threshold=args.threshold)
print(f'Final scores rank {best_rank}:', scores)
usr_data = (torch.sparse_csr_tensor(train_data.indptr, train_data.indices, train_data.data, train_data.shape,
                                   dtype=torch.float64).to_dense().cuda() > args.threshold).double()
item_data = torch.eye(usr_data.shape[1], dtype=torch.float64).cuda()
usr_embeddings = best_model.get_embedding(usr_data).cpu().detach().numpy()
item_embeddings = best_model.get_embedding(item_data).cpu().detach().numpy()

usr_euc_dist = np.linalg.norm(usr_embeddings, axis=1)
item_euc_dist = np.linalg.norm(item_embeddings, axis=1)

inds = usr_data.sum(dim=1).argsort().cpu().numpy()
usr_euc_dist = usr_euc_dist[inds]

inds = usr_data.sum(dim=0).argsort().cpu().numpy()
item_euc_dist = item_euc_dist[inds]

scores['usr_popquarter_mx'] = np.max(usr_euc_dist[-214:])
scores['usr_popquarter_mean'] = np.mean(usr_euc_dist[-214:])

scores['usr_modpophalf_mx'] = np.max(usr_euc_dist[-1432:-214])
scores['usr_modpophalf_mean'] = np.mean(usr_euc_dist[-1432:-214])

scores['usr_unpopquarter_mx'] = np.max(usr_euc_dist[:-1432])
scores['usr_unpopquarter_mean'] = np.mean(usr_euc_dist[:-1432])

scores['usr_pop_euc@100'] = np.max(usr_euc_dist[-100:])
scores['usr_pop_euc@200'] = np.max(usr_euc_dist[-200:])

scores['item_pop_euc@100'] = np.max(item_euc_dist[-100:])
scores['item_pop_euc@200'] = np.max(item_euc_dist[-200:])

scores['usr_antipop_euc@100'] = np.max(usr_euc_dist[:100])
scores['usr_antipop_euc@200'] = np.max(usr_euc_dist[:200])

scores['item_antipop_euc@100'] = np.max(item_euc_dist[:100])
scores['item_antipop_euc@200'] = np.max(item_euc_dist[:200])

for key, value in scores.items():
    scores[key] = np.float64(value)

with open("model_scores1.txt", "w") as fp:
    json.dump(scores, fp)