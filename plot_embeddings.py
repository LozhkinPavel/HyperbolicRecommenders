import os
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd

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
from src.vae.vae_runner import HypVaeDataset
import matplotlib.pyplot as plt
from src.geoopt_plusplus.modules import PoincareLinear, UnidirectionalPoincareMLR
from matplotlib import collections as mc
import matplotlib as mpl
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--datapack", type=str, required=True, choices=["persdiff", "urm"])
parser.add_argument("--dataname", type=str, required=True) # depends on choice of data pack
parser.add_argument("--encoder", type=str, required=True, choices=['HypLinear', 'Mobius', 'PoincareLinear'])
parser.add_argument("--decoder", type=str, required=True, choices=['HypLinear', 'Mobius', 'PoincareLinear', 'HypMLR'])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--threshold", type=float, default=3.5)
parser.add_argument("--c", type=float, default=0.5)
parser.add_argument("--gamma", type=float, default=0.7)
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

# criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()

data_dir, data_pack, data_name = args.data_dir, args.datapack, args.dataname

train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = read_data(data_dir, data_pack, data_name)

train_loader, valid_loader, test_loader, train_val_loader = make_loaders_weak(train_data, valid_in_data, valid_out_data,
                                                                         test_in_data, test_out_data, args.batch_size)
total_size = train_data.shape[0] + valid_in_data.shape[0] + test_in_data.shape[0]

criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

layer_factories = {'HypLinear': HypLinear, 'Mobius': MobiusLinear, 'PoincareLinear': PoincareLinear,
                   'HypMLR': UnidirectionalPoincareMLR}

optimizers = {'Mobius': geoopt.optim.RiemannianAdam, 'HypLinear': torch.optim.Adam, 'PoincareLinear': torch.optim.Adam,
              'HypMLR': torch.optim.Adam}

fig1, ax = plt.subplots(2, 7, figsize=(140, 40))
ndcgs = []
time = []
scores_list = []

for z, embedding_dim in tqdm(enumerate([4, 8, 16, 32, 64, 128, 256])):
    args.embedding_dim = embedding_dim
    ball = PoincareBall(args.c)
    encoder_layer = layer_factories[args.encoder](train_data.shape[1], args.embedding_dim, bias=True, ball=ball)
    decoder_layer = layer_factories[args.decoder](args.embedding_dim, train_data.shape[1], bias=True, ball=ball)
    model = nn.Sequential(ExpMap0(ball), encoder_layer, decoder_layer).cuda()

    encoder_optimizer = optimizers[args.encoder](model[1].parameters(), lr=args.learning_rate)
    decoder_optimizer = optimizers[args.decoder](model[2].parameters(), lr=args.learning_rate)

    optimizer = MultipleOptimizer(encoder_optimizer, decoder_optimizer)

    scheduler = None

    show_progress = args.show_progress

    best_ndcg = -np.inf
    for epoch in range(args.epochs):
        losses = train(train_loader, model, optimizer, criterion,
                       masked_loss=False, show_progress=show_progress, threshold=args.threshold)
        scores = eval_model(model, criterion, valid_loader, valid_out_data, topk=[1, 10, 20], show_progress=args.show_progress, threshold=args.threshold)
        scores.update({'train loss': np.mean(losses)})
        if scores['ndcg@10'] > best_ndcg:
            best_ndcg = scores['ndcg@10']
            best_model = deepcopy(model)
            cur_time = epoch
        # report_metrics(scores, epoch)
    time.append(cur_time)
    ndcgs.append(best_ndcg)
    new_model = nn.Sequential(ExpMap0(ball), best_model[1],
                              PoincareLinear(args.embedding_dim, 2, bias=True, ball=ball),
                              PoincareLinear(2, args.embedding_dim, bias=True, ball=ball), best_model[2]).cuda()

    for param in new_model[1].parameters():
        param.requires_grad = False
    for param in new_model[4].parameters():
        param.requires_grad = False

    print("Training new model")

    optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

    scheduler = None
    if scheduler_on:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )

    for epoch in range(args.epochs):
        losses = train(train_loader, new_model, optimizer, criterion,
                       masked_loss=False, show_progress=show_progress, threshold=args.threshold)
        scores = eval_model(new_model, criterion, valid_loader, valid_out_data, topk=[10, 20], show_progress=args.show_progress, threshold=args.threshold)
        scores.update({'train loss': np.mean(losses)})
        # report_metrics(scores, epoch)

    # torch.save(model, 'model')
    # torch.save(new_model, 'new_model')

    # model = torch.load('UMAP_models/model')
    # new_model = torch.load('UMAP_models/new_model')

    device = torch.device('cuda')

    usr_data = (torch.sparse_csr_tensor(train_data.indptr, train_data.indices, train_data.data, train_data.shape,
                                       dtype=torch.float64).to(device).to_dense()).double()
    item_data = torch.eye(usr_data.shape[1], dtype=torch.float64).to(device)

    scores_list.append({})
    usr_embeddings, item_embeddings = add_scores(new_model[:3], usr_data, item_data, ball, scores_list[-1])

    radius = ball.radius.cpu()

    usr_degrees = usr_data.sum(dim=1).cpu().numpy()
    inds = usr_degrees.argsort()[::10]
    usr_data = usr_data[inds]
    usr_embeddings = usr_embeddings[inds]

    ax[0][z].scatter(x=usr_embeddings[:, 0], y=usr_embeddings[:, 1], linewidths=0.5, cmap='Reds', c=usr_degrees[inds])
    ax[0][z].set_xlim(-1.25 * radius, 1.25 * radius)
    ax[0][z].set_ylim(-1.25 * radius, 1.25 * radius)
    ax[0][z].set_title(embedding_dim)
    ax[0][z].add_patch(plt.Circle((0, 0), radius, fill=False))

    item_degrees = usr_data.sum(dim=0).cpu().numpy()
    inds = item_degrees.argsort()
    item_embeddings = item_embeddings[inds]

    ax[1][z].scatter(x=item_embeddings[:, 0], y=item_embeddings[:, 1], linewidths=0.5, cmap='Blues', c=item_degrees[inds])
    ax[1][z].add_patch(plt.Circle((0, 0), radius, fill=False))
    ax[1][z].set_xlim(-1.25 * radius, 1.25 * radius)
    ax[1][z].set_ylim(-1.25 * radius, 1.25 * radius)

    edges = []

    for usr in range(usr_data.shape[0]):
        similarities = (usr_data * usr_data[None, usr, :])
        sims_sort = similarities.sum(dim=1).argsort()
        usr_pt = (usr_embeddings[usr, 0], usr_embeddings[usr, 1])
        for i in range(1):
            edges.append([usr_pt, (usr_embeddings[sims_sort[- i - 2], 0], usr_embeddings[sims_sort[- i - 2], 1])])

    lc = mc.LineCollection(edges, colors='g', linewidths=0.1)
    ax[0][z].add_collection(lc)
    # edges = []
    #
    # for item in range(usr_data.shape[1]):
    #     similarities = (usr_data * usr_data[:, item, None])
    #     inds = similarities.sum(dim=0).argsort()
    #     item_pt = (item_embeddings[item, 0], item_embeddings[item, 1])
    #     for i in range(1):
    #        edges.append([item_pt, (item_embeddings[inds[- i - 2], 0], item_embeddings[inds[- i - 2], 1])])
    #
    # lc = mc.LineCollection(edges, colors='g', linewidths=0.1)
    # ax[1][z].add_collection(lc)
    # for usr in range(usr_data.shape[0]):
    #     usr_pt = (usr_embeddings[usr, 0], usr_embeddings[usr, 1])
    #     item = inds[np.random.randint(0, inds.shape[0])]
    #     t = 0
    #     while usr_data[usr, item] == 0 and t < 10000:
    #         item = inds[np.random.randint(0, inds.shape[0])]
    #         t += 1
    #     if t != 10000:
    #         edges.append([usr_pt, (item_embeddings[item, 0], item_embeddings[item, 1])])
    #

plt.show()
plt.savefig('image1.png')
print(scores_list)
print("NDCG's:", ndcgs)
print("Time to converge:", time)


# fig2, ax2 = plt.subplots(1, 1, figsize=(20, 20))
# # mapper = umap.UMAP().fit(embeddings)
# ax1.scatter(x=item_embeddings[:, 0], y=item_embeddings[:, 1], linewidths=0.5, cmap='Blues', c=item_degrees, vmin=0, vmax=100)
# ax2.set_xlim(-1.25 * args.c, 1.25 * args.c)
# ax2.set_ylim(-1.25 * args.c, 1.25 * args.c)
# # umap.plot.points(mapper, ax=ax)
# ax2.add_patch(plt.Circle((0, 0), args.c, fill=False))
# plt.show()
# plt.savefig('image2.png')
