import json

import argparse
from copy import deepcopy

import torch
import numpy as np
import geoopt

from src.models import HypLinear, MultipleOptimizer, MobiusLinear
from src.datareader import read_data
from src.datasets import make_loaders_weak
from src.geoopt_plusplus import UnidirectionalPoincareMLR, PoincareLinear
from src.vae.vae_models import HyperbolicVAE
from src.vae.vae_runner import Trainer
from src.batchrunner import eval_model, add_scores
from src.random import fix_torch_seed
from src.geoopt_plusplus.manifolds import PoincareBall

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataname", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data/")
parser.add_argument("--threshold", type=float, default=3.5)
parser.add_argument("--num_setups", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--c", type=float, default=0.005)
parser.add_argument('--total_anneal_steps', type=int, default=10)
parser.add_argument('--anneal_cap', type=float, default=0.2)
parser.add_argument("--show_progress", default=False, action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rates = np.logspace(-4, -2, 3)
embedding_dims = [16, 32, 64, 128]
curvatures = [0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
anneal_caps = [0.05]

# learning_rates = [1e-3]
# embedding_dims = [32]
# curvatures = [0.5]
# anneal_caps = [0.05]

lr_grid, embedding_dim_grid, curvature_grid, anneal_grid = np.meshgrid(learning_rates, embedding_dims, curvatures, anneal_caps)

lr_grid = lr_grid.flatten()
embedding_dim_grid = embedding_dim_grid.flatten()
curvature_grid = curvature_grid.flatten()
anneal_grid = anneal_grid.flatten()

fix_torch_seed(args.seed)

data_dir, data_name = args.data_dir, args.dataname

train_data, valid_in_data, valid_out_data, test_in_data, test_out_data, valid_unbias, test_unbias = read_data(data_dir, data_name)

train_loader, valid_loader, test_loader, train_val_loader = make_loaders_weak(train_data, valid_in_data, valid_out_data,
                                                                              test_in_data, test_out_data, args.batch_size, device)
total_size = train_data.shape[0] + valid_in_data.shape[0] + test_in_data.shape[0]

layer_factories = {'HypLinear': HypLinear, 'Mobius': MobiusLinear, 'PoincareLinear': PoincareLinear,
                   'HypMLR': UnidirectionalPoincareMLR}

optimizers = {'Mobius': geoopt.optim.RiemannianAdam, 'HypLinear': torch.optim.Adam, 'PoincareLinear': torch.optim.Adam,
              'HypMLR': torch.optim.Adam}

criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

if data_name == 'ml1m':
    usr_data = torch.tensor((train_data > args.threshold).to_array(), dtype=torch.float64).to(device)
    item_data = torch.eye(usr_data.shape[1], dtype=torch.float64).to(device)

models_scores = {}
models_best_args = {}

for encoder in tqdm(['PoincareLinear']):
    for decoder in ['HypMLR']:
# for encoder in tqdm(['HypLinear']):
#     for decoder in ['HypMLR']:
        best_ndcg = -np.inf
        for i in tqdm(np.random.choice(np.arange(lr_grid.shape[0]), size=args.num_setups, replace=False)):
            args.anneal_cap = anneal_grid[i]
            args.learning_rate = lr_grid[i]
            args.embedding_dim = embedding_dim_grid[i]
            args.c = curvature_grid[i]
            ball = PoincareBall(args.c)
            model = HyperbolicVAE(train_data.shape[1], args.embedding_dim, ball, layer_factories[encoder],
                                  layer_factories[decoder], False).to(device)

            trainer = Trainer(model, total_anneal_steps=args.total_anneal_steps, anneal_cap=args.anneal_cap)

            encoder_optimizer = optimizers[encoder](model.component.fc_mean.parameters(), lr=args.learning_rate)
            decoder_optimizer = optimizers[decoder](model.decoder.parameters(), lr=args.learning_rate)
            var_optimizer = torch.optim.Adam(model.component.fc_logvar.parameters(), lr=args.learning_rate)

            optimizer = MultipleOptimizer(encoder_optimizer, decoder_optimizer, var_optimizer)

            scheduler = None
            cur_best_ndcg = -np.inf
            for epoch in range(1, args.epochs + 1):
                trainer.train(optimizer=optimizer, train_loader=train_loader,
                                             threshold=args.threshold)
                scores = eval_model(trainer.model, criterion, valid_loader, valid_out_data, valid_unbias, topk=[10],
                                    show_progress=args.show_progress, variational=True, threshold=args.threshold, only_ndcg=True)
                if scores['ndcg@10'] > cur_best_ndcg:
                    cur_best_ndcg = scores['ndcg@10']
                    cur_val_scores = deepcopy(scores)
                # report_metrics(scores, epoch)

            if cur_best_ndcg > best_ndcg:
                best_ndcg = cur_best_ndcg
                best_args = deepcopy(args)
                val_scores = deepcopy(cur_val_scores)

        ball = PoincareBall(best_args.c)
        best_model = HyperbolicVAE(train_data.shape[1], best_args.embedding_dim, ball, layer_factories[encoder],
                              layer_factories[decoder], False).to(device)

        encoder_optimizer = optimizers[encoder](best_model.component.fc_mean.parameters(), lr=best_args.learning_rate)
        decoder_optimizer = optimizers[decoder](best_model.decoder.parameters(), lr=best_args.learning_rate)
        var_optimizer = torch.optim.Adam(best_model.component.fc_logvar.parameters(), lr=best_args.learning_rate)

        optimizer = MultipleOptimizer(encoder_optimizer, decoder_optimizer, var_optimizer)
        trainer = Trainer(best_model, total_anneal_steps=best_args.total_anneal_steps, anneal_cap=best_args.anneal_cap)
        scheduler = None

        best_ndcg = -np.inf
        for epoch in range(args.epochs):
            trainer.train(optimizer=optimizer, train_loader=train_val_loader,
                          threshold=args.threshold)
            scores = eval_model(trainer.model, criterion, test_loader, test_out_data, test_unbias, topk=[10],
                                      show_progress=args.show_progress, variational=True, threshold=args.threshold, only_ndcg=True)
            if scores['ndcg@10'] > best_ndcg:
                cur_best_model = deepcopy(best_model)
                best_ndcg = scores['ndcg@10']

        best_model = deepcopy(cur_best_model)
        best_scores = eval_model(best_model, criterion, test_loader, test_out_data, test_unbias, topk=[1, 5, 10, 20, 50, 100],
                                      show_progress=args.show_progress, variational=True, threshold=args.threshold)
        if data_name == 'ml1m':
            sample_model = nn.Sequential(best_model.component, SampleLayer(best_model.ball))
            add_scores(sample_model, usr_data, item_data, best_model.ball, best_scores)

        for key, value in val_scores.items():
            best_scores['val_' + key] = value

        models_scores[encoder + '+' + decoder] = best_scores
        models_best_args[encoder + '+' + decoder] = vars(best_args)

        print(encoder + '+' + decoder, 'best ndcg@10 is', best_scores['ndcg@10'])

        for _, metrics in models_scores.items():
            for key, value in metrics.items():
                metrics[key] = np.float64(value)

        for key, value in models_best_args.items():
            new_args = {}
            new_args['c'] = np.float64(value['c'])
            new_args['learning_rate'] = np.float64(value['learning_rate'])
            new_args['embedding_dim'] = np.float64(value['embedding_dim'])
            new_args['anneal_cap'] = value['anneal_cap']
            models_best_args[key] = new_args

        with open("vae_model_scores.txt", "w") as fp:
            json.dump(models_scores, fp)

        with open("vae_model_best_args.txt", "w") as fp:
            json.dump(models_best_args, fp)

