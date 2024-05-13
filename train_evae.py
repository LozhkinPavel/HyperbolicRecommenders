import argparse
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm

from src.models import MultipleOptimizer
from src.datareader import read_data
from src.datasets import make_loaders_weak
from src.vae.vae_models import VAE
from src.vae.vae_runner import Trainer
from src.batchrunner import report_metrics, eval_model
from src.random import fix_torch_seed

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

data_dir, data_name = args.data_dir, args.dataname

train_data, valid_in_data, valid_out_data, test_in_data, test_out_data, valid_unbias, test_unbias = read_data(data_dir, data_name)

train_loader, valid_loader, test_loader, train_val_loader = make_loaders_weak(train_data, valid_in_data, valid_out_data,
                                                                              test_in_data, test_out_data, args.batch_size, device)

criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

model = VAE(train_data.shape[1], args.embedding_dim, False).to(device)

trainer = Trainer(model, total_anneal_steps=args.total_anneal_steps, anneal_cap=args.anneal_cap)

encoder_optimizer = torch.optim.Adam(model.component.fc_mean.parameters(), lr=args.learning_rate)
decoder_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.learning_rate)
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
        best_scores = deepcopy(scores)
        best_model = deepcopy(trainer.model)
    report_metrics(scores, epoch)

scores = eval_model(best_model, criterion, test_loader, test_out_data, test_unbias, topk=[1, 5, 10, 20, 50, 100],
                    show_progress=args.show_progress, variational=True, threshold=args.threshold)
print(scores)
