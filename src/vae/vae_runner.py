import torch

from ..batchrunner import eval_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, total_anneal_steps, anneal_cap):
        self.model = model
        self.epoch = 0
        
        self.total_anneal_steps = total_anneal_steps 
        self.anneal_cap = anneal_cap

    def train(self, optimizer, train_loader, threshold=3.5):
        self.model.train()

        for batch, _ in train_loader:
            batch = (batch.to_dense() > threshold).double()
            batch = batch.to(device)
            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, self.anneal_cap * self.epoch / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            self.model.train_step(optimizer, batch, anneal)
        
        self.epoch += 1
        
    def evaluate(self, criterion, loader, data_te, data_tb, topk=[20, 100], show_progress=False, threshold=3.5, only_ndcg=False):
        self.model.eval()
        return eval_model(self.model, criterion, loader, data_te, data_tb, topk=topk, show_progress=show_progress,
                          variational=True, threshold=threshold, only_ndcg=only_ndcg)
