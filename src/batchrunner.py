import scipy.stats
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .recvae import recall, ndcg, hr, mr, count_stat, precision, \
    unbiased_recall, unbiased_precision


def count_alpha_beta(model, loader, data_tb, topk, show_progress=False, variational=False, threshold=3.5):
    alphas = np.zeros((len(topk), data_tb.shape[0]), dtype=np.float32)
    betas = np.zeros((len(topk), data_tb.shape[0]), dtype=np.float32)

    if show_progress:
        loader = tqdm(loader)

    for i, (batch, _) in enumerate(loader):
        dense_batch = batch.to_dense()
        pos_batch = (dense_batch > threshold).double()
        with torch.no_grad():
            predictions = model(pos_batch)

        if variational:
            predictions = predictions[2]

        predictions[pos_batch == 1] = -np.inf
        pred_arr = predictions.cpu().numpy()
        top_idx = np.argpartition(-pred_arr, max(topk), axis=1)[:, :max(topk)]
        inds = np.argsort(-np.take_along_axis(pred_arr, top_idx, axis=1), axis=1)
        top_idx = np.take_along_axis(top_idx, inds, axis=1)

        batch_size = batch.shape[0]
        idx = i * batch_size
        unbias_batch = data_tb[idx:idx + batch_size]

        for i, k in enumerate(topk):
            tp_, fp_, tn_, fn_ = count_stat(top_idx[:, :k], unbias_batch, threshold)
            alphas[i, idx:idx + batch_size] = tp_ / np.maximum(tp_ + fp_, 1)
            betas[i, idx:idx + batch_size] = fn_ / np.maximum(fn_ + tn_, 1)
    return alphas.mean(axis=1), betas.mean(axis=1)


def train(loader, model, optimizer, criterion, scheduler=None, show_progress=True, threshold=3.5):
    model.train()
    losses = []
    
    if show_progress:
        loader = tqdm(loader)    
    
    for batch, _ in loader:
        optimizer.zero_grad()
        dense_batch = (batch.to_dense() > threshold).double()
        predictions = model(dense_batch)
        loss = criterion(predictions, dense_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    if scheduler is not None:        
        scheduler.step()
    return losses


def eval_model(model, criterion, loader, data_te, data_tb, topk=[100], show_progress=False, variational=False,
               threshold=3.5, only_ndcg=False):
    model.eval()
    scores = defaultdict(list)
    coverage_set = defaultdict(set)
    losses = []

    if show_progress:
        loader = tqdm(loader)

    if not only_ndcg:
        # alpha = tp / (tp + fp)
        # beta = fn / (fn + tn)
        alpha, beta = count_alpha_beta(model, loader, data_tb, topk, show_progress, variational, threshold)

    for i, (batch, _) in enumerate(loader):
        dense_batch = batch.to_dense()
        pos_batch = (dense_batch > threshold).double()

        batch_size = batch.shape[0]
        idx = i * batch_size
        test_batch = data_te[idx:idx + batch_size]
        unbias_batch = torch.tensor((data_tb[idx:idx + batch_size] > threshold).toarray())

        with torch.no_grad():
            predictions = model(pos_batch)

        if variational:
            predictions = predictions[2]

        loss = criterion(predictions, pos_batch)
        losses.append(loss.item())
        # exclude examples from training and unbias
        predictions[pos_batch == 1] = -np.inf
        predictions[unbias_batch == 1] = -np.inf
        pred_arr = predictions.cpu().numpy()

        top_idx = np.argpartition(-pred_arr, max(topk), axis=1)[:, :max(topk)]
        inds = np.argsort(-np.take_along_axis(pred_arr, top_idx, axis=1), axis=1)
        top_idx = np.take_along_axis(top_idx, inds, axis=1)

        for i, k in enumerate(topk):
            scores[f'ndcg@{k}'].append(ndcg(top_idx[:, :k], test_batch, threshold))
            if not only_ndcg:
                scores[f'recall@{k}'].append(recall(top_idx[:, :k], test_batch, threshold))
                scores[f'precision@{k}'].append(precision(top_idx[:, :k], test_batch, threshold))
                scores[f'unbiased_recall@{k}'].append(unbiased_recall(top_idx[:, :k], test_batch, alpha[i], beta[i], threshold))
                scores[f'unbiased_precision@{k}'].append(unbiased_precision(top_idx[:, :k], test_batch, alpha[i], beta[i], threshold))
                scores[f'hr@{k}'].append(hr(top_idx[:, :k], test_batch, threshold))
                scores[f'mr@{k}'].append(mr(top_idx[:, :k], test_batch, threshold))
                coverage_set[f'cov@{k}'].update(np.unique(top_idx[:, :k]))

    results = {metric: np.mean(np.concatenate(score)) for metric, score in scores.items()}
    results.update({'test loss': np.mean(losses)})
    if not only_ndcg:
        for i, k in enumerate(topk):
            results.update({f'alpha@{k}': alpha[i]})
            results.update({f'beta@{k}': beta[i]})

        n_items = batch.size()[1]
        results.update({metric: len(inds) / n_items for metric, inds in coverage_set.items()})
    return results


def report_metrics(scores, epoch=None):
    log_str = f'Epoch: {epoch}' if epoch is not None else 'Scores'
    log = f"{log_str} | " + " | ".join(map(lambda x: f'{x[0]}: {x[1]:.6f}', scores.items()))
    print(log)


def train_edge(loader, model, optimizer, criterion, scheduler=None, show_progress=True, threshold=3.5):
    model.train()
    losses = []

    if show_progress:
        loader = tqdm(loader)

    for batch, inds in loader:
        optimizer.zero_grad()
        dense_batch = (batch.to_dense() > threshold).double()
        predictions = model([dense_batch, inds])
        loss = criterion(predictions, dense_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if scheduler is not None:
        scheduler.step()
    return losses


def eval_edge_model(model, criterion, loader, data_te, data_tb, topk=[100], show_progress=False, variational=False,
                    threshold=3.5, only_ndcg=False):
    model.eval()
    scores = defaultdict(list)
    coverage_set = defaultdict(set)
    losses = []

    if show_progress:
        loader = tqdm(loader)

    if not only_ndcg:
        # alpha = tp / (tp + fp)
        # beta = fn / (fn + tn)
        alpha, beta = count_alpha_beta(model, loader, data_tb, topk, show_progress, variational, threshold)

    for i, (batch, inds) in enumerate(loader):
        dense_batch = batch.to_dense()
        pos_batch = (dense_batch > threshold).double()

        batch_size = batch.shape[0]
        idx = i * batch_size
        test_batch = data_te[idx:idx + batch_size]
        unbias_batch = torch.tensor((data_tb[idx:idx + batch_size] > threshold).toarray())

        with torch.no_grad():
            predictions = model([pos_batch, inds])

        if variational:
            predictions = predictions[2]

        loss = criterion(predictions, pos_batch)
        losses.append(loss.item())
        # exclude examples from training and validation (if any)
        predictions[pos_batch == 1] = -np.inf
        predictions[unbias_batch == 1] = -np.inf
        pred_arr = predictions.cpu().numpy()

        top_idx = np.argpartition(-pred_arr, max(topk), axis=1)[:, :max(topk)]
        inds = np.argsort(-np.take_along_axis(pred_arr, top_idx, axis=1), axis=1)
        top_idx = np.take_along_axis(top_idx, inds, axis=1)

        for i, k in enumerate(topk):
            scores[f'ndcg@{k}'].append(ndcg(top_idx[:, :k], test_batch, threshold))
            if not only_ndcg:
                scores[f'recall@{k}'].append(recall(top_idx[:, :k], test_batch, threshold))
                scores[f'precision@{k}'].append(precision(top_idx[:, :k], test_batch, threshold))
                scores[f'unbiased_recall@{k}'].append(
                    unbiased_recall(top_idx[:, :k], test_batch, alpha[i], beta[i], threshold))
                scores[f'unbiased_precision@{k}'].append(
                    unbiased_precision(top_idx[:, :k], test_batch, alpha[i], beta[i], threshold))
                scores[f'hr@{k}'].append(hr(top_idx[:, :k], test_batch, threshold))
                scores[f'mr@{k}'].append(mr(top_idx[:, :k], test_batch, threshold))
                coverage_set[f'cov@{k}'].update(np.unique(top_idx[:, :k]))

    results = {metric: np.mean(np.concatenate(score)) for metric, score in scores.items()}
    results.update({'test loss': np.mean(losses)})
    if not only_ndcg:
        for i, k in enumerate(topk):
            results.update({f'alpha@{k}': alpha[i]})
            results.update({f'beta@{k}': beta[i]})

        n_items = batch.size()[1]
        results.update({metric: len(inds) / n_items for metric, inds in coverage_set.items()})
    return results


def add_scores(model, usr_data, item_data, ball, scores, edge=False, total_size=None):
    if not edge:
        usr_embeddings = model(usr_data)

        item_embeddings = model(item_data)
    else:
        usr_embeddings = model([usr_data, torch.arange(usr_data.shape[0]).cuda()])

        item_embeddings = model([item_data, torch.arange(total_size, total_size + item_data.shape[0]).cuda()])

    inds = usr_data.sum(dim=1).argsort()
    usr_embeddings = usr_embeddings[inds]
    cum_usr = np.cumsum(usr_data[inds, :].sum(dim=1).cpu().detach().numpy())

    inds = usr_data.sum(dim=0).argsort()
    item_embeddings = item_embeddings[inds]
    cum_item = np.cumsum(usr_data[:, inds].sum(dim=0).cpu().detach().numpy())

    usr_euc_dist = torch.linalg.vector_norm(usr_embeddings, dim=1).cpu().detach().numpy()
    usr_hyp_dist = ball.dist0(usr_embeddings, dim=1).cpu().detach().numpy()

    item_euc_dist = torch.linalg.vector_norm(item_embeddings, dim=1).cpu().detach().numpy()
    item_hyp_dist = ball.dist0(item_embeddings, dim=1).cpu().detach().numpy()

    radius = ball.radius.cpu()

    first_quarter = np.argmax(cum_usr > cum_usr[-1] / 4)
    last_quarter = np.argmax(cum_usr > cum_usr[-1] * 3 / 4)

    scores['usr_popquarter_mx'] = np.max(usr_euc_dist[last_quarter:]) / radius
    scores['usr_popquarter_mean'] = np.mean(usr_euc_dist[last_quarter:]) / radius

    scores['usr_modpophalf_mx'] = np.max(usr_euc_dist[first_quarter:last_quarter]) / radius
    scores['usr_modpophalf_mean'] = np.mean(usr_euc_dist[first_quarter:last_quarter]) / radius

    scores['usr_unpopquarter_mx'] = np.max(usr_euc_dist[:first_quarter]) / radius
    scores['usr_unpopquarter_mean'] = np.mean(usr_euc_dist[:first_quarter]) / radius

    scores['usr_popquarter_mx_hyp'] = np.max(usr_hyp_dist[last_quarter:])
    scores['usr_popquarter_mean_hyp'] = np.mean(usr_hyp_dist[last_quarter:])

    scores['usr_modpophalf_mx_hyp'] = np.max(usr_hyp_dist[first_quarter:last_quarter])
    scores['usr_modpophalf_mean_hyp'] = np.mean(usr_hyp_dist[first_quarter:last_quarter])

    scores['usr_unpopquarter_mx_hyp'] = np.max(usr_hyp_dist[:first_quarter])
    scores['usr_unpopquarter_mean_hyp'] = np.mean(usr_hyp_dist[:first_quarter])

    first_quarter = np.argmax(cum_item > cum_item[-1] / 4)
    last_quarter = np.argmax(cum_item > cum_item[-1] * 3 / 4)

    scores['item_popquarter_mx'] = np.max(item_euc_dist[last_quarter:]) / radius
    scores['item_popquarter_mean'] = np.mean(item_euc_dist[last_quarter:]) / radius

    scores['item_modpophalf_mx'] = np.max(item_euc_dist[first_quarter:last_quarter]) / radius
    scores['item_modpophalf_mean'] = np.mean(item_euc_dist[first_quarter:last_quarter]) / radius

    scores['item_unpopquarter_mx'] = np.max(item_euc_dist[:first_quarter]) / radius
    scores['item_unpopquarter_mean'] = np.mean(item_euc_dist[:first_quarter]) / radius

    scores['item_popquarter_mx_hyp'] = np.max(item_hyp_dist[last_quarter:])
    scores['item_popquarter_mean_hyp'] = np.mean(item_hyp_dist[last_quarter:])

    scores['item_modpophalf_mx_hyp'] = np.max(item_hyp_dist[first_quarter:last_quarter])
    scores['item_modpophalf_mean_hyp'] = np.mean(item_hyp_dist[first_quarter:last_quarter])

    scores['item_unpopquarter_mx_hyp'] = np.max(item_hyp_dist[:first_quarter])
    scores['item_unpopquarter_mean_hyp'] = np.mean(item_hyp_dist[:first_quarter])

    usr_degrees = usr_data.sum(dim=1).sort()[0].cpu()
    item_degrees = usr_data.sum(dim=0).sort()[0].cpu()

    scores['usr_pearson'] = torch.corrcoef(torch.stack((torch.tensor(usr_hyp_dist), usr_degrees)))[0, 1]
    scores['item_pearson'] = torch.corrcoef(torch.stack((torch.tensor(item_hyp_dist), item_degrees)))[0, 1]

    scores['usr_spearman'] = scipy.stats.spearmanr(usr_hyp_dist, usr_degrees).statistic
    scores['item_spearman'] = scipy.stats.spearmanr(item_hyp_dist, item_degrees).statistic

    usr_embeddings_unbiased = usr_embeddings.clone().detach()
    item_embeddings_unbiased = item_embeddings.clone().detach()

    usr_embeddings = ball.mobius_sub(usr_embeddings, usr_embeddings[-1, :])
    item_embeddings = ball.mobius_sub(item_embeddings, item_embeddings[-1, :])

    usr_euc_dist = torch.linalg.vector_norm(usr_embeddings, dim=1).cpu().detach().numpy()
    usr_hyp_dist = ball.dist0(usr_embeddings, dim=1).cpu().detach().numpy()

    item_euc_dist = torch.linalg.vector_norm(item_embeddings, dim=1).cpu().detach().numpy()
    item_hyp_dist = ball.dist0(item_embeddings, dim=1).cpu().detach().numpy()

    radius = ball.radius.cpu()

    first_quarter = np.argmax(cum_usr > cum_usr[-1] / 4)
    last_quarter = np.argmax(cum_usr > cum_usr[-1] * 3 / 4)

    scores['usr_popquarter_mx_biased'] = np.max(usr_euc_dist[last_quarter:]) / radius
    scores['usr_popquarter_mean_biased'] = np.mean(usr_euc_dist[last_quarter:]) / radius

    scores['usr_modpophalf_mx_biased'] = np.max(usr_euc_dist[first_quarter:last_quarter]) / radius
    scores['usr_modpophalf_mean_biased'] = np.mean(usr_euc_dist[first_quarter:last_quarter]) / radius

    scores['usr_unpopquarter_mx_biased'] = np.max(usr_euc_dist[:first_quarter]) / radius
    scores['usr_unpopquarter_mean_biased'] = np.mean(usr_euc_dist[:first_quarter]) / radius

    scores['usr_popquarter_mx_hyp_biased'] = np.max(usr_hyp_dist[last_quarter:])
    scores['usr_popquarter_mean_hyp_biased'] = np.mean(usr_hyp_dist[last_quarter:])

    scores['usr_modpophalf_mx_hyp_biased'] = np.max(usr_hyp_dist[first_quarter:last_quarter])
    scores['usr_modpophalf_mean_hyp_biased'] = np.mean(usr_hyp_dist[first_quarter:last_quarter])

    scores['usr_unpopquarter_mx_hyp_biased'] = np.max(usr_hyp_dist[:first_quarter])
    scores['usr_unpopquarter_mean_hyp_biased'] = np.mean(usr_hyp_dist[:first_quarter])

    first_quarter = np.argmax(cum_item > cum_item[-1] / 4)
    last_quarter = np.argmax(cum_item > cum_item[-1] * 3 / 4)

    scores['item_popquarter_mx_biased'] = np.max(item_euc_dist[last_quarter:]) / radius
    scores['item_popquarter_mean_biased'] = np.mean(item_euc_dist[last_quarter:]) / radius

    scores['item_modpophalf_mx_biased'] = np.max(item_euc_dist[first_quarter:last_quarter]) / radius
    scores['item_modpophalf_mean_biased'] = np.mean(item_euc_dist[first_quarter:last_quarter]) / radius

    scores['item_unpopquarter_mx_biased'] = np.max(item_euc_dist[:first_quarter]) / radius
    scores['item_unpopquarter_mean_biased'] = np.mean(item_euc_dist[:first_quarter]) / radius

    scores['item_popquarter_mx_hyp_biased'] = np.max(item_hyp_dist[last_quarter:])
    scores['item_popquarter_mean_hyp_biased'] = np.mean(item_hyp_dist[last_quarter:])

    scores['item_modpophalf_mx_hyp_biased'] = np.max(item_hyp_dist[first_quarter:last_quarter])
    scores['item_modpophalf_mean_hyp_biased'] = np.mean(item_hyp_dist[first_quarter:last_quarter])

    scores['item_unpopquarter_mx_hyp_biased'] = np.max(item_hyp_dist[:first_quarter])
    scores['item_unpopquarter_mean_hyp_biased'] = np.mean(item_hyp_dist[:first_quarter])

    usr_degrees = usr_data.sum(dim=1).sort()[0].cpu()
    item_degrees = usr_data.sum(dim=0).sort()[0].cpu()

    scores['usr_pearson_biased'] = torch.corrcoef(torch.stack((torch.tensor(usr_hyp_dist), usr_degrees)))[0, 1]
    scores['item_pearson_biased'] = torch.corrcoef(torch.stack((torch.tensor(item_hyp_dist), item_degrees)))[0, 1]

    scores['usr_spearman_biased'] = scipy.stats.spearmanr(usr_hyp_dist, usr_degrees).statistic
    scores['item_spearman_biased'] = scipy.stats.spearmanr(item_hyp_dist, item_degrees).statistic

    return usr_embeddings_unbiased, item_embeddings_unbiased
