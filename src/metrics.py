from collections import defaultdict
import numpy as np
import scipy.sparse
from scipy import sparse
from tqdm import tqdm
import torch


def hr(idx_topk, heldout_batch, threshold):
    batch_users, k = idx_topk.shape

    X_pred = scipy.sparse.csr_matrix((np.full((batch_users * k), True), idx_topk.flatten(), k * np.arange(batch_users + 1)),
                                     dtype='bool', shape=heldout_batch.shape)

    X_plus = (heldout_batch > threshold)

    return np.minimum(X_pred.multiply(X_plus).sum(axis=1).A[:, 0], 1)

def mr(idx_topk, heldout_batch, threshold):
    batch_users, k = idx_topk.shape

    X_pred = scipy.sparse.csr_matrix((np.full((batch_users * k), True), idx_topk.flatten(), k * np.arange(batch_users + 1)),
                                     dtype='bool', shape=heldout_batch.shape)

    X_minus = (heldout_batch < threshold).multiply(heldout_batch > 0)

    return np.minimum(X_pred.multiply(X_minus).sum(axis=1).A[:, 0], 1)


def ndcg(idx_topk, heldout_batch, threshold):
    batch_users, k = idx_topk.shape
    
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    preds = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() > threshold)
    
    dcg = (preds * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])
    return dcg / idcg


def recall(idx_topk, heldout_batch, threshold):
    batch_users, k = idx_topk.shape

    X_pred = scipy.sparse.csr_matrix((np.full((batch_users * k), True), idx_topk.flatten(), k * np.arange(batch_users + 1)),
                                     dtype='bool', shape=heldout_batch.shape)

    X_plus = (heldout_batch > threshold)

    tp = X_plus.multiply(X_pred).sum(axis=1).A[:, 0]
    recall = tp / np.minimum(k, X_plus.sum(axis=1).A[:, 0])

    return recall


def precision(idx_topk, heldout_batch, threshold):
    batch_users, k = idx_topk.shape

    X_pred = scipy.sparse.csr_matrix((np.full((batch_users * k), True), idx_topk.flatten(), k * np.arange(batch_users + 1)),
                                     dtype='bool', shape=heldout_batch.shape)

    X_plus = (heldout_batch > threshold)
    tp = X_plus.multiply(X_pred).sum(axis=1).A[:, 0]
    precision = tp / k

    return precision


def unbiased_recall(idx_topk, heldout_batch, alpha, beta, threshold):
    batch_users, k = idx_topk.shape

    X_pred = scipy.sparse.csr_matrix((np.full((batch_users * k), True), idx_topk.flatten(), k * np.arange(batch_users + 1)),
                                     dtype='bool', shape=heldout_batch.shape)

    X_question = (heldout_batch == 0)
    num_questions = X_question.sum(axis=1).A[:, 0]

    num_pred_questions = X_pred.multiply(X_question).sum(axis=1).A[:, 0]
    num_unpred_questions = num_questions - num_pred_questions

    X_plus = (heldout_batch > threshold)

    tp = X_plus.multiply(X_pred).sum(axis=1).A[:, 0] + num_pred_questions * alpha
    recall = tp / np.minimum(k, X_plus.sum(axis=1).A[:, 0] + num_pred_questions * alpha)

    return recall


def unbiased_precision(idx_topk, heldout_batch, alpha, beta, threshold):
    batch_users, k = idx_topk.shape

    X_pred = scipy.sparse.csr_matrix((np.full((batch_users * k), True), idx_topk.flatten(), k * np.arange(batch_users + 1)),
                                     dtype='bool', shape=heldout_batch.shape)

    X_question = (heldout_batch == 0)
    num_pred_questions = X_pred.multiply(X_question).sum(axis=1).A[:, 0]

    X_plus = (heldout_batch > threshold)
    tp = X_plus.multiply(X_pred).sum(axis=1).A[:, 0] + num_pred_questions * alpha
    recall = tp / k

    return recall


def count_stat(idx_topk, heldout_batch, threshold):
    batch_users, k = idx_topk.shape
    X_pred = scipy.sparse.csr_matrix((np.full((batch_users * k), True), idx_topk.flatten(), k * np.arange(batch_users + 1)),
                                     dtype='bool', shape=heldout_batch.shape)
    X_plus = (heldout_batch > threshold)
    X_minus = (heldout_batch < threshold).multiply(heldout_batch > 0)

    tp = X_plus.multiply(X_pred).sum(axis=1).A[:, 0]
    fp = X_minus.multiply(X_pred).sum(axis=1).A[:, 0]
    tn = X_minus.sum(axis=1).A[:, 0] - fp
    fn = X_plus.sum(axis=1).A[:, 0] - tp

    return tp, fp, tn, fn