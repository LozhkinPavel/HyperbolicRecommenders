import os
import numpy as np
from scipy.sparse import csr_matrix


def read_data(data_dir, dataname):
    if dataname == 'ml1m':
        n_test_users = 1000
    elif dataname == 'amazon_videogames':
        n_test_users = 3000
    else:
        raise ValueError("Unrecognized dataname")

    train, heldout, unbias = read_urm_data(os.path.join(data_dir, dataname))

    train_data = train[:-2 * n_test_users]
    valid_in_data = train[-2 * n_test_users:-n_test_users]
    valid_out_data = heldout[-2 * n_test_users:-n_test_users] + valid_in_data
    test_in_data = train[-n_test_users:]
    test_out_data = heldout[-n_test_users:] + test_in_data

    valid_unbias = unbias[-2 * n_test_users:-n_test_users]
    test_unbias = unbias[-n_test_users:]

    assert (valid_in_data.getnnz(axis=1) > 0).all()
    assert (valid_out_data.getnnz(axis=1) > 0).all()

    assert np.any(valid_in_data.toarray() != valid_out_data.toarray())

    return train_data, valid_in_data, valid_out_data, test_in_data, test_out_data, valid_unbias, test_unbias


def npz_to_csr(data):
    assert data['format'].item().decode() == 'csr'
    matrix = csr_matrix(
        (data['data'], data['indices'], data['indptr']),
        shape=data['shape'],
    )
    if not matrix.has_sorted_indices:
        matrix.sort_indices()
    return matrix


def read_urm_data(data_dir):
    files = ["URM_train.npz", "URM_heldout.npz", "URM_unbias.npz"]
    return [npz_to_csr(np.load(os.path.join(data_dir, file))) for file in files]
