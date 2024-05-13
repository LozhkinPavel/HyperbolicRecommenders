import numpy as np
from scipy.sparse import isspmatrix_csr
import torch
import torch.utils.data as td
from scipy.sparse import vstack


class MyDataset(td.Dataset):
    def __init__(self, matrix, device, shift=0):
        self.matrix = matrix
        self.device = device
        # self.matrix = torch.sparse_csr_tensor(matrix.indptr, matrix.indices, matrix.data, matrix.shape, dtype=torch.float64)
        self.shift = shift

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, index):
        return torch.squeeze(torch.tensor(self.matrix[index, :].toarray(), dtype=torch.float64), dim=0).to(self.device),\
               torch.tensor(index + self.shift).to(self.device)


def make_loaders_strong(train_data, valid_in_data, valid_out_data, test_in_data, test_out_data, batch_size, device):
    train_loader = td.DataLoader(MyDataset(train_data, device), batch_size=batch_size, shuffle=False)
    train_val_loader = td.DataLoader(MyDataset(vstack([train_data, valid_out_data]), device), batch_size=batch_size,
                                     shuffle=False)
    val_loader = td.DataLoader(MyDataset(valid_in_data, device, shift=train_data.shape[0]), batch_size=batch_size,
                               shuffle=False)
    test_loader = td.DataLoader(MyDataset(test_in_data, device, shift=train_data.shape[0] + valid_in_data.shape[0]),
                                batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_val_loader


def make_loaders_weak(train_data, valid_in_data, valid_out_data, test_in_data, test_out_data, batch_size, device):
    train_loader = td.DataLoader(MyDataset(vstack([train_data, valid_in_data, test_in_data]), device),
                                 batch_size=batch_size, shuffle=False)
    train_val_loader = td.DataLoader(MyDataset(vstack([train_data, valid_out_data, test_in_data]), device),
                                     batch_size=batch_size, shuffle=False)
    val_loader = td.DataLoader(MyDataset(valid_in_data, device, shift=train_data.shape[0]), batch_size=batch_size,
                               shuffle=False)
    test_loader = td.DataLoader(MyDataset(test_in_data, device, shift=train_data.shape[0] + valid_in_data.shape[0]),
                                batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_val_loader
