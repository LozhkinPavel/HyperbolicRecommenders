import numpy as np
import torch
import os


def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=10)
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(mode=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
