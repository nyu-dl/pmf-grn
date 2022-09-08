import os
import scipy
import torch
import numpy as np
import scanpy as sc

from torch.utils.data.dataset import Dataset
from torch.utils.data._utils.collate import default_collate
from typing import Tuple, Callable

class PMFDataset(Dataset):
    def __init__(self, W: sc.AnnData, U_prior_hparams: sc.AnnData, V_prior_hparams: sc.AnnData):
        super().__init__()
        self.W = W
        self.U_prior_hparams = U_prior_hparams
        self.V_prior_hparams = V_prior_hparams
        self.i = torch.arange(len(W))
        self.W_tensor = torch.tensor(self.W.X)
        self.U_prior_hparams_tensor = torch.tensor(self.U_prior_hparams.X)
        self.V_prior_hparams_tensor = torch.tensor(self.V_prior_hparams.X)


    def __getitem__(self, i: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        W_i = self.W_tensor[i]
        prior_hparams_U_i = self.U_prior_hparams_tensor[i, :]
        i = self.i[i]
        return i, prior_hparams_U_i, W_i

    def __len__(self) -> int:
        return len(self.W)

    @property
    def num_u(self):
        return self.U_prior_hparams.shape[0]

    @property
    def dim_u(self):
        return self.U_prior_hparams.shape[1]

    @property
    def num_v(self):
        return self.V_prior_hparams.shape[0]

    @property
    def dim_v(self):
        return self.V_prior_hparams.shape[1]

class MaskPMFDataset(PMFDataset):
    def __init__(self, W: sc.AnnData, U_prior_hparams: sc.AnnData, V_prior_hparams: sc.AnnData, mask: np.ndarray):
        super().__init__(W, U_prior_hparams, V_prior_hparams)
        self.mask = torch.tensor(mask)

    def __getitem__(self, i: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        return *super().__getitem__(i), self.mask[i]


def get_sc_dataset(
        expression_path: str,
        gene_tf_prior_path: str,
        tf_activity_prior_path: str = "",
        gene_tf_prior_dist_type: str = "",
        tf_activity_prior_dist_type: str = "",
        debug_num_entries: int = -1,
        mask_path: str = None,
        tfa_from_expression: bool = False,
        move_each_batch_to_gpu: bool = False
) -> Tuple[PMFDataset, Callable]:
    expression_data = load_file(expression_path)
    gene_tf_data = load_file(gene_tf_prior_path)
    if tf_activity_prior_path:
        tf_activity_data = load_file(tf_activity_prior_path)
    else:
        tf_activity_data = generate_zeros_tfa(expression_data, gene_tf_data)
    if tfa_from_expression is True:
        intersecting_tf_names = gene_tf_data.var_names.intersection(expression_data.var_names)
        tf_activity_data[:, intersecting_tf_names].X = expression_data[:, intersecting_tf_names].X

    expression_data, gene_tf_data = align_data(expression_data, gene_tf_data, "var")
    expression_data, tf_activity_data = align_data(expression_data, tf_activity_data, "obs")

    if debug_num_entries > 0:
        expression_data = expression_data[:debug_num_entries, :]
        #gene_tf_data = gene_tf_data[:debug_num_entries, :]
        tf_activity_data = tf_activity_data[:debug_num_entries, :]

    gene_tf_prior_hparams = convert_to_prior_hparams(gene_tf_data, gene_tf_prior_dist_type)
    tf_activity_prior_hparams = convert_to_prior_hparams(tf_activity_data, tf_activity_prior_dist_type)

    expression_data = to_dense(expression_data.copy())
    tf_activity_prior_hparams = to_dense(tf_activity_prior_hparams.copy())
    gene_tf_prior_hparams = to_dense(gene_tf_prior_hparams.copy())

    if mask_path is None:
        dataset = PMFDataset(expression_data, tf_activity_prior_hparams, gene_tf_prior_hparams)
        coll_fn = lambda data: no_mask_collate_fn_gpu_wrapper(data, move_each_batch_to_gpu)
    else:
        mask = create_mask(mask_path, expression_data.shape[0], expression_data.shape[1])
        dataset = MaskPMFDataset(expression_data, tf_activity_prior_hparams, gene_tf_prior_hparams, mask)
        coll_fn = lambda data: default_collate_gpu_wrapper(data, move_each_batch_to_gpu)
    return dataset, coll_fn

def load_file(file_path: str) -> sc.AnnData:
    data = sc.read_h5ad(file_path)
    return data

def generate_zeros_tfa(expression_data: sc.AnnData, gene_tf_data: sc.AnnData) -> sc.AnnData:
    tfa = sc.AnnData(np.zeros((expression_data.shape[0], gene_tf_data.shape[1])))
    tfa.obs = expression_data.obs
    tfa.var = gene_tf_data.var
    tfa.X = scipy.sparse.csr_matrix(tfa.X)
    return tfa

def to_dense(data: sc.AnnData):
    if type(data.X) == scipy.sparse.csr_matrix:
        data.X = data.X.todense()
    data.X = np.array(data.X)
    return data

def align_data(expression: sc.AnnData, other: sc.AnnData, alignment_type: str = "obs") -> Tuple[sc.AnnData, sc.AnnData]:
    if alignment_type == "var":
        intersection = other.obs.index.intersection(expression.var.index)
        expression = expression[:, intersection]
    else:
        intersection = other.obs.index.intersection(expression.obs.index)
        expression = expression[intersection]

    other = other[intersection]

    return expression, other

def convert_to_prior_hparams(data: sc.AnnData, dist_type: str = "exponential", eps: float = 0.001) -> sc.AnnData:
    if dist_type == "exponential":
        data.X = 1/(data.X + eps)
    elif dist_type == "lognormal":
        pass
    return data

def to_lognormal(mean: torch.Tensor, variance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates mean and variance of log X from mean and variance of X. For use in lognormal distribution"""
    lognormal_variance = torch.log(variance/torch.square(mean) + 1)
    lognormal_mean = torch.log(mean) - torch.div(lognormal_variance, 2)
    return lognormal_mean, lognormal_variance

def create_mask(mask_path, num_rows, num_cols):
    if os.path.exists(mask_path):
        mask = torch.BoolTensor(scipy.sparse.load_npz(mask_path).todense())[:num_rows, :num_cols]
    else:
        mask = torch.distributions.Categorical(torch.tensor([0.1, 0.9])).sample((num_rows, num_cols))
        mask = mask.bool()
        scipy.sparse.save_npz(mask_path, scipy.sparse.csr_matrix(mask.numpy()))
    return mask

def to_cuda(*variables):
    return [variable.cuda() if variable is not None else None for variable in variables]

def default_collate_gpu_wrapper(data, move_to_gpu):
    batch = default_collate(data)
    if move_to_gpu:
        batch = to_cuda(*batch)
    return batch

def no_mask_collate_fn_gpu_wrapper(data, move_to_gpu):
    return *default_collate_gpu_wrapper(data, move_to_gpu), None