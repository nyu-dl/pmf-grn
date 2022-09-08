import argparse
import os
import torch
import numpy as np

from torch.utils.data.dataloader import DataLoader
from pyro import poutine
from collections import defaultdict

from data_processing.dataset import get_sc_dataset, to_cuda
from modules.pmf import PMF_MODULES


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression-path")
    parser.add_argument("--gene-tf-prior-path")
    parser.add_argument("--tf-activity-prior-path")
    parser.add_argument("--save-dir")
    parser.add_argument("--module-name", choices=PMF_MODULES.keys())
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--use-gpu", action="store_true")

    return parser

def simulate_from_prior(args):
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = get_sc_dataset(
        args.expression_path,
        args.gene_tf_prior_path,
        args.tf_activity_prior_path,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size if args.batch_size > 0 else len(dataset))

    module = PMF_MODULES.get(args.module_name)(
        dataset.num_u,
        dataset.dim_u,
        0.1,
        torch.tensor(dataset.V_prior_hparams.X),
        len(dataset),
        args.use_gpu
    )

    sampled_vars = defaultdict(list)
    # will not work properly if more than one batch, bc that would imply multiple samples of V
    # TODO: figure out how to deal with multiple batches
    for i, prior_hparams_U_i, W_i in dataloader:
        if args.use_gpu:
            i, prior_hparams_U_i, W_i = to_cuda(i, prior_hparams_U_i, W_i)
        with torch.no_grad():
            with poutine.trace() as param_capture:
                poutine.uncondition(module.model)(i, prior_hparams_U_i, W_i, 1)
        for node_name, node_dict in param_capture.trace.nodes.items():
            if node_name in ["locals", "globals"]:
                continue
            sampled_vars[node_name].append(node_dict["value"].cpu().numpy())
    for name, value in sampled_vars.items():
        value = np.vstack(value)
        # hack for saving sampled_obs as sc AnnData
        if name == "obs":
            dataset.W.X = value
            dataset.W.write_h5ad(os.path.join(args.save_dir, "sampled_obs.h5ad"))
        else:
            np.save(os.path.join(args.save_dir, "sampled_{}.npy".format(name)), value)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    simulate_from_prior(args)