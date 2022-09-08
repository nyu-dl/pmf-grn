#!/usr/bin/env python

import argparse
import pandas as pd
import torch
import pyro
from inferelator.postprocessing.model_metrics import CombinedMetric as CM

def lognormal_to_mean(mean_log: torch.tensor, std_log: torch.tensor):
    """
    Convert distributions to point estimates for V matrix (TFs x Genes)
    :param mean_log: torch.tensor
        mean of the log of the random variable
    :param var_log: torch.tensor
        variance of the log of the random variable
    """
    var_log = std_log**2
    mu = torch.exp(mean_log + (var_log/2))
    return mu

def lognormal_to_var(mean_log: torch.tensor, std_log: torch.tensor):
    var_log = std_log**2
    var = (torch.exp(var_log)-1) * torch.exp(2 * mean_log + var_log)
    return var

def logit_normal_to_mean_and_var(mean_logit: torch.tensor, std_logit: torch.tensor, num_samples: int):
    logit_variable = torch.distributions.Normal(mean_logit, std_logit).sample((num_samples,))
    variable = torch.sigmoid(logit_variable)
    var, mean = torch.var_mean(variable, 0)
    return mean, var

def get_CM_A(
        V_obs,
        V_vars,
        gold_standard,
        num_sampling_iters=100,
        positive_class=None,
        filter_method='keep_all_gold_standard'
):
    A_means, A_vars = logit_normal_to_mean_and_var(
        pyro.get_param_store()['A_means'].detach(),
        pyro.get_param_store()['A_stds'].detach(),
        num_sampling_iters
    )
    if positive_class is not None:
        A_means = A_means == positive_class
    to_eval = pd.DataFrame(
        data=A_means.cpu().numpy(),
        columns=V_vars[0],
        index=V_obs[0]
    )
    metrics = CM([to_eval], gold_standard.to_df(), filter_method)
    return metrics

def main():
    cfg = handle_args()

    pyro.get_param_store().load(cfg.output_matrix, map_location='cpu')
    #TODO: resolve ambiguity where variables named *_log_V could represent *_logit_V depending on distribution type
    mean_log_V = pyro.get_param_store()["V_means"]
    std_log_V = pyro.get_param_store()["V_vars"]
    V_obs = pd.read_csv(cfg.V_obs, sep=',', header=None)
    V_vars = pd.read_csv(cfg.V_var, sep=',', header=None)
    gold_standard = pd.read_csv(cfg.gold_standard, sep="\t", index_col=0)

    if cfg.V_dist_type == "log_normal":
        V_mat = lognormal_to_mean(mean_log_V, std_log_V)
    elif cfg.V_dist_type == "logit_normal":
        V_mat = logit_normal_to_mean_and_var(mean_log_V, std_log_V, cfg.num_samples)[0]


    V_matrix = pd.DataFrame(data=V_mat.detach().numpy(), columns=V_vars[0], index=V_obs[0])

    metrics = CM([V_matrix], gold_standard)
    metrics.output_curve_pdf(cfg.output_path, 'V_matrix_metrics.png')

    # mean_log_U = pyro.get_param_store()["U_means"]
    # var_log_U = pyro.get_param_store()["U_vars"]
    # assert that mean_log_V.shape and var_log_V.shape are the same
    # assert that mean_log_U.shape and var_log_U.shape are the same


def handle_args():
    parser = argparse.ArgumentParser(description="PMF Evaluation")
    parser.add_argument("--output_matrix", help="distributions from PMF model")
    parser.add_argument("--gold_standard", help="known gold standard matrix")
    parser.add_argument("--V_var", help="TF names for V matrix")
    parser.add_argument("--V_obs", help="Gene names for V matrix")
    parser.add_argument("--V_dist_type", choices=["log_normal", "logit_normal"], default="log_normal")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples for MC estimate of mean and var")
    parser.add_argument("--output_path", default=".", help="Filename for output evaluation metrics")
    ret = parser.parse_args()
    return ret

if __name__ == "__main__":
    main()
