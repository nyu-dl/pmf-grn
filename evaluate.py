import argparse, os
import pandas as pd
import scanpy as sc
import torch
import pyro

from collections import OrderedDict
from functools import reduce
from inferelator.postprocessing.model_metrics import CombinedMetric as CM

def logit_normal_to_mean_and_var(mean_logit: torch.tensor, std_logit: torch.tensor, num_samples: int):
    logit_variable = torch.distributions.Normal(mean_logit, std_logit).sample((num_samples,))
    variable = torch.sigmoid(logit_variable)
    var, mean = torch.var_mean(variable, 0)
    return mean, var


def get_A_means_dfs(exp_dirs, num_sampling_iters=100):
    all_A_means = []
    for dname in exp_dirs:
        pyro.get_param_store().clear()
        pyro.get_param_store().load(os.path.join(dname, "best_iter.params"), map_location="cpu")
        A_means, A_vars = logit_normal_to_mean_and_var(
            pyro.get_param_store()['A_means'].detach(),
            pyro.get_param_store()['A_stds'].detach(),
            num_sampling_iters
        )
        pyro.get_param_store().clear()
        V_obs = pd.read_csv(os.path.join(dname, "V_obs_names.csv"), sep=',', header=None)
        V_vars = pd.read_csv(os.path.join(dname, "V_var_names.csv"), sep=',', header=None)
        A_means_df = pd.DataFrame(data=A_means.cpu().numpy(), columns=V_vars[0], index=V_obs[0])
        all_A_means.append(A_means_df)


    combined_A_means = reduce(lambda a, b: a.add(b, fill_value=0), all_A_means)/len(all_A_means)
    combined_A_means_df = pd.DataFrame(
        data=combined_A_means,
        columns=V_vars[0],
        index=V_obs[0]
    )

    A_means_dfs = OrderedDict()
    for i, dname in enumerate(exp_dirs):
        A_means_dfs[dname] = all_A_means[i]
    A_means_dfs['combined'] = combined_A_means_df
    return A_means_dfs


def get_separate_and_combined_auprcs_A(A_means_dfs, gold_standard_path, filter_method='keep_all_gold_standard'):
    # gold_standard and params file should have axes: genes x tfs
    auprc_results = OrderedDict()
    gold_standard = sc.read_csv(gold_standard_path, delimiter="\t", first_column_names=True).to_df()
    for name, A_means_df in A_means_dfs.items():
        metrics = CM([A_means_df], gold_standard, filter_method)
        auprc_results[name] = metrics.aupr
    return auprc_results

def main():
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    A_means_dfs = get_A_means_dfs(args.exp_dirs, args.num_sampling_iters)
    for i, (name, df) in enumerate(A_means_dfs.items()):
        if name == "combined":
            A_means_dfs[name].to_csv(os.path.join(args.output_dir, "inferred_consensus_grn.tsv"), sep="\t")
        else:
            A_means_dfs[name].to_csv(
                os.path.join(args.output_dir, "inferred_{}_grn.tsv".format(args.expression_names[i])), sep="\t"
            )
    if args.gold_standard_path is not None:
        auprcs = get_separate_and_combined_auprcs_A(A_means_dfs, args.gold_standard_path)
        with open(os.path.join(args.output_dir, "auprcs.txt"), 'w') as f:
            for i, (name, auprc) in enumerate(auprcs.items()):
                if name == "combined":
                    f.write("consensus: " + str(auprc) + "\n")
                else:
                    f.write(args.expression_names[i] + ": " + str(auprc) + "\n")
        return auprcs

parser = argparse.ArgumentParser()
parser.add_argument("--exp-dirs", nargs="+")
parser.add_argument("--expression-names", nargs="+", help="must be in the same order as for --exp-dirs")
parser.add_argument("--gold-standard-path", default=None)
parser.add_argument("--num-sampling-iters", type=int, default=500)
parser.add_argument("--output-dir")

if __name__ == "__main__":
    main()
