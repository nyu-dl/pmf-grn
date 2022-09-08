import argparse, os, copy
import pyro

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from evaluation_metrics.evaluation_V import logit_normal_to_mean_and_var
from inferelator.postprocessing.model_metrics import RankSummingMetric, RankSummaryPR

RankSummingMetricCopy = copy.deepcopy(RankSummingMetric)

@staticmethod
def compute_combined_confidences(rankable_data):
    """
    Same as in original module's method except use nanmin to calculate min_element
    """
    # Create an 0s dataframe shaped to the data to be ranked
    combine_conf = pd.DataFrame(np.zeros(rankable_data[0].shape),
                                index=rankable_data[0].index,
                                columns=rankable_data[0].columns)

    for replicate in rankable_data:
        # Flatten and rank based on the beta error reductions
        ranked_replicate = np.reshape(pd.DataFrame(replicate.values.flatten()).rank().values, replicate.shape)
        # Sum the rankings for each bootstrap
        combine_conf += ranked_replicate

    # Convert rankings to confidence values
    min_element = np.nanmin(combine_conf.values.flatten())
    combine_conf = (combine_conf - min_element) / (len(rankable_data) * combine_conf.size - min_element)
    return combine_conf

RankSummingMetricCopy.compute_combined_confidences = compute_combined_confidences

def get_calibration_auprc(to_eval, gold_standard, filter_method='overlap'):
    metrics = RankSummingMetricCopy([to_eval], gold_standard, filter_method)
    data = RankSummaryPR.calculate_precision_recall(metrics.filtered_data)
    auprc = RankSummaryPR.calculate_aupr(data)
    return auprc

def plot_fig(auprcs, fig_output_path):
    plt.plot(np.arange(1, len(auprcs)+1) * 100/len(auprcs), auprcs)
    plt.xlim(0, 100)
    plt.grid()
    plt.xlabel("Percentile Cutoff", fontsize=18)
    plt.ylabel("AUPRC", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_output_path)

def evaluate_calibration(args):
    gold_standard = sc.read_csv(args.gold_standard_path, delimiter="\t", first_column_names=True).to_df()

    V_obs = pd.read_csv(os.path.join(args.exp_dir, "V_obs_names.csv"), sep=',', header=None)
    V_vars = pd.read_csv(os.path.join(args.exp_dir, "V_var_names.csv"), sep=',', header=None)

    pyro.get_param_store().clear()
    pyro.get_param_store().load(os.path.join(args.exp_dir, "best_iter.params"), map_location="cpu")
    A_means, A_vars = logit_normal_to_mean_and_var(
        pyro.get_param_store()['A_means'].detach(),
        pyro.get_param_store()['A_stds'].detach(),
        args.num_sampling_iters
    )
    pyro.get_param_store().clear()

    percentile_values = np.percentile(A_vars, np.arange(1, 11)*10)
    auprcs = []
    for i in range(len(percentile_values)):
        A_means_copy = copy.deepcopy(A_means)
        A_means_copy[A_vars > percentile_values[i]] = np.nan
        to_eval = pd.DataFrame(
            data=A_means_copy.cpu().numpy(),
            columns=V_vars[0],
            index=V_obs[0]
        )
        auprcs.append(get_calibration_auprc(to_eval, gold_standard, filter_method='overlap'))
    plot_fig(auprcs, os.path.join(args.output_dir, "calibration_plot.pdf"))
    np.savetxt(os.path.join(args.output_dir, "calibration_values.txt"), auprcs)



parser = argparse.ArgumentParser()
parser.add_argument("--exp-dir")
parser.add_argument("--gold-standard-path")
parser.add_argument("--num-sampling-iters", type=int, default=500)
parser.add_argument("--output-dir")

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    evaluate_calibration(args)