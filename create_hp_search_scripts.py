import os
import numpy as np

template = """
!/bin/bash
#SBATCH --job-name=yeast
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
"""

template += " ".join([
    "python -m training.train --expression-path data/yeast/{0}.h5ad",
    "--gene-tf-prior-path data/yeast/hp_search_prior_and_gs/training_prior_1.h5ad --gd-type full_dataset",
    "--batch-size 0 --num-epochs 4000",
    "--save-period 4000 --save-dir {1} --module-name Gaussian3M --lr 0.1 --min-lr 0.1 --clip-norm 0.0001",
    "--min-prior-hparam 0.0005 --max-prior-hparam 0.9995 --use-gpu --prior-std-logit-A {2} --prior-std-log-U {3}",
    "--val-after 100 --first-100-val-after 10 --val-gold-standard-path data/yeast/hp_search_prior_and_gs/val_gs_1.tsv",
    "--num-auprc-logit-sampling-iters 100",
    "--val-kl-anneal-factor 1 --num-annealing-iters 999999 --min-annealing-factor {4} --guide-max-std-log-U 2"
])

possible_A_stds = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 4]
possible_U_stds = [2, 5, 10]
possible_annealing_factors = [1, 10]
num_experiments = 10

A_stds = np.random.choice(possible_A_stds, num_experiments)
U_stds = np.random.choice(possible_U_stds, num_experiments)
annealing_factors = np.random.choice(possible_annealing_factors, num_experiments)

expression_names = ["GSE125162", "GSE144820"]

for expression_name in expression_names:
    for i in range(num_experiments):
        exp_dir = "experiments/yeast-hpsearch/{}/A-{}-U-{}-klweight-{}".format(
            expression_name, A_stds[i], U_stds[i], annealing_factors[i]
        )
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        to_write = template.format(expression_name, exp_dir, A_stds[i], U_stds[i], annealing_factors[i])
        fname_to_format = "train_yeast_expr_{}_Astd_{}_Ustd_{}_klweight_{}.sbatch"
        with open(fname_to_format.format(expression_name, A_stds[i], U_stds[i], annealing_factors[i]), "w") as f:
            f.write(to_write)