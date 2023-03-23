![PMFGRN](PMF-GRN_logo.png)

# Probabilistic Matrix Factorization for Gene Regulatory Network Inference

This repository and its references contain the models, data and scripts used to carry out the experiments in the
Probabilistic Matrix Factorization for Gene Regulatory Network Inference paper.

## Installation Guide

We used a Linux OS with an Nvidia GPU with 16 GB of memory.

A conda environment file (environment.yml) is provided as part of this repository. It may contain packages beyond those
needed to run the scripts here.

### GPU
CUDA (We used version 11.1)

### Data
All data used by PMF-GRN in this paper can be found
[here](https://drive.google.com/drive/folders/1LNt2Fy4p8NBNzq5SUeKP7jKay8JhQeGL?usp=sharing).
We recommend downloading this entire folder to the repository root for ease of use with the scripts provided
in this README.

## Datasets and Inference Scripts
### S. cerevisiae
Single-cell gene expression datasets obtained from NCBI GEO (GSE125162 & GSE144820).
Prior-known TF-target gene interaction matrix obtained from the YEASTRACT database (Monteiro et al. 2020, Teixeira et al. 2018).
Gold Standard TF-target gene interactions matrix obtained from Tchourine et al. 2018.

To carry out inference using the hyperparameter configurations used in the paper, use the following scripts for
GSE125162 and GSE144820 respectively:
#### GSE125162
`python -m training.train --expression-path data/yeast/GSE125162.h5ad
--gene-tf-prior-path data/yeast/YEASTRACT_20190713_BOTH.h5ad --gd-type full_dataset --batch-size 0 --num-epochs 20000
--save-period 20000 --save-dir experiments/yeast/GSE125162 --module-name Gaussian3M --lr 0.1
--min-lr 0.1 --clip-norm 0.0001 --min-prior-hparam 0.0005 --max-prior-hparam 0.9995 --use-gpu --prior-std-logit-A 2
--prior-std-log-U 5 --val-after 100 --first-100-val-after 10 --num-auprc-logit-sampling-iters 100
--val-loss-type neg_mll --num-mll-samples 10 --val-kl-anneal-factor 1 --num-annealing-iters 999999
--initial-annealing-factor 10 --guide-max-std-log-U 2`
#### GSE144820
`python -m training.train --expression-path data/yeast/GSE144820.h5ad
--gene-tf-prior-path data/yeast/YEASTRACT_20190713_BOTH.h5ad --gd-type full_dataset --batch-size 0 --num-epochs 20000
--save-period 20000 --save-dir experiments/yeast/GSE144820 --module-name Gaussian3M --lr 0.1 --min-lr 0.1
--clip-norm 0.0001 --min-prior-hparam 0.0005 --max-prior-hparam 0.9995 --use-gpu --prior-std-logit-A 2
--prior-std-log-U 5 --val-after 100 --first-100-val-after 10 --val-loss-type neg_mll --num-mll-samples 10
--val-kl-anneal-factor 1 --num-annealing-iters 999999 --initial-annealing-factor 10 --guide-max-std-log-U 2`

To evaluate inferred GRNs and obtain an inferred consensus GRN (results will be saved to arg of --output-dir
i.e. yeast_results):
`python evaluate.py --exp-dirs experiments/yeast/GSE125162 experiments/yeast/GSE144820
--gold-standard-path data/yeast/gold_standard.tsv --output-dir experiments/yeast/results
--expression-names GSE125162 GSE144820 --num-sampling-iters 500`

### B. subtilis
Microarray gene expression datasets obtained from NCBI GEO GSE27219 (B1) & GSE67023 (B2). 
Prior-known TF-target gene and gold standard interactions matrix were obtained from the Subtiwiki database (Faria et al. 2016).

To carry out inference using the hyperparameter configurations used in the paper, use the following scripts for
B1 and B2 respectively, replacing {} with the desired cross validation split (1, 2, 3, 4 or 5):

#### GSE27219 (B1)
"python -m training.train --expression-path data/bsubtilis/bsubtilis_expression_b1.h5ad",
    "--gene-tf-prior-path data/bsubtilis/cv_training_prior_{}.h5ad --gd-type full_dataset",
    "--batch-size 0 --num-epochs 4000",
    "--save-period 4000 --save-dir experiments/bsubtilis/B1/split_{} --module-name Gaussian3M --lr 0.1 --min-lr 0.1 --clip-norm 0.0001",
    "--min-prior-hparam 0.0005 --max-prior-hparam 0.9995 --use-gpu --prior-std-logit-A {2} --prior-std-log-U {3}",
    "--val-after 100 --first-100-val-after 10 --val-gold-standard-path data/bsubtilis/cv_validation_gs_{}.tsv",
    "--num-auprc-logit-sampling-iters 100",
    "--val-kl-anneal-factor 1 --num-annealing-iters 999999 --initial-annealing-factor {4} --guide-max-std-log-U 2"

#### GSE67023 (B2)
"python -m training.train --expression-path data/bsubtilis/bsubtilis_expression_b2.h5ad",
    "--gene-tf-prior-path data/bsubtilis/cv_training_prior_{}.h5ad --gd-type full_dataset",
    "--batch-size 0 --num-epochs 4000",
    "--save-period 4000 --save-dir experiments/bsubtilis/B2/split_{} --module-name Gaussian3M --lr 0.1 --min-lr 0.1 --clip-norm 0.0001",
    "--min-prior-hparam 0.0005 --max-prior-hparam 0.9995 --use-gpu --prior-std-logit-A {2} --prior-std-log-U {3}",
    "--val-after 100 --first-100-val-after 10 --val-gold-standard-path data/bsubtilis/cv_validation_gs_{}.tsv",
    "--num-auprc-logit-sampling-iters 100",
    "--val-kl-anneal-factor 1 --num-annealing-iters 999999 --initial-annealing-factor {4} --guide-max-std-log-U 2"

To evaluate inferred GRNs and obtain an inferred consensus GRN (again, replace {} with the desired cross validation split
(1, 2, 3, 4 or 5)):
`python evaluate.py --exp-dirs experiments/bsubtilis/B1/split_{} experiments/bsubtilis/B1/split_{}
--gold-standard-path data/bsubtilis/validation_gs_{}.tsv --output-dir bsubtilis/results/B1/split_{}
--expression-names m14 m15`

An easy way of running this script for all splits (1-5 in this case) is as follows:
`for i in {1..5}; do python evaluate.py --exp-dirs experiments/bsubtilis/B1/split_$i experiments/bsubtilis/B1/split_$i
--gold-standard-path data/bsubtilis/validation_gs_$i.tsv
--output-dir experiments/bsubtilis/results/B1/split_$i --expression-names B1 B2`


## Hyperparameter search
To carry out hyperparameter search, first split prior hyperparameters into training and validation using
`split_prior.py`. Using S. Cerevisiae GSE125162 as an example:

`python split_prior.py --original-prior-path data/yeast/YEASTRACT_20190713_BOTH.h5ad
--output-dir data/yeast/hp_search_prior_and_gs --val-frac 0.2`

To create hyperparameter scripts using this split, use `create_hp_search_scripts.py`.
This script has been designed for a slurm-based system but can be modified to run without slurm.
Its contents can be modified depending on the dataset including expression and prior files,
hyperparameters over which to carry out a random search, and the possible values of these hyperparameters.
Other options can be changed as well such as the use of minibatches instead of full dataset gradient descent.
Use the script as follows:
`python create_hp_search_scripts.py`.

Running the script above will have created several files, each of which is a slurm script for an experiment
corresponding to a particular hyperparameter configuration. Run all these scripts using the following command:
`for FNAME in train_yeast_expr_*sbatch; do sbatch $FNAME; done`

Once all the hyperparameter search experiments have finished running, the results of the search can be
saved to a file as follows (this file is `yeast_hp_results.txt` in this example):
`python get_hp_search_results.py --parent-dnames experiments/yeast-hpsearch/GSE125162
experiments/yeast-hpsearch/GSE144820 --output-path experiments/yeast-hpsearch/yeast_hp_results.txt`

The best hyperparameter configuration for each expression file as given in the results file from the above script
can then be used to run a new experiment with the original, full prior as done in the **Datasets and Inference**
part of this README.

## Inferred GRNs
Posterior parameters for the matrix **A** inferred by PMF-GRN are given
[here](https://drive.google.com/drive/folders/1RIRP1VDm8s0m44FKl_FgODogpoO_BNrQ?usp=sharing)
as `best_iter.params` in each case.
For each dataset (yeast and b. subtilis), we have also provided a `results` folder
containing the output of the `evaluate.py` script, which gives each inferred GRN including one additional consensus GRN
as a tsv file following Monte Carlo sampling.
Associated AUPRC values are also given in a text file for each dataset.
Note that the inferred GRNs and AUPRC values will change slightly each time `evaluate.py` is run due to the stochastic
nature of Monte Carlo sampling.

## Calibration Evaluation
To evaluate calibration for an inferred GRN, use the `evaluate_calibration.py` script as in the following example:
`python evaluate_calibration.py --exp-dir experiments/yeast/GSE144820 --gold-standard-path data/yeast/gold_standard.tsv
--output-dir experiments/yeast/results/calibration_GSE144820`

Results will be saved to the argument of `--output-dir`.

## Inferelator scripts and inferred GRNs
Scripts used to run the Inferelator experiments in the paper and the resulting inferred GRNs are provided
[here](https://drive.google.com/drive/folders/1YKiKlJ2qHcAMtzbzWo6JTQR1U_EUyEE9?usp=sharing).
