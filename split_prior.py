import argparse, os
import scanpy as sc
import numpy as np

def generate_train_val_test_prior_gs_splits(original_prior_path, val_frac, test_frac, num_splits, output_dir):
    original_prior = sc.read_h5ad(original_prior_path)
    for i in range(1, num_splits+1):
        prior_from_gs = original_prior.copy()
        permuted_row_nums = np.random.permutation(original_prior.shape[0])
        num_val_rows = int(val_frac * original_prior.shape[0])
        num_test_rows = int(test_frac * original_prior.shape[0])
        val_rows = permuted_row_nums[:num_val_rows]
        test_rows = permuted_row_nums[num_val_rows:num_val_rows+num_test_rows]

        prior_from_gs.X[val_rows, :] = 0
        prior_from_gs.X[test_rows, :] = 0
        prior_output_path = os.path.join(output_dir, "training_prior_{}.h5ad")
        prior_from_gs.write(prior_output_path.format(i))

        val_gs = original_prior[val_rows, :]
        val_gs_output_path = os.path.join(output_dir, "validation_gs_{}.tsv")
        val_gs.to_df().to_csv(val_gs_output_path.format(i), sep="\t")

        test_gs = original_prior[test_rows, :]
        if len(test_gs) > 0:
            test_gs_output_path = os.path.join(output_dir, "test_gs_{}.tsv")
            test_gs.to_df().to_csv(test_gs_output_path.format(i), sep="\t")

parser = argparse.ArgumentParser()
parser.add_argument("--original-prior-path")
parser.add_argument("--output-dir")
parser.add_argument("--val-frac", type=float, default=0.2)
parser.add_argument("--test-frac", type=float, default=0)
parser.add_argument("--num-splits", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    generate_train_val_test_prior_gs_splits(args.original_prior_path, args.val_frac, args.test_frac, args.num_splits,
                                            args.output_dir)
