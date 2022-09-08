import argparse
import os
import torch
import pyro
import logging
import scanpy as sc

from sys import argv
from tensorboardX import SummaryWriter
from pyro.infer import SVI, Trace_ELBO
from pyro import poutine
from torch.utils.data.dataloader import DataLoader
from pyro.infer.util import torch_item

from data_processing.dataset import get_sc_dataset, to_cuda
from modules.pmf import PMF_MODULES
from evaluation_metrics.evaluation_V import get_CM_A

def get_parser():
    parser = argparse.ArgumentParser()

    opt_group = parser.add_argument_group("opt-group")
    opt_group.add_argument('--lr', type=float, default=0.001)
    opt_group.add_argument('--beta1', type=float, default=0.9)
    opt_group.add_argument('--beta2', type=float, default=0.99)
    opt_group.add_argument("--lr-decay-rate", type=float, default=1)
    opt_group.add_argument("--lr-decay-period", type=int, default=1)
    opt_group.add_argument("--min-lr", type=float, default=0)
    opt_group.add_argument("--clip-norm", type=float, default=-1)
    opt_group.add_argument("--clip-value", type=float, default=-1)
    opt_group.add_argument("--initial-annealing-factor", type=float, default=1)
    opt_group.add_argument("--num-annealing-iters", type=float, default=1)

    data_group = parser.add_argument_group("data-group")
    data_group.add_argument("--expression-path")
    data_group.add_argument("--gene-tf-prior-path")
    data_group.add_argument("--tf-activity-prior-path", default="")
    data_group.add_argument("--tfa-from-expression", action="store_true",
                            help="initialise tfa matrix using expression data")

    iteration_group = parser.add_argument_group("iteration-group")
    iteration_group.add_argument("--gd-type", choices=["sgd", "full_dataset"], default="sgd")
    iteration_group.add_argument("--batch-size", type=int)
    iteration_group.add_argument("--num-epochs", type=int)

    module_group = parser.add_argument_group("module-group")
    module_group.add_argument("--module-name", choices=PMF_MODULES.keys())
    module_group.add_argument("--prior-mean-log-U", type=float, default=None)
    module_group.add_argument("--prior-std-log-U", type=float, default=1.0)
    module_group.add_argument("--U-max", type=float, default=None)
    module_group.add_argument("--truncate-U", action="store_true")
    module_group.add_argument("--guide-max-mean-log-U", type=float, default=None)
    module_group.add_argument("--guide-max-std-log-U", type=float, default=10)

    # Bernoulli3M
    module_group.add_argument("--prior-positive-prob", type=float, default=0.7)
    # Gaussian3M
    module_group.add_argument("--min-prior-hparam", type=float, default=0.01)
    module_group.add_argument("--max-prior-hparam", type=float, default=0.99)
    module_group.add_argument("--prior-std-logit-A", type=float, default=1.0)
    module_group.add_argument("--prior-std-B", type=float, default=1.0)

    validation_group = parser.add_argument_group("--validation-group")
    validation_group.add_argument("--num-posterior-samples", type=int, default=1)
    validation_group.add_argument("--mask-path", default=None)
    validation_group.add_argument("--val-after", type=int, default=0,
                                  help="number of epochs between validations ( if > 0)")
    validation_group.add_argument("--first-100-val-after", type=int, default=0)
    validation_group.add_argument("--val-kl-anneal-factor", type=float, default=1e-10)
    validation_group.add_argument("--val-gold-standard-path", default=None)
    validation_group.add_argument("--test-gold-standard-path", default=None)
    validation_group.add_argument("--num-auprc-logit-sampling-iters", type=int, default=100)
    validation_group.add_argument("--val-loss-type", choices=["mask", "neg_mll"], default="mask")
    validation_group.add_argument("--num-mll-samples", type=int, default=100)
    validation_group.add_argument("--use-self-norm-estimator", action="store_true")

    progress_group = parser.add_argument_group("--progress-group")
    progress_group.add_argument("--save-dir")
    progress_group.add_argument("--save-period", type=int, default=1)
    progress_group.add_argument("--params-load-path", default=None)

    debugger_group = parser.add_argument_group("--debugger-group")
    debugger_group.add_argument("--debug-num-entries", type=int, default=-1)

    gpu_group = parser.add_argument_group("--gpu-group")
    gpu_group.add_argument("--use-gpu", action="store_true")

    return parser


def train(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, 'log.out'), mode='a'),
            logging.StreamHandler()
        ]
    )
    logging.info(' '.join(argv))

    clip_args = {}
    if args.clip_norm > 0:
        clip_args["clip_norm"] = args.clip_norm
    if args.clip_value > 0:
        clip_args["clip_value"] = args.clip_value
    opt = pyro.optim.ExponentialLR(
        {
            "optimizer": torch.optim.Adam,
            "optim_args": {"lr": args.lr, "betas": (args.beta1, args.beta2)},
            "gamma": args.lr_decay_rate
        },
        clip_args=clip_args
    )

    move_each_batch_to_gpu = True if args.use_gpu and args.gd_type == "sgd" else False

    dataset, coll_fn = get_sc_dataset(
        args.expression_path,
        args.gene_tf_prior_path,
        args.tf_activity_prior_path,
        debug_num_entries=args.debug_num_entries,
        mask_path=args.mask_path,
        tfa_from_expression=args.tfa_from_expression,
        move_each_batch_to_gpu=move_each_batch_to_gpu
    )

    if args.use_gpu and args.gd_type == "full_dataset":
        dataset.i, dataset.U_prior_hparams_tensor, dataset.V_prior_hparams_tensor, dataset.W_tensor = to_cuda(
            dataset.i, dataset.U_prior_hparams_tensor, dataset.V_prior_hparams_tensor, dataset.W_tensor
        )
        if hasattr(dataset, "mask"):
            dataset.mask = to_cuda(dataset.mask)

    batch_size = args.batch_size if args.gd_type == "sgd" else len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=coll_fn)

    # TODO: implement dataset shuffling
    module_kwargs = {
        "num_u": dataset.num_u,
        "dim_u": dataset.dim_u,
        "V_prior_hparams": dataset.V_prior_hparams_tensor,
        "prior_mean_log_U": args.prior_mean_log_U,
        "prior_std_log_U": args.prior_std_log_U,
        "U_max": args.U_max,
        "truncate_U": args.truncate_U,
        "guide_max_mean_log_U": args.guide_max_mean_log_U,
        "guide_max_std_log_U": args.guide_max_std_log_U,
        "dataset_size": len(dataset),
        "use_mask": args.mask_path is not None,
        "use_gpu": args.use_gpu
    }
    if args.module_name == "Bernoulli3M":
        module_kwargs["prior_positive_prob"] = args.prior_positive_prob
    elif args.module_name == "Gaussian3M":
        module_kwargs["min_prior_hparam"] = args.min_prior_hparam
        module_kwargs["max_prior_hparam"] = args.max_prior_hparam
        module_kwargs["prior_std_logit_A"] = args.prior_std_logit_A
        module_kwargs["prior_std_B"] = args.prior_std_B

    module = PMF_MODULES.get(args.module_name)(**module_kwargs)
    scale = 1/(len(dataset) * dataset.dim_v)

    svi = SVI(
        poutine.scale(module.model, scale=scale),
        poutine.scale(module.guide, scale=scale),
        opt,
        loss=Trace_ELBO()
    )

    if args.params_load_path is not None:
        pyro.get_param_store().load(args.params_load_path, map_location='cpu')

    # Save row and column names for U and V
    dataset.U_prior_hparams.obs_names.to_series().to_csv(
        os.path.join(args.save_dir, "U_obs_names.csv"),
        index=False,
        header=False
    )
    dataset.U_prior_hparams.var_names.to_series().to_csv(
        os.path.join(args.save_dir, "U_var_names.csv"),
        index=False,
        header=False
    )
    dataset.V_prior_hparams.obs_names.to_series().to_csv(
        os.path.join(args.save_dir, "V_obs_names.csv"),
        index=False,
        header=False
    )
    dataset.V_prior_hparams.var_names.to_series().to_csv(
        os.path.join(args.save_dir, "V_var_names.csv"),
        index=False,
        header=False
    )
    writer = SummaryWriter(logdir=args.save_dir)

    def get_annealing_factor(iter_num, initial_annealing_factor, num_annealing_iters):
        if iter_num < num_annealing_iters:
            return initial_annealing_factor + (1 - initial_annealing_factor) * iter_num/num_annealing_iters
        else:
            return 1

    def train_iteration(iter_num, i, prior_hparams_U_i, W_i, annealing_factor, mask):
        if iter_num % args.save_period == 0:
            pyro.get_param_store().save(
                os.path.join(args.save_dir, "iter_{}.params".format(iter_num))
            )
        loss = svi.step(i, prior_hparams_U_i, W_i, annealing_factor, mask)
        writer.add_scalar('loss', loss, iter_num)
        logging.info("Iter: {} Loss: {}".format(iter_num, loss))

    def validation_tb_and_save(validation_loss, best_loss, val_iter_num):
        logging.info("Validation Loss: {}".format(validation_loss))
        writer.add_scalar('validation_loss', validation_loss, val_iter_num)
        if validation_loss < best_loss:
            best_loss = validation_loss
            logging.info("Best validation loss so far")
            pyro.get_param_store().save(os.path.join(args.save_dir, "best_iter.params"))
        return best_loss

    def auprc_validation_iteration(dataset, gold_standard, num_sampling_iters, best_auprc, val_iter_num, gs_type="val"):
        metrics = get_CM_A(
            dataset.V_prior_hparams.obs_names.to_frame(),
            dataset.V_prior_hparams.var_names.to_frame(),
            gold_standard,
            num_sampling_iters,
            positive_class=None,
            filter_method='keep_all_gold_standard'
        )
        logging.info("{} AUPRC: {}".format(gs_type, metrics.aupr))
        writer.add_scalar('{}_AUPRC'.format(gs_type), metrics.aupr, val_iter_num)
        if metrics.aupr > best_auprc:
            best_auprc = metrics.aupr
            logging.info("Best {} AUPRC so far".format(gs_type))
            pyro.get_param_store().save(os.path.join(args.save_dir, "best_{}_auprc_iter.params".format(gs_type)))
        return best_auprc

    def get_marginal_log_likelihood(num_samples=1000, use_self_norm_estimator=False):
        elbos = []
        self_norm_weights = []
        with torch.no_grad():
            for k in range(num_samples):
                minibatch_elbos = []
                minibatch_self_norm_weights = []
                for iter_num, (i, prior_hparams_U_i, W_i, mask) in enumerate(dataloader):
                    model_trace, guide_trace = Trace_ELBO()._get_trace(
                        module.model, module.guide, (i, prior_hparams_U_i, W_i, 1.0, None), {}
                    )

                    minibatch_log_obs = 0
                    minibatch_log_prior = 0
                    minibatch_log_posterior = 0
                    for name, site in model_trace.nodes.items():
                        if site["type"] == "sample":
                            if name == "obs":
                                minibatch_log_obs += torch_item(site["log_prob_sum"])
                            else:
                                minibatch_log_prior += torch_item(site["log_prob_sum"])
                    for name, site in guide_trace.nodes.items():
                        if site["type"] == "sample":
                            minibatch_log_posterior += torch_item(site["log_prob_sum"])

                    minibatch_elbos.append(minibatch_log_obs + minibatch_log_prior - minibatch_log_posterior)
                    minibatch_self_norm_weights.append(minibatch_log_prior - minibatch_log_posterior)
                elbos.append(torch.tensor(minibatch_elbos).mean())
                self_norm_weights.append(torch.tensor(minibatch_self_norm_weights).mean())

        if use_self_norm_estimator is True:
            return (torch.logsumexp(torch.tensor(elbos), 0) -
                    torch.logsumexp(torch.tensor(self_norm_weights), 0)
                    ).detach().cpu().item()
        else:
            return (torch.logsumexp(torch.tensor(elbos), 0) -
                    torch.log(torch.tensor([num_samples]))
                    ).detach().cpu().item()

    def validation_iteration(val_iter_num, best_loss, best_val_auprc, best_test_auprc, pre_first_epoch=False,
                             val_loss_type="mask", use_self_norm_estimator=False):
        if args.mask_path is not None and val_loss_type == "mask":
            val_loss = 0
            for iter_num, (i, prior_hparams_U_i, W_i, mask) in enumerate(dataloader, 1):
                val_mask = mask if mask is None else ~mask
                with torch.no_grad():
                    val_loss += svi.step(i, prior_hparams_U_i, W_i, args.val_kl_anneal_factor, val_mask)
            val_loss = val_loss/iter_num
            best_loss = validation_tb_and_save(val_loss, best_loss, val_iter_num)
        elif val_loss_type == "neg_mll":
            val_loss = -get_marginal_log_likelihood(args.num_mll_samples, use_self_norm_estimator)
            best_loss = validation_tb_and_save(val_loss, best_loss, val_iter_num)
        if args.val_gold_standard_path is not None or args.test_gold_standard_path is not None:
            if pre_first_epoch is True:
                # initialise params in guide for auprc calculation before first training iteration
                for i, prior_hparams_U_i, W_i, mask in dataloader:
                    val_mask = mask if mask is None else ~mask
                    with torch.no_grad():
                        _ = module.guide(i, prior_hparams_U_i, W_i, 1, val_mask)
                        del _
            if args.val_gold_standard_path is not None:
                best_val_auprc = auprc_validation_iteration(dataset, val_gold_standard,
                                                            args.num_auprc_logit_sampling_iters,
                                                            best_val_auprc, val_iter_num, gs_type="val")
            if args.test_gold_standard_path is not None:
                best_test_auprc = auprc_validation_iteration(dataset, test_gold_standard,
                                                             args.num_auprc_logit_sampling_iters,
                                                             best_test_auprc, val_iter_num, gs_type="test")

        val_iter_num += 1
        return val_iter_num, best_loss, best_val_auprc, best_test_auprc

    best_loss = 999999999999999999999
    best_val_auprc = 0
    best_test_auprc = 0
    val_iter_num = 0
    if args.val_gold_standard_path is not None:
        val_gold_standard = sc.read_csv(args.val_gold_standard_path, delimiter="\t", first_column_names=True)
    if args.test_gold_standard_path is not None:
        test_gold_standard = sc.read_csv(args.test_gold_standard_path, delimiter="\t", first_column_names=True)

    if args.val_after > 0:
        val_iter_num, best_loss, best_val_auprc, best_test_auprc = validation_iteration(val_iter_num, best_loss,
            best_val_auprc, best_test_auprc, pre_first_epoch=True, val_loss_type=args.val_loss_type,
            use_self_norm_estimator=args.use_self_norm_estimator)
    for epoch in range(args.num_epochs):
        for iter_num, (i, prior_hparams_U_i, W_i, mask) in enumerate(dataloader):
            annealing_factor = get_annealing_factor(epoch, args.initial_annealing_factor, args.num_annealing_iters)
            train_iteration(epoch*len(dataloader)+iter_num, i, prior_hparams_U_i, W_i, annealing_factor, mask)

        if (args.val_after > 0 and (epoch % args.val_after == 0 or epoch == args.num_epochs)) or \
                (args.first_100_val_after > 0 and epoch <= 100 and epoch % args.first_100_val_after == 0):
            val_iter_num, best_loss, best_val_auprc, best_test_auprc = validation_iteration(
                epoch*len(dataloader), best_loss, best_val_auprc, best_test_auprc, pre_first_epoch=True,
                val_loss_type=args.val_loss_type, use_self_norm_estimator=args.use_self_norm_estimator)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)
