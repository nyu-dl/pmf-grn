import pyro
import torch

from torch.distributions import constraints
from pyro.nn import PyroModule
import pyro.distributions as dist
from truncated_normal import TruncatedNormal


class PMF(PyroModule):
    def __init__(
            self,
            num_u: int,
            dim_u: int,
            V_prior_hparams: torch.tensor,
            dataset_size: int,
            prior_mean_log_U: float = 0.0,
            prior_std_log_U: float = 1.0,
            U_max: float = None,
            truncate_U: bool = False,
            guide_max_mean_log_U: float = 100,
            guide_max_std_log_U: float = 10,
            use_gpu: bool = False,
            use_mask: bool = False,
            eps: float = 1e-6
    ):
        super().__init__()

        self.num_u = num_u; self.dim_u = dim_u
        self.V_prior_hparams = V_prior_hparams
        self.dataset_size = dataset_size
        self.prior_mean_log_U = prior_mean_log_U
        self.prior_std_log_U = prior_std_log_U
        self.U_max = U_max
        self.truncate_U = truncate_U
        self.guide_max_mean_log_U = guide_max_mean_log_U
        self.guide_max_std_log_U = guide_max_std_log_U

        self.use_gpu = use_gpu
        if self.use_gpu is True:
            self.cuda()
            self.V_prior_hparams = self.V_prior_hparams.cuda()
        self.use_mask = use_mask

        self.eps = eps

    def sample_locals_prior(self, data, prior_hparams_U_i):
        if self.prior_mean_log_U is not None:
            prior_hparams_U_i = torch.ones_like(prior_hparams_U_i) * self.prior_mean_log_U

        if self.truncate_U is True:
            U = pyro.sample(
                "U",
                dist.TransformedDistribution(
                    TruncatedNormal(prior_hparams_U_i, self.prior_std_log_U, -51, 11).to_event(1),
                    dist.transforms.ExpTransform()
                )
            )
        else:
            U = pyro.sample(
                "U",
                dist.LogNormal(prior_hparams_U_i, self.prior_std_log_U).to_event(1)
            )

        seq_depth = pyro.sample(
            "seq_depth",
            dist.TransformedDistribution(
                dist.Normal(torch.zeros(data.shape[0], 1, device=data.device), 3).to_event(1),
                dist.transforms.SigmoidTransform()
            )
        )
        return U, seq_depth

    def initialise_locals_guide(self, data):
        if self.guide_max_mean_log_U is None:
            mean_log_U_constraint = constraints.real
        else:
            mean_log_U_constraint = constraints.less_than(self.guide_max_mean_log_U)
        self.posterior_means_log_U = pyro.param(
            "U_means",
            torch.distributions.Normal(torch.zeros(self.num_u, self.dim_u, device=data.device), 0.05).sample(),
            constraint=mean_log_U_constraint,
            event_dim=-1
        )
        self.posterior_stds_log_U = pyro.param(
            "U_stds",
            torch.ones(self.num_u, self.dim_u, device=data.device)*0.05,
            constraint=constraints.interval(0.001, self.guide_max_std_log_U),
            event_dim=-1
        )
        self.posterior_seq_depth_logit_means = pyro.param(
            "seq_depth_means",
            torch.distributions.Normal(torch.zeros(self.num_u, 1, device=data.device), 0.1).sample(),
            constraint=constraints.interval(-5, 5),
            event_dim=-1
        )
        self.posterior_seq_depth_logit_stds = pyro.param(
            "seq_depth_stds",
            torch.ones(self.num_u, 1, device=data.device)*0.05,
            constraint=constraints.interval(0.001, 5),
            event_dim=-1
        )

    def sample_locals_guide(self, i):
        stds_log_U = self.posterior_stds_log_U[i]
        means_log_U = self.posterior_means_log_U[i]
        if self.truncate_U is True:
            pyro.sample(
                "U",
                dist.TransformedDistribution(
                    TruncatedNormal(means_log_U, stds_log_U, -50, 10).to_event(1),
                    dist.transforms.ExpTransform()
                )
            )
        else:
            pyro.sample("U", dist.LogNormal(means_log_U, stds_log_U).to_event(1))

        means_logit_seq_depth = self.posterior_seq_depth_logit_means[i]
        stds_logit_seq_depth = self.posterior_seq_depth_logit_stds[i]
        pyro.sample(
            "seq_depth",
            dist.TransformedDistribution(
                dist.Normal(means_logit_seq_depth, stds_logit_seq_depth).to_event(1),
                dist.transforms.SigmoidTransform()
            )
        )

    def model(
            self,
            i: torch.LongTensor,
            prior_hparams_U_i: torch.Tensor,
            data: torch.Tensor,
            annealing_factor: float,
            mask: torch.Tensor
    ):

        with pyro.plate("globals", len(self.V_prior_hparams)):
            with pyro.poutine.scale(None, annealing_factor):
                V = pyro.sample(
                    "V",
                    dist.Normal(torch.logit(self.V_prior_hparams.clip(min=0.1, max=0.9)), 3).to_event(1)
                )

        with pyro.plate("locals", self.dataset_size, subsample=data):
            with pyro.poutine.scale(None, annealing_factor):
                U, seq_depth = self.sample_locals_prior(data, prior_hparams_U_i)

            obs_means = torch.relu(torch.matmul(U, torch.transpose(torch.sigmoid(V), 0, 1)) * seq_depth) + self.eps
            pyro.sample("obs", dist.Poisson(obs_means).to_event(1), obs=data)

    def guide(
            self,
            i: torch.LongTensor,
            prior_hparams_U_i: torch.Tensor,
            data: torch.Tensor,
            annealing_factor: float,
            mask: torch.Tensor
    ):
        self.initialise_locals_guide(data)
        self.posterior_means_logit_V = pyro.param(
            "V_means",
            torch.logit(self.V_prior_hparams.clip(min=0.1, max=0.9)),
            event_dim=-1
        )
        self.posterior_stds_logit_V = pyro.param(
            "V_stds",
            torch.ones_like(self.V_prior_hparams)*0.05,
            constraint=constraints.greater_than(0.001),
            event_dim=-1
        )



        with pyro.poutine.scale(None, annealing_factor):
            with pyro.plate("globals", len(self.V_prior_hparams)):
                pyro.sample("V", dist.Normal(self.posterior_means_logit_V, self.posterior_stds_logit_V).to_event(1))
            with pyro.plate("locals", self.dataset_size, subsample=i):
                self.sample_locals_guide(i)


class SimpleModule(PyroModule):
    def __init__(
            self,
            num_u: int,
            dim_u: int,
            V_prior_hparams: torch.tensor,
            dataset_size: int,
            use_gpu: bool,
            use_mask: bool,
            eps: float = 1e-6
    ):
        super().__init__()

        self.num_u = num_u; self.dim_u = dim_u
        self.V_prior_hparams = V_prior_hparams
        self.dataset_size = dataset_size

        self.use_gpu = use_gpu
        if self.use_gpu is True:
            self.cuda()
            self.V_prior_hparams = self.V_prior_hparams.cuda()

        self.eps = eps

    def model(
            self,
            i: torch.LongTensor,
            prior_hparams_U_i: torch.Tensor,
            data: torch.Tensor,
            annealing_factor: float
    ):

        with pyro.plate("globals", len(self.V_prior_hparams)):
            with pyro.poutine.scale(None, annealing_factor):
                V = pyro.sample("V", dist.Normal(self.V_prior_hparams, 10).to_event(1))

        with pyro.plate("locals", self.dataset_size, subsample=data):
            with pyro.poutine.scale(None, annealing_factor):
                U = pyro.sample(
                    "U",
                    dist.Normal(torch.zeros_like(prior_hparams_U_i), 10).to_event(1)
                )

            obs_means = torch.matmul(U, torch.transpose(V, 0, 1))
            pyro.sample("obs", dist.Normal(obs_means, 0.1).to_event(1), obs=data)

    def guide(
            self,
            i: torch.LongTensor,
            prior_hparams_U_i: torch.Tensor,
            data: torch.Tensor,
            annealing_factor: float
    ):
        self.posterior_means_U = pyro.param(
            "U_means",
            torch.distributions.Normal(torch.zeros(self.num_u, self.dim_u, device=data.device), 0.05).sample(),
            event_dim=-1
        )
        self.posterior_stds_U = pyro.param(
            "U_stds",
            torch.ones(self.num_u, self.dim_u, device=data.device)/20,
            constraint=constraints.greater_than(0.001),
            event_dim=-1
        )
        self.posterior_means_V = pyro.param(
            "V_means",
            self.V_prior_hparams,
            event_dim=-1
        )
        self.posterior_stds_V = pyro.param(
            "V_stds",
            torch.ones_like(self.V_prior_hparams)/20,
            constraint=constraints.greater_than(0.001),
            event_dim=-1
        )

        with pyro.poutine.scale(None, annealing_factor):
            with pyro.plate("globals", len(self.V_prior_hparams)):
                pyro.sample("V", dist.Normal(self.posterior_means_V, self.posterior_stds_V).to_event(1))
            with pyro.plate("locals", self.dataset_size, subsample=i):
                stds_U = self.posterior_stds_U[i]
                means_U = self.posterior_means_U[i]
                pyro.sample("U", dist.Normal(means_U, stds_U).to_event(1))


class Bernoulli3M(PMF):
    def __init__(
            self,
            num_u: int,
            dim_u: int,
            V_prior_hparams: torch.tensor,
            dataset_size: int,
            use_gpu: bool,
            use_mask: bool,
            eps: float = 1e-6,
            prior_positive_prob: float = 0.7
    ):
        super().__init__(num_u, dim_u, V_prior_hparams, dataset_size, use_gpu, use_mask, eps)
        self.prior_positive_prob = prior_positive_prob

    def model(self,
              i: torch.LongTensor,
              prior_hparams_U_i: torch.Tensor,
              data: torch.Tensor,
              annealing_factor: float
              ):
        with pyro.plate("globals", len(self.V_prior_hparams)):
            with pyro.poutine.scale(None, annealing_factor):
                A = pyro.sample(
                    "A",
                    dist.ContinuousBernoulli(self.V_prior_hparams.clip(min=0.01, max=0.99)).to_event(1)
                )
                B = pyro.sample(
                    "B",
                    dist.ContinuousBernoulli(
                        torch.ones_like(self.V_prior_hparams) * self.prior_positive_prob
                    ).to_event(1)
                )
                V = A * (2*B - 1)

        with pyro.plate("locals", self.dataset_size, subsample=data):
            with pyro.poutine.scale(None, annealing_factor):
                U, seq_depth = self.sample_locals_prior(data, prior_hparams_U_i)

            obs_means = torch.relu(torch.matmul(U, torch.transpose(torch.sigmoid(V), 0, 1)) * seq_depth) + self.eps
            pyro.sample("obs", dist.Poisson(obs_means).to_event(1), obs=data)

    def guide(
            self,
            i: torch.LongTensor,
            prior_hparams_U_i: torch.Tensor,
            data: torch.Tensor,
            annealing_factor: float
    ):
        self.initialise_locals_guide(data)
        self.posterior_temp_A = pyro.param(
            "A_temp",
            self.V_prior_hparams.clip(min=0.01, max=0.99),
            constraint=constraints.interval(0.01, 0.99),
            event_dim=-1
        )
        self.posterior_temp_B = pyro.param(
            "B_temp",
            torch.ones_like(self.V_prior_hparams)*self.prior_positive_prob,
            constraint=constraints.interval(0.01, 0.99),
            event_dim=-1
        )

        with pyro.poutine.scale(None, annealing_factor):
            with pyro.plate("globals", len(self.V_prior_hparams)):
                pyro.sample("A", dist.ContinuousBernoulli(self.posterior_temp_A).to_event(1))
                pyro.sample("B", dist.ContinuousBernoulli(self.posterior_temp_B).to_event(1))
            with pyro.plate("locals", self.dataset_size, subsample=i):
                self.sample_locals_guide(i)


class Gaussian3M(PMF):
    def __init__(
            self,
            num_u: int,
            dim_u: int,
            V_prior_hparams: torch.tensor,
            dataset_size: int,
            prior_mean_log_U: float = 0.0,
            prior_std_log_U: float = 1.0,
            U_max: float = None,
            truncate_U: bool = False,
            guide_max_mean_log_U: float = 100,
            guide_max_std_log_U: float = 10,
            prior_std_logit_A: float = 1.0,
            prior_std_B: float = 1.0,
            use_gpu: bool = False,
            use_mask: bool = False,
            eps: float = 1e-6,
            min_prior_hparam: float = 0.01,
            max_prior_hparam: float = 0.99
    ):
        super().__init__(
            num_u,
            dim_u,
            V_prior_hparams,
            dataset_size,
            prior_mean_log_U,
            prior_std_log_U,
            U_max,
            truncate_U,
            guide_max_mean_log_U,
            guide_max_std_log_U,
            use_gpu,
            use_mask,
            eps
        )
        self.min_prior_hparam = min_prior_hparam
        self.max_prior_hparam = max_prior_hparam
        self.prior_std_logit_A = prior_std_logit_A
        self.prior_std_B = prior_std_B

    def model(self,
              i: torch.LongTensor,
              prior_hparams_U_i: torch.Tensor,
              data: torch.Tensor,
              annealing_factor: float,
              mask: torch.Tensor
              ):
        with pyro.poutine.scale(None, annealing_factor):
            with pyro.plate("gene_globals", len(self.V_prior_hparams)):
                A = pyro.sample(
                    "A",
                    dist.TransformedDistribution(
                        dist.Normal(
                            torch.logit(
                                self.V_prior_hparams.clip(min=self.min_prior_hparam, max=self.max_prior_hparam)
                            ),
                            self.prior_std_logit_A
                        ).to_event(1),
                        dist.transforms.SigmoidTransform()
                    )
                )
                B = pyro.sample("B", dist.Normal(torch.zeros_like(self.V_prior_hparams), self.prior_std_B).to_event(1))
                V = A * B
            obs_std = pyro.sample("obs_std", dist.LogNormal(torch.tensor([0.], device=data.device), 1))

        with pyro.plate("locals", self.dataset_size, subsample=data):
            with pyro.poutine.scale(None, annealing_factor):
                U, seq_depth = self.sample_locals_prior(data, prior_hparams_U_i)
                if self.U_max is not None: U = torch.clip(U, max=self.U_max)

            obs_means = torch.matmul(U, torch.transpose(V, 0, 1)) * seq_depth
            if self.use_mask is True:
                obs_means = obs_means * mask
                obs = data * mask
            else:
                obs = data
            pyro.sample("obs", dist.Normal(obs_means, obs_std+0.001).to_event(1), obs=obs)

    def guide(
            self,
            i: torch.LongTensor,
            prior_hparams_U_i: torch.Tensor,
            data: torch.Tensor,
            annealing_factor: float,
            mask: torch.Tensor
    ):
        self.initialise_locals_guide(data)
        self.posterior_means_logit_A = pyro.param(
            "A_means",
            torch.logit(self.V_prior_hparams.clip(min=self.min_prior_hparam, max=self.max_prior_hparam)),
            constraint=constraints.interval(-10, 10),
            event_dim=-1
        )
        self.posterior_stds_logit_A = pyro.param(
            "A_stds",
            torch.ones_like(self.V_prior_hparams)*0.1,
            constraint=constraints.greater_than(0.001),
            event_dim=-1
        )

        self.posterior_means_B = pyro.param(
            "B_means",
            torch.zeros_like(self.V_prior_hparams),
            event_dim=-1
        )
        self.posterior_stds_B = pyro.param(
            "B_stds",
            torch.ones_like(self.V_prior_hparams)*0.1,
            constraint=constraints.greater_than(0.001),
            event_dim=-1
        )

        self.posterior_mean_obs_std = pyro.param(
            "obs_std_mean",
            torch.tensor([0.], device=data.device)
        )
        self.posterior_std_obs_std = pyro.param(
            "obs_std_std",
            torch.tensor([1.], device=data.device),
            constraint=constraints.greater_than(0.001),
        )

        with pyro.poutine.scale(None, annealing_factor):
            with pyro.plate("gene_globals", len(self.V_prior_hparams)):
                pyro.sample(
                    "A",
                    dist.TransformedDistribution(
                        dist.Normal(self.posterior_means_logit_A, self.posterior_stds_logit_A).to_event(1),
                        dist.transforms.SigmoidTransform()
                    )
                )
                pyro.sample("B", dist.Normal(self.posterior_means_B, self.posterior_stds_B).to_event(1))
            pyro.sample("obs_std", dist.LogNormal(self.posterior_mean_obs_std, self.posterior_std_obs_std))
            with pyro.plate("locals", self.dataset_size, subsample=i):
                self.sample_locals_guide(i)

PMF_MODULES = {
    "PMF": PMF,
    "SimpleModule": SimpleModule,
    "Bernoulli3M": Bernoulli3M,
    "Gaussian3M": Gaussian3M
}
