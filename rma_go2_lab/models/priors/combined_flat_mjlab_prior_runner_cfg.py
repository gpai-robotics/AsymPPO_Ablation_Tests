"""Runner config for the combined AsymPPO flat prior stage."""

from isaaclab.utils import configclass

from rma_go2_lab.models.priors.flat_mjlab_prior_runner_cfg import Go2FlatMjlabPriorPPORunnerCfg


@configclass
class Go2CombinedFlatMjlabPriorPPORunnerCfg(Go2FlatMjlabPriorPPORunnerCfg):
    """Stage 1: train a clean flat deployable actor for the combined branch."""

    experiment_name = "go2_combined_flat_mjlab_prior_v1"
