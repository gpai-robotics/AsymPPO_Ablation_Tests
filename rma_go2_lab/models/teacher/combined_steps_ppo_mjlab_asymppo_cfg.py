"""Runner config for the combined stair-aware AsymPPO baseline."""

from isaaclab.utils import configclass

from rma_go2_lab.models.teacher.combined_rough_ppo_mjlab_asymppo_cfg import (
    Go2BlindRoughMjlabCombinedRoughRunnerCfg,
    make_combined_asymppo_policy,
)
from rma_go2_lab.models.teacher.combined_stage_checkpoints import resolve_stage_checkpoint


@configclass
class Go2BlindRoughMjlabCombinedStepsRunnerCfg(Go2BlindRoughMjlabCombinedRoughRunnerCfg):
    """Stage 3: warm-start from the combined rough checkpoint and fine-tune stairs."""

    max_iterations = 3000
    save_interval = 50
    experiment_name = "go2_blind_rough_combined_asymppo_steps_v1"

    policy = make_combined_asymppo_policy(
        actor_init_path=resolve_stage_checkpoint(
            ("COMBINED_ROUGH_CKPT", "GO2_COMBINED_ROUGH_CKPT"),
            "stairs-stage rough",
        )
    )

    
