import gymnasium as gym

from rma_go2_lab.models.adaptation.actor_critic import HistoryEncoderStudentActorCritic
from rma_go2_lab.models.adaptation.frozen_adapt_v3_phase1 import FrozenAdaptV3Phase1
from rma_go2_lab.models.adaptation.modular_actor_critic import ModularHistoryAdaptationActorCritic
from rma_go2_lab.models.adaptation.ppo_rma_v3_phase1 import RmaV3Phase1PPO
from rma_go2_lab.models.adaptation.ppo_rma_v3_phase2 import RmaV3Phase2PPO
from rma_go2_lab.models.adaptation.ppo_with_v3_expert import AdaptationPPOWithV3Expert
from rma_go2_lab.models.adaptation.ppo_with_v3_latent import AdaptationPPOWithV3Latent
from rma_go2_lab.models.adaptation.rma_v3_actor_critic import RmaV3ActorCritic
from rma_go2_lab.models.blind.actor_critic import WarmStartActorCritic
from rma_go2_lab.models.blind.history_actor_critic import TemporalBlindActorCritic
from rma_go2_lab.models.blind.ppo_with_flat_expert import BlindPPOWithFlatExpert
from rma_go2_lab.models.blind.ppo_with_v3_teacher import BlindPPOWithV3Teacher
from rma_go2_lab.models.teacher.actor_critic import TerrainEncoderActorCritic
from rma_go2_lab.models.teacher.ppo_with_terrain_aux import TeacherPPOWithTerrainAux
import rsl_rl.runners.on_policy_runner as _rsl_on_policy_runner

# Inject custom classes so rsl_rl's eval() can find them.
_rsl_on_policy_runner.TemporalBlindActorCritic = TemporalBlindActorCritic
_rsl_on_policy_runner.WarmStartActorCritic = WarmStartActorCritic
_rsl_on_policy_runner.BlindPPOWithFlatExpert = BlindPPOWithFlatExpert
_rsl_on_policy_runner.BlindPPOWithV3Teacher = BlindPPOWithV3Teacher
_rsl_on_policy_runner.TeacherPPOWithTerrainAux = TeacherPPOWithTerrainAux
_rsl_on_policy_runner.HistoryEncoderStudentActorCritic = HistoryEncoderStudentActorCritic
_rsl_on_policy_runner.FrozenAdaptV3Phase1 = FrozenAdaptV3Phase1
_rsl_on_policy_runner.ModularHistoryAdaptationActorCritic = ModularHistoryAdaptationActorCritic
_rsl_on_policy_runner.RmaV3Phase1PPO = RmaV3Phase1PPO
_rsl_on_policy_runner.RmaV3Phase2PPO = RmaV3Phase2PPO
_rsl_on_policy_runner.AdaptationPPOWithV3Expert = AdaptationPPOWithV3Expert
_rsl_on_policy_runner.AdaptationPPOWithV3Latent = AdaptationPPOWithV3Latent
_rsl_on_policy_runner.RmaV3ActorCritic = RmaV3ActorCritic
_rsl_on_policy_runner.TerrainEncoderActorCritic = TerrainEncoderActorCritic


def _register_task(task_id: str, env_cfg_entry_point: str, rsl_rl_cfg_entry_point: str, *,
                   entry_point: str = "isaaclab.envs:ManagerBasedRLEnv") -> None:
    gym.register(
        id=task_id,
        entry_point=entry_point,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": rsl_rl_cfg_entry_point,
        },
    )


# -----------------------------------------------------------------------------
# Priors
# -----------------------------------------------------------------------------
_register_task(
    "RMA-Go2-Flat",
    "rma_go2_lab.envs.priors.flat_cfg:Go2FlatPriorEnvCfg",
    "rma_go2_lab.models.priors.flat_ppo_cfg:Go2FlatPriorPPORunnerCfg",
)
_register_task(
    "Go2-Combined-Flat-MJLAB-Prior-V1",
    "rma_go2_lab.envs.priors.combined_flat_mjlab_prior_cfg:Go2CombinedFlatMjlabPriorEnvCfg",
    "rma_go2_lab.models.priors.combined_flat_mjlab_prior_runner_cfg:"
    "Go2CombinedFlatMjlabPriorPPORunnerCfg",
)


# -----------------------------------------------------------------------------
# Blind baselines and canonical C1 line
# -----------------------------------------------------------------------------
_register_task(
    "RMA-Go2-Blind-Baseline-Rough",
    "rma_go2_lab.envs.blind.rough_cfg:Go2BlindBaselineRoughEnvCfg",
    "rma_go2_lab.models.blind.variants_ppo_cfg:Go2BlindBaselineScratchPPORunnerCfg",
)
_register_task(
    "RMA-Go2-Blind-Baseline-Rough-WarmStart",
    "rma_go2_lab.envs.blind.rough_cfg:Go2BlindBaselineRoughEnvCfg",
    "rma_go2_lab.models.blind.variants_ppo_cfg:Go2BlindBaselineWarmStartPPORunnerCfg",
)
_register_task(
    "RMA-Go2-Blind-Baseline-Rough-WarmStart-Imitation",
    "rma_go2_lab.envs.blind.rough_cfg:Go2BlindBaselineRoughEnvCfg",
    "rma_go2_lab.models.blind.variants_ppo_cfg:Go2BlindBaselineWarmStartImitationPPORunnerCfg",
)
_register_task(
    "RMA-Go2-C1-ETHLike-V1-StageA",
    "rma_go2_lab.envs.blind.c1_ethlike_v1_cfg:Go2C1EthLikeV1EnvCfg",
    "rma_go2_lab.models.blind.variants_ppo_cfg:Go2C1EthLikeV1PPORunnerCfg",
)
_register_task(
    "RMA-Go2-C1-ETHLike-V3-StageA",
    "rma_go2_lab.envs.blind.c1_ethlike_v1_cfg:Go2C1EthLikeV1EnvCfg",
    "rma_go2_lab.models.blind.variants_ppo_cfg:Go2C1EthLikeV3PPORunnerCfg",
)


# -----------------------------------------------------------------------------
# Canonical privileged teachers
# Keep only the teacher roots that still matter in the frozen story.
# -----------------------------------------------------------------------------
_register_task(
    "RMA-Go2-Privileged-Teacher-Rough-V3",
    "rma_go2_lab.envs.teacher.rough_v3_cfg:Go2PrivilegedTeacherRoughV3EnvCfg",
    "rma_go2_lab.models.teacher.ppo_v3_cfg:Go2PrivilegedTeacherRoughV3PPORunnerCfg",
)
_register_task(
    "RMA-Go2-Privileged-Teacher-Rough-V4",
    "rma_go2_lab.envs.teacher.rough_v3_cfg:Go2PrivilegedTeacherRoughV3EnvCfg",
    "rma_go2_lab.models.teacher.ppo_v4_cfg:Go2PrivilegedTeacherRoughV4PPORunnerCfg",
)


# -----------------------------------------------------------------------------
# Active deployable asymmetric PPO mainline
# -----------------------------------------------------------------------------
_register_task(
    "Go2-Blind-Rough-MJLAB-AsymPPO-V1",
    "rma_go2_lab.envs.teacher.blind_rough_mjlab_asymppo_cfg:Go2BlindRoughMjlabAsymPpoEnvCfg",
    "rma_go2_lab.models.teacher.ppo_mjlab_asymppo_cfg:Go2BlindRoughMjlabAsymPpoRunnerCfg",
)

_register_task(
    "Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Rough-V1",
    "rma_go2_lab.envs.teacher.combined_rough_blind_mjlab_asymppo_cfg:"
    "Go2BlindRoughMjlabCombinedRoughEnvCfg",
    "rma_go2_lab.models.teacher.combined_rough_ppo_mjlab_asymppo_cfg:"
    "Go2BlindRoughMjlabCombinedRoughRunnerCfg",
)

_register_task(
    "Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1",
    "rma_go2_lab.envs.teacher.combined_steps_blind_rough_mjlab_asymppo_cfg:"
    "Go2BlindRoughMjlabCombinedStepsEnvCfg",
    "rma_go2_lab.models.teacher.combined_steps_ppo_mjlab_asymppo_cfg:"
    "Go2BlindRoughMjlabCombinedStepsRunnerCfg",
)

# -----------------------------------------------------------------------------
# Ablation Tests Task Registry
# -----------------------------------------------------------------------------

_register_task(
    "MJLAB-Combined-AsymPPO-Ablations",
    "rma_go2_lab.envs.teacher.final_ablations_asymppo_cfg:"
    "Go2BlindRoughMjlabCombinedRoughEnvCfg",
    "rma_go2_lab.models.teacher.final_ablations_asymppo_cfg:"
    "Go2BlindRoughMjlabCombinedRoughRunnerCfg",
)

# -----------------------------------------------------------------------------
# Legacy adaptation baselines
# Keep these because they still anchor comparison docs and eval artifacts.
# -----------------------------------------------------------------------------
_register_task(
    "RMA-Go2-Adaptation-Student-Rough-NoAdapt",
    "rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnvCfg",
    "rma_go2_lab.models.adaptation.no_adapt_ppo_cfg:Go2AdaptationStudentNoAdaptPPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)
_register_task(
    "RMA-Go2-Adaptation-Student-Rough-History",
    "rma_go2_lab.envs.adaptation.rough_history_cfg:Go2AdaptationStudentHistoryRoughEnvCfg",
    "rma_go2_lab.models.adaptation.adapt_ppo_cfg:Go2AdaptationStudentPPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)
_register_task(
    "RMA-Go2-Adaptation-Student-Rough-History-V1",
    "rma_go2_lab.envs.adaptation.rough_history_cfg:Go2AdaptationStudentHistoryRoughEnvCfg",
    "rma_go2_lab.models.adaptation.adapt_v1_ppo_cfg:Go2AdaptationStudentV1PPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)
_register_task(
    "RMA-Go2-Adaptation-Student-Rough-History-V2",
    "rma_go2_lab.envs.adaptation.rough_history_cfg:Go2AdaptationStudentHistoryRoughEnvCfg",
    "rma_go2_lab.models.adaptation.adapt_v2_ppo_cfg:Go2AdaptationStudentV2PPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)


# -----------------------------------------------------------------------------
# Adapt-V3 dyn-only mainline
# -----------------------------------------------------------------------------
_register_task(
    "RMA-Go2-Adapt-V3-Phase1-StageA",
    "rma_go2_lab.envs.adaptation.rough_history_stage_a_cfg:Go2AdaptationStudentHistoryStageARoughEnvCfg",
    "rma_go2_lab.models.adaptation.adapt_v3_ppo_cfg:Go2AdaptV3Phase1StageAPPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)
_register_task(
    "RMA-Go2-Adapt-V3-Phase2-StageA",
    "rma_go2_lab.envs.adaptation.rough_history_stage_a_cfg:Go2AdaptationStudentHistoryStageARoughEnvCfg",
    "rma_go2_lab.models.adaptation.adapt_v3_ppo_cfg:Go2AdaptV3DynOnlyPhase2StageAPPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)
_register_task(
    "RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch",
    "rma_go2_lab.envs.adaptation.rough_history_switch_recovery_dyn_only_cfg:"
    "Go2AdaptationStudentHistoryDynOnlyRecoverySwitchLowProbEnvCfg",
    "rma_go2_lab.models.adaptation.adapt_v3_ppo_cfg:"
    "Go2AdaptV3DynOnlyPhase2RecoveryLowProbPPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)
_register_task(
    "RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg",
    "rma_go2_lab.envs.adaptation.rough_history_switch_recovery_dyn_only_cfg:"
    "Go2AdaptationStudentHistoryDynOnlyRecoverySwitchLowProbEnvCfg",
    "rma_go2_lab.models.adaptation.adapt_v3_ppo_cfg:"
    "Go2AdaptV3DynOnlyPhase2RecoveryLowProbLatentRegPPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)
_register_task(
    "RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase1-StageA",
    "rma_go2_lab.envs.adaptation.rough_history_stage_a_cfg:Go2AdaptationStudentHistoryStageARoughEnvCfg",
    "rma_go2_lab.models.adaptation.adapt_v3_ppo_cfg:"
    "Go2AdaptV3DynOnlyStructuredZ27Phase1StageAPPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)
_register_task(
    "RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch",
    "rma_go2_lab.envs.adaptation.rough_history_switch_recovery_dyn_only_cfg:"
    "Go2AdaptationStudentHistoryDynOnlyRecoverySwitchLowProbEnvCfg",
    "rma_go2_lab.models.adaptation.adapt_v3_ppo_cfg:"
    "Go2AdaptV3DynOnlyStructuredZ27Phase2RecoveryLowProbLatentRegPPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)


# -----------------------------------------------------------------------------
# Adapt-V3 terrain-lite branch
# Keep for historical comparison, but separated from the dyn-only mainline.
# -----------------------------------------------------------------------------
_register_task(
    "RMA-Go2-Adapt-V3-TerrainLite-Phase1-PerEpisode",
    "rma_go2_lab.envs.adaptation.rough_history_per_episode_terrain_lite_cfg:"
    "Go2AdaptationStudentHistoryPerEpisodeTerrainLiteRoughEnvCfg",
    "rma_go2_lab.models.adaptation.adapt_v3_ppo_cfg:"
    "Go2AdaptV3TerrainLitePhase1PerEpisodePPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)
_register_task(
    "RMA-Go2-Adapt-V3-TerrainLite-Phase2-StageA",
    "rma_go2_lab.envs.adaptation.rough_history_stage_a_terrain_lite_cfg:"
    "Go2AdaptationStudentHistoryStageATerrainLiteRoughEnvCfg",
    "rma_go2_lab.models.adaptation.adapt_v3_ppo_cfg:"
    "Go2AdaptV3TerrainLitePhase2StageAPPORunnerCfg",
    entry_point="rma_go2_lab.envs.adaptation.rough_cfg:Go2AdaptationStudentRoughEnv",
)
