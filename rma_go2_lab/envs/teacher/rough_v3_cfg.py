"""Privileged teacher rough-terrain environment, V3.

Teacher V3 keeps the V2 terrain-only recipe fixed and adds explicit hidden
dynamics privilege:

- local terrain height scan in a dedicated terrain branch
- tracked friction privilege from the startup material randomizer
- tracked base-mass privilege from the startup mass randomizer
- live actuator stiffness/damping scale privilege from the articulation state

This file defines what privileged groups are exposed to the teacher. It should
not be read as proof that a frozen checkpoint uses every exposed group. The
later dependency audits on `model_1999` show clear dependence on
`dynamics_privileged`, while `terrain_privileged` appears unused on the
audited forward probes.
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.events import randomize_rigid_body_mass, randomize_rigid_body_material
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rma_go2_lab.envs.teacher.rough_cfg import TeacherPrivilegedObsCfg
from rma_go2_lab.envs.teacher.rough_v1_cfg import Go2PrivilegedTeacherRoughV1EnvCfg


class TrackedRandomizeRigidBodyMaterial(randomize_rigid_body_material):
    """Startup material randomizer that records per-env friction summaries."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env_static_friction = torch.ones(env.num_envs, 1, device=env.device)
        self.env_dynamic_friction = torch.ones(env.num_envs, 1, device=env.device)

    def __call__(self, *args, **kwargs):
        env = args[0]
        env_ids = args[1]
        super().__call__(*args, **kwargs)

        if env_ids is None:
            env_ids_cpu = torch.arange(env.num_envs, device="cpu")
        elif isinstance(env_ids, slice):
            env_ids_cpu = torch.arange(env.num_envs, device="cpu")[env_ids]
        else:
            env_ids_cpu = env_ids.cpu()

        materials = self.asset.root_physx_view.get_material_properties()[env_ids_cpu]
        self.env_static_friction[env_ids_cpu] = materials[:, :, 0].mean(dim=1, keepdim=True).to(env.device)
        self.env_dynamic_friction[env_ids_cpu] = materials[:, :, 1].mean(dim=1, keepdim=True).to(env.device)


class TrackedRandomizeRigidBodyMass(randomize_rigid_body_mass):
    """Startup mass randomizer that records the realized base-mass ratio."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env_base_mass_ratio = torch.ones(env.num_envs, 1, device=env.device)

    def __call__(self, *args, **kwargs):
        env = args[0]
        env_ids = args[1]
        super().__call__(*args, **kwargs)

        if env_ids is None:
            env_ids_cpu = torch.arange(env.num_envs, device="cpu")
        elif isinstance(env_ids, slice):
            env_ids_cpu = torch.arange(env.num_envs, device="cpu")[env_ids]
        else:
            env_ids_cpu = env_ids.cpu()

        if self.asset_cfg.body_ids == slice(None):
            body_ids_cpu = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids_cpu = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        masses = self.asset.root_physx_view.get_masses()[env_ids_cpu[:, None], body_ids_cpu]
        default_masses = self.asset.data.default_mass[env_ids_cpu[:, None], body_ids_cpu].cpu()
        mass_ratio = masses / torch.clamp(default_masses, min=1e-6)
        self.env_base_mass_ratio[env_ids_cpu] = mass_ratio.mean(dim=1, keepdim=True).to(env.device)


def tracked_static_friction(env) -> torch.Tensor:
    term = env.event_manager.cfg.physics_material.func
    return term.env_static_friction.clone()


def tracked_dynamic_friction(env) -> torch.Tensor:
    term = env.event_manager.cfg.physics_material.func
    return term.env_dynamic_friction.clone()


def tracked_base_mass_ratio(env) -> torch.Tensor:
    term = env.event_manager.cfg.add_base_mass.func
    return term.env_base_mass_ratio.clone()


def joint_stiffness_scale(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    scales = torch.zeros_like(asset.data.default_joint_stiffness)
    for actuator in asset.actuators.values():
        joint_ids = actuator.joint_indices
        scales[:, joint_ids] = actuator.stiffness / torch.clamp(
            asset.data.default_joint_stiffness[:, joint_ids],
            min=1e-6,
        )
    return scales


def joint_damping_scale(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    scales = torch.zeros_like(asset.data.default_joint_damping)
    for actuator in asset.actuators.values():
        joint_ids = actuator.joint_indices
        scales[:, joint_ids] = actuator.damping / torch.clamp(
            asset.data.default_joint_damping[:, joint_ids],
            min=1e-6,
        )
    return scales


@configclass
class TeacherDynamicsPrivilegedObsCfg(ObsGroup):
    """Low-dimensional hidden-dynamics privilege for Teacher V3."""

    static_friction = ObsTerm(func=tracked_static_friction, clip=(0.0, 3.0))
    dynamic_friction = ObsTerm(func=tracked_dynamic_friction, clip=(0.0, 3.0))
    base_mass_ratio = ObsTerm(func=tracked_base_mass_ratio, clip=(0.1, 4.0))
    joint_stiffness_scale = ObsTerm(
        func=joint_stiffness_scale,
        params={"asset_cfg": SceneEntityCfg("robot")},
        clip=(0.0, 3.0),
    )
    joint_damping_scale = ObsTerm(
        func=joint_damping_scale,
        params={"asset_cfg": SceneEntityCfg("robot")},
        clip=(0.0, 3.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Go2PrivilegedTeacherRoughV3EnvCfg(Go2PrivilegedTeacherRoughV1EnvCfg):
    """Teacher V3: V2 recipe plus explicit hidden-dynamics privilege.

    The environment exposes both terrain and dynamics privileged groups.
    Frozen-checkpoint usage must be established separately through audit.
    """

    def __post_init__(self):
        super().__post_init__()

        print("\n========== PRIVILEGED TEACHER ROUGH V3 ==========\n")

        # Track realized startup randomization so dynamics privilege reflects
        # what the simulator actually applied, not just configured ranges.
        self.events.physics_material.func = TrackedRandomizeRigidBodyMaterial
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.func = TrackedRandomizeRigidBodyMass

        # Split terrain and hidden-dynamics privilege into separate groups so
        # the model can encode terrain while consuming raw dynamics scalars.
        self.observations.privileged = None
        self.observations.terrain_privileged = TeacherPrivilegedObsCfg()
        self.observations.dynamics_privileged = TeacherDynamicsPrivilegedObsCfg()
