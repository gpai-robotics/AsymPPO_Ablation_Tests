# GO2 Old Robot Baseline And Experiment Split 2026-05-29

This note records the clean separation between:

- the protected working old-robot baseline
- the isolated experimentation runtime

## Protected Baseline

The protected working runtime remains:

- runtime root:
  - `reference_repos/unitree_rl_lab`
- active bundle path:
  - `reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final`
- launcher:
  - `scripts/deploy/run_go2_old_robot_stack.sh`

The deploy config snapshot for the current working baseline, including the
joint-target slew limit experiment that improved responsiveness, is:

- `reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy_frozen_old_robot_joint_slew_20260529.yaml`

This baseline should not be edited further during risky experiments.

## Isolated Experiment Runtime

A full cloned runtime is available at:

- `reference_repos/unitree_rl_lab_go2_old_robot_experiments`

This clone is the place for:

- code experiments
- YAML experiments
- build experiments
- stopping / blending / control-bridge experiments

without touching the protected baseline runtime.

### Experiment Launcher

Use:

- `scripts/deploy/run_go2_old_robot_experiment_stack.sh`

This launcher uses:

- experimental runtime root:
  - `reference_repos/unitree_rl_lab_go2_old_robot_experiments`
- experimental params dir:
  - `reference_repos/unitree_rl_lab_go2_old_robot_experiments/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params`
- experimental logs:
  - `logs/go2_ctrl_experiment`

### Experimental Control Binary

Experimental control logging is isolated through:

- `scripts/deploy/run_go2_ctrl_logged_experiment.sh`

This ensures future rebuilds and control experiments can happen without mixing
logs with the protected baseline.

## Operational Rule

Going forward:

- baseline validation runs:
  - use `run_go2_old_robot_stack.sh`
- risky runtime or control experiments:
  - use `run_go2_old_robot_experiment_stack.sh`

This keeps the working baseline available at all times while still allowing
aggressive experimentation in a separate runtime tree.
