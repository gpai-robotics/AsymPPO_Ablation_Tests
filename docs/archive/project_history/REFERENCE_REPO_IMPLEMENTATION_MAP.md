# Reference Repo Implementation Map

This note captures what the major reference repos are doing that is directly relevant to our current struggles:

- making temporal/adaptation pathways actually matter
- keeping deploy contracts honest
- structuring sim2sim and sim2real without ambiguous glue

It is intentionally selective. The goal is not to summarize every repo, but to extract the implementation patterns that should influence our decisions.

Current repo-specific qualification:

- our frozen `Teacher V3` checkpoint should no longer be described as a
  teacher that is already using both terrain and dynamics privilege as
  intended
- current dependency audits show clear dependence on `dynamics_privileged`
- the same checkpoint appears to ignore `terrain_privileged` on the tested
  `random_rough`, `boxes`, and `pyramid_stairs` forward probes
- later recovery branches improved this, but not with one universal winner:
  - `Teacher V4 model_300` is the current canonical overall candidate
  - `Teacher V4.1 model_1999` is archived as a stair-specialized diagnostic
  - `Teacher V5` and `Teacher V6` are archived exploratory branches

## Main lesson

The strongest pattern across the major repos is:

- if a repo uses temporal or adaptive inference, it treats it as a first-class contract
- if a repo does not want that complexity, it stays explicitly stateless or recurrent
- the mature repos do not rely on "history was added, so it must be helping"

This is directly relevant to our C1 debugging. Our original failure mode came from adding a history path that existed architecturally but was not meaningfully participating in policy behavior.

The same lesson now applies one level upstream:

- exposing a privileged channel in the teacher architecture is not evidence
  that the frozen teacher uses it
- our current V3 audits show exactly that failure mode for terrain privilege

## Repo-by-repo

### `walk-these-ways-go2`

Most relevant files:

- `reference_repos/walk-these-ways-go2/go2_gym_learn/ppo/actor_critic.py`
- `reference_repos/walk-these-ways-go2/go2_gym_learn/ppo/ppo.py`
- `reference_repos/walk-these-ways-go2/go2_gym_learn/eval_metrics/metrics.py`
- `reference_repos/walk-these-ways-go2/go2_gym_deploy/envs/history_wrapper.py`
- `reference_repos/walk-these-ways-go2/go2_gym_deploy/scripts/deploy_policy.py`
- `reference_repos/walk-these-ways-go2/go2_gym_deploy/utils/deployment_runner.py`

What it does:

- separates the privileged encoder and student adaptation module explicitly
- trains the adaptation module with a direct supervised loss:
  - `adaptation_module(obs_history) ~= env_factor_encoder(privileged_obs)`
- exports deployment as two runtime pieces:
  - `adaptation_module_latest.jit`
  - `body_latest.jit`
- deploys with a real rolling history buffer instead of treating history as an optional extra

Why it matters for us:

- this repo does not merely hope that history becomes useful through PPO
- it explicitly supervises the history-to-latent pathway every update
- it also measures adaptation loss directly in eval metrics

Takeaway:

- if we want a history-conditioned or adaptive path to be trustworthy, we should either:
  - supervise it directly, or
  - ablate it repeatedly during training and eval
- "history is present in the network input" is not enough

### `unitree_rl_lab`

Most relevant files:

- `reference_repos/unitree_rl_lab/deploy/robots/go2/config/config.yaml`
- `reference_repos/unitree_rl_lab/deploy/robots/go2/src/State_RLBase.cpp`
- `reference_repos/unitree_rl_lab/deploy/include/FSM/State_RLBase.h`
- `reference_repos/unitree_rl_lab/deploy/include/isaaclab/envs/manager_based_rl_env.h`
- `reference_repos/unitree_rl_lab/deploy/include/isaaclab/algorithms/algorithms.h`
- `reference_repos/unitree_rl_lab/deploy/include/isaaclab/manager/observation_manager.h`

What it does:

- deploys through an FSM instead of a single monolithic script
- keeps a real mode structure:
  - `Passive`
  - `FixStand`
  - `Velocity`
- runs ONNX in C++ through a config-driven RL env wrapper
- defines observations, action scaling, default pose, stiffness, damping, and command ranges inside `deploy.yaml`
- supports observation history explicitly through `history_length` and `use_gym_history`

Why it matters for us:

- this is the cleanest example of deploy runtime as a system, not just policy inference
- it reduces ambiguity by turning training-time semantics into a deploy-time config contract
- history is part of the observation manager contract, not hidden in ad hoc Python glue

Takeaway:

- our bundle-driven `deploy.yaml` compatibility path is the right direction
- our long-term deploy target should be:
  - config-driven observation/action semantics
  - ONNX runtime support
  - FSM-managed hardware states

### `unitree_rl_gym`

Most relevant files:

- `reference_repos/unitree_rl_gym/README.md`
- `reference_repos/unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py`
- `reference_repos/unitree_rl_gym/deploy/deploy_real/deploy_real.py`

What it does:

- keeps the workflow very explicit:
  - `Train -> Play -> Sim2Sim -> Sim2Real`
- exports recurrent policies directly as `policy_lstm_1.pt`
- reuses the same observation semantics across MuJoCo and hardware

Why it matters for us:

- if a policy is recurrent, this repo keeps that explicit in export and deploy naming
- it avoids the "pseudo-memory but unclear runtime role" ambiguity

Takeaway:

- there are two clean ways to do temporal inference:
  - explicit recurrence with deploy-time hidden state
  - explicit history/adaptation supervision
- the messy middle is what we need to avoid

### `sim2real_unitree_sdk2py`

Most relevant files:

- `reference_repos/sim2real_unitree_sdk2py/example/go2/low_level/deploy/v1/dep_walk_v1.1.py`
- `reference_repos/sim2real_unitree_sdk2py/example/go2/low_level/docs/README.md`
- `reference_repos/sim2real_unitree_sdk2py/example/go2/low_level/docs/policy_contract.md`
- `reference_repos/sim2real_unitree_sdk2py/example/go2/low_level/debug/policyparser.py`

What it does:

- gives a disciplined hardware shell:
  - zero torque
  - move to default
  - hold default
  - start on operator signal
  - stop safely
- documents the observation/action contract very explicitly
- warns clearly when recurrent policies require persistent hidden state

Why it matters for us:

- this repo is still the best reference for practical first-hardware caution
- its `policyparser.py` warning is exactly the kind of guardrail we want:
  - recurrent/stateless mismatch should be called out directly

Takeaway:

- the old repo remains useful as the operational safety shell
- we should keep stealing that style of preflight validation

### `mujoco_playground`

Most relevant files:

- `reference_repos/mujoco_playground/mujoco_playground/experimental/sim2sim/README.md`
- `reference_repos/mujoco_playground/mujoco_playground/experimental/sim2sim/play_*`
- `reference_repos/mujoco_playground/learning/train_rsl_rl.py`

What it does:

- uses ONNXRuntime for lightweight sim2sim inference
- keeps joystick/gamepad control simple and explicit
- uses asymmetric actor/critic observations in training configs

Why it matters for us:

- this is a good reference for a lightweight ONNX deploy surface
- it is not the right repo to copy for our full deploy architecture

Takeaway:

- borrow ONNX runtime simplicity
- do not replace our bundle contract with ad hoc single-script deployment

### `rl_locomotion`

Most relevant files:

- `reference_repos/rl_locomotion/README.md`
- `reference_repos/rl_locomotion/raisimGymTorch/algo/ppo/dagger.py`

What it does:

- explicitly distills privileged behavior into a blind student
- supervises the student latent/history encoder against expert-produced latent structure
- keeps the student and expert roles cleanly separated

Why it matters for us:

- this is another strong confirmation that when hidden structure matters, successful repos supervise it directly

Takeaway:

- our current C1 path is conceptually closer to distillation than to pure blind PPO
- that means direct supervision of the temporal path is a natural next design option, not a hack

### Root audits matter too

The history-path failure taught us to ablate the student path. The teacher
dependency audits extend that rule:

- the frozen `Teacher V3` checkpoint changes materially under
  `zero_dynamics`
- it does not change at all under `zero_terrain` on the audited forward probes

So the repo now needs to treat teacher-side pathway audits the same way it
treats student-side history audits:

- inputs exposed is not enough
- frozen-checkpoint dependence must be demonstrated

### `quadrupeds_locomotion`

Most relevant file:

- `reference_repos/quadrupeds_locomotion/README.md`

What it contributes:

- a clean educational baseline for command tracking with 48D-ish proprioceptive observations and residual joint targets

Why it matters less:

- this repo is useful for baseline observation/action logic
- it does not answer our temporal-path or adaptation-path debugging questions as strongly as the repos above

## What the reference repos collectively say about our current struggle

### 1. Our original history failure was not unusual

The reference repos implicitly protect against this in one of three ways:

- direct supervision of the temporal/adaptation path
- explicit recurrent deploy contracts
- no temporal path at all

The one thing they do not do is silently assume a newly added history pathway will become important on its own.

### 2. Eval must test the temporal mechanism, not just the final score

`walk-these-ways-go2` is especially important here:

- it logs adaptation loss directly
- it treats the latent-prediction problem as something measurable

That supports what we already learned the hard way:

- rollout reward alone is not enough to validate that history is doing useful work

### 3. Deploy contracts need to encode semantics, not just file paths

`unitree_rl_lab` is the clearest example:

- `deploy.yaml` carries observation history, joint map, command ranges, action scaling, stiffness, damping, and default pose

That reinforces our current direction:

- frozen bundle + explicit deploy metadata is the right foundation

### 4. Hardware flow should stay conservative and stateful

`sim2real_unitree_sdk2py` and `unitree_rl_lab` both reinforce this:

- safe stance gate before RL
- explicit operator transition
- clear emergency exit path
- deploy runtime is more than policy inference

## Direct implications for us

### What we should keep doing

- keep the bundle-driven deploy contract
- keep ONNX compatibility work
- keep MuJoCo as a deploy-side validation gate
- keep explicit history ablations in Isaac eval

### What we should change or add

1. Add a first-class temporal-path validation signal during training.

Good candidates:

- latent prediction loss against privileged teacher features
- periodic history ablation eval on checkpoints
- adaptation/path activation diagnostics

2. Stop treating "history present in config" as proof of usefulness.

We should only claim history matters when at least one of these is true:

- direct supervised metric improves
- ablation hurts clearly on trusted probes
- recovery metrics improve on switch/disturbance tasks

3. Push deploy closer to config-native semantics.

The `unitree_rl_lab` style is the right end-state:

- deploy config defines what observations are
- runtime consumes those groups directly
- history length is explicit and not buried in side logic

## Most actionable repo answers

If we want the shortest list of answers from the reference repos, it is this:

1. `walk-these-ways-go2`
   - tells us how to make history/adaptation measurable and deployable
2. `unitree_rl_lab`
   - tells us what the mature deploy architecture should look like
3. `sim2real_unitree_sdk2py`
   - tells us how cautious first hardware bring-up should feel
4. `unitree_rl_gym`
   - reminds us that explicit recurrence is cleaner than ambiguous pseudo-memory
5. `rl_locomotion`
   - reinforces that distillation-style hidden structure is often directly supervised

## Bottom line

The reference repos do not suggest that our overall direction is wrong.

They suggest something more specific:

- our deploy contract direction is good
- our MuJoCo rigor work is good
- our hardware shell direction is good
- but our temporal/history validation was too weak, and our first implementation let the path die silently

The major repos solve that by making temporal structure explicit, supervised, and measurable.

That is the main answer hidden across these repos, and it is probably the most important one for us right now.
