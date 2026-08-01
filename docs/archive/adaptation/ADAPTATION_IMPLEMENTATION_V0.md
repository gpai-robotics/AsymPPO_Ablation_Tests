# Adaptation Implementation V0

This note locks the first concrete adaptation-phase scaffold after freezing the
teacher chapter at `V3 final`.

## Scope

This is **not** the full adaptation module yet.

It establishes the first real code branch for:

- adaptation-phase environment ownership
- deployable student observation interface
- no-adaptation student baseline registration

The point is to create a clean base for the comparison ladder:

1. blind robust baseline
2. student without adaptation
3. student with adaptation
4. privileged expert (`V3 final`)

## Current Pipeline

The repo now implements the following pipeline:

```text
Blind Baselines (B1/B2/B3)
  proprio-only PPO
  48-dim deployable observation
  no privileged channels
          |
          v
Privileged Teacher Phase (V0 -> V3)
  V3 final = frozen privileged expert
  policy: 48
  terrain_privileged: 187 -> terrain encoder -> 8
  dynamics_privileged: 27
  actor input: 48 + 8 + 27 = 83
          |
          v
Adaptation Phase B: studentNA
  task: RMA-Go2-Adaptation-Student-Rough-NoAdapt
  deployable proprio-only student
  one hidden mid-episode switch per env per episode
  no history latent
          |
          v
Adaptation Phase C: studentAdapt
  task: RMA-Go2-Adaptation-Student-Rough-History
  student consumes:
    - policy (48)
    - policy_history (960) -> history encoder -> z_hat_t (8)
  actor input: 48 + 8 = 56
  frozen V3 teacher consumes:
    - policy
    - terrain_privileged
    - dynamics_privileged
  training signal:
    - PPO
    - teacher action imitation
```

In words:

- `B2` gives us a strong deployable blind locomotion prior.
- `V3 final` gives us a strong privileged upper bound.
- `studentNA` measures how far a deployable policy can get under hidden
  within-episode changes without adaptation machinery.
- `studentAdapt` keeps the deployable interface clean, adds a history encoder,
  and learns under supervision from the frozen privileged expert.

## Locked Expert

- `V3 final`
- `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v3/2026-04-21_15-35-03/model_1999.pt`

## Phase-B Baseline

Registered task target:

- `RMA-Go2-Adaptation-Student-Rough-NoAdapt`

This baseline uses:

- the deployable proprio-only observation interface
- no height scan
- no direct dynamics privilege
- no history latent
- one sampled hidden dynamics switch per env per episode

Current switch family:

- ultra-low friction
- very heavy payload
- very weak motor

## Future Scaffold Added

The repo now also carries a future-facing Phase-C scaffold:

- history observation task:
  - `RMA-Go2-Adaptation-Student-Rough-History`
- history observation group:
  - `policy_history`
- future student model:
  - `HistoryEncoderStudentActorCritic`
- frozen teacher wrapper:
  - `FrozenV3Expert`
- current adaptation algorithm path:
  - `AdaptationPPOWithV3Expert`

This scaffold is intentionally not treated as the final adaptation recipe yet.
It exists so the next implementation step is incremental rather than a fresh
architectural jump.

## Initial Interface Decisions

These are locked unless we find a concrete reason to change them:

- deployable policy observation dim: `48`
- initial latent size candidates for Phase C: `8`, `16`
- initial history window candidates for Phase C: `20`, `50` steps

The current scaffold only implements the no-adaptation baseline. The history
encoder and latent-consuming student will be added on top of this branch.

That branch is now partly realized:

- `studentNA` is the active no-adaptation baseline
- `studentAdapt` is the active history-student route
- the current supervision path is teacher action imitation, not yet latent
  regression

## Adapt-V0 Final Status

The restarted post-fix `Adapt-V0` comparison is now frozen.

Frozen checkpoints:

- `studentNA`
  - `/home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_no_adapt_v0/2026-04-22_15-22-13/model_1999.pt`
- `studentAdapt-V0`
  - `/home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v0/2026-04-22_15-22-16/model_1999.pt`

Final outcome:

- `studentAdapt-V0` finished slightly stronger overall than `studentNA`
- this is enough to count as the first positive adaptation result in the repo
- the detailed freeze/write-up now lives in:
  - `docs/ADAPTATION_PHASE_SYNTHESIS.md`

## Adapt-V1 Decision

We are explicitly committing to a next-step adaptation route beyond pure
teacher action imitation.

The naming is:

- `Adapt-V0`
  - current history student
  - PPO + frozen teacher action imitation
- `Adapt-V1`
  - explicit latent prediction from history
  - student policy consumes `z_hat_t`
  - frozen teacher exports a canonical latent target
  - PPO remains in the loop
  - teacher action imitation becomes optional auxiliary supervision

### Locked first explicit latent target

For the first `Adapt-V1` implementation, the latent target is:

- frozen teacher actor penultimate feature
- dimension `128`

This is intentionally practical.

We are **not** starting by regressing raw friction/mass/motor values.

We originally tried to target the frozen teacher terrain latent, but explicit
validation showed that the saved `V3` checkpoint's terrain encoder is fully
zeroed, making that target useless for latent regression. `Adapt-V1` therefore
switches to a live internal teacher feature on the actor path.

Why this choice:

- it sits on a path the teacher actually uses to produce actions
- it gives a non-degenerate target with measurable pre/post-switch signal
- it avoids inventing an extra projection head before we know we need one

### Planned `Adapt-V1` training shape

The intended student path becomes:

```text
policy_history -> history encoder -> z_hat_t (128)
policy + z_hat_t -> student actor / critic
```

The intended frozen teacher path becomes:

```text
policy + terrain_privileged + dynamics_privileged
  -> frozen teacher actor penultimate feature
  -> z_target (128)
```

And the intended loss stack becomes:

- PPO loss
- latent regression loss:
  - `MSE(z_hat_t, z_target)`
- optional weak action imitation loss early in training

### Expected follow-up after the current runs

With `studentNA` and `studentAdapt-V0` now frozen, the next implementation
step is:

1. run the full eval ladder on both frozen checkpoints
2. preserve `Adapt-V0` as the first completed adaptation result
3. continue with:
   - `Adapt-V1` (explicit latent prediction)
4. compare:
   - `studentNA`
   - `Adapt-V0`
   - `Adapt-V1`

### Validation status

`Adapt-V1` has now passed the key pre-training validation checks:

- task/env/runner registration is correct
- student and teacher latent dims match at `128`
- hidden switch mechanism is reached in the validator
- latent regression loss is nonzero in both the standalone debug harness and a
  1-iteration PPO smoke run

The current remaining uncertainty is behavioral quality, not wiring integrity.

## Adapt-V2 Decision

`Adapt-V2` is now reserved for the first explicitly modular RMA-like
implementation in the repo.

Intended structure:

```text
phi(history) -> z_hat_t
pi(current_obs, z_hat_t) -> action
```

This differs from `V1` in one important way:

- `V1` has explicit latent supervision but remains architecturally bundled
- `V2` makes the module split itself explicit in code

Current `V2` scaffold:

- policy scaffold:
  - `rma_go2_lab/models/adaptation/modular_actor_critic.py`
- config:
  - `rma_go2_lab/models/adaptation/adapt_v2_ppo_cfg.py`
- registered task:
  - `RMA-Go2-Adaptation-Student-Rough-History-V2`
- design note:
  - `docs/archive/adaptation/ADAPTATION_V2_PLAN.md`

### Separate file ownership

`Adapt-V1` is intentionally kept in separate files so it can evolve without
destabilizing the current imitation route.

Current split:

- `Adapt-V0`
  - config:
    `rma_go2_lab/models/adaptation/adapt_ppo_cfg.py`
  - algorithm:
    `rma_go2_lab/models/adaptation/ppo_with_v3_expert.py`
  - task:
    `RMA-Go2-Adaptation-Student-Rough-History`

- `Adapt-V1`
  - config:
    `rma_go2_lab/models/adaptation/adapt_v1_ppo_cfg.py`
  - algorithm:
    `rma_go2_lab/models/adaptation/ppo_with_v3_latent.py`
  - task:
    `RMA-Go2-Adaptation-Student-Rough-History-V1`

- `Adapt-V2`
  - config:
    `rma_go2_lab/models/adaptation/adapt_v2_ppo_cfg.py`
  - policy scaffold:
    `rma_go2_lab/models/adaptation/modular_actor_critic.py`
  - task:
    `RMA-Go2-Adaptation-Student-Rough-History-V2`
