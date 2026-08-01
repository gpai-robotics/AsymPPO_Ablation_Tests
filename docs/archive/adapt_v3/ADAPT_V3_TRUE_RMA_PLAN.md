# Adapt-V3 True RMA Plan

This note defines the next major adaptation branch after `V0`, `V1`, and `V2`.

`Adapt-V3` is reserved for the first version in this repo that would count as a
complete original-RMA-style architecture in the important structural sense.

Interpretation rule:

- `V3` should be as faithful as practical to original RMA
- differences are acceptable only when they are forced by the changed robot,
  simulator, or training regime
- any such differences must be named explicitly, not hidden under the label
"RMA"

Current repo note:

- the first terrain-plus-dynamics `V3` implementation was valuable, but it is
  now treated as exploratory lineage rather than the active forward path
- the active reboot starts from a dynamics-only latent so we can answer the
  more basic question first: can a blind history student reliably recover
  hidden dynamics in this stack without behavior collapsing?

## Why V3 exists

The current progression has been valuable, but it still stops short of the full
original RMA contract.

Current state:

- `V0`
  - adaptation works
  - teacher action imitation
  - architecture bundled
- `V1`
  - explicit latent prediction
  - latent target taken from a useful internal teacher feature
  - still bundled at the student level
- `V2`
  - explicit modular split:
    - `phi(history) -> z_hat`
    - `pi(current_obs, z_hat) -> action`
  - but still targets a teacher hidden feature rather than an explicit
    privileged extrinsics latent

So the missing step is:

- a true privileged-factor encoder `mu`
- a base policy explicitly conditioned on `z`
- an adaptation module explicitly trained to recover that same `z`

That is the purpose of `Adapt-V3`.

## Target Architecture

The intended architecture is:

```text
Training in simulation:

e_t -> mu -> z_t
x_t, a_t-1, z_t -> pi -> a_t

history_t = (x_t-k:t, a_t-k-1:t-1) -> phi -> z_hat_t
```

Deployment:

```text
history_t -> phi -> z_hat_t
x_t, a_t-1, z_hat_t -> pi -> a_t
```

This is the first point where the project can honestly claim the full original
RMA-style latent contract.

## Explicit Modules

### 1. Extrinsics encoder

`mu`

Input:

- privileged teacher-side factors `e_t`

Recommended `e_t` contents:

- terrain geometry signal
- friction signal
- mass / payload signal
- motor strength signal

The first faithful `V3` should prefer the closest practical subset of original
RMA extrinsics before widening the privileged factor set.

Output:

- compact latent `z_t`

This latent is the teacher-side extrinsics code.

### 2. Base policy

`pi`

Input:

- deployable current observation `x_t`
- latent `z_t` during teacher/base-policy training
- latent `z_hat_t` during deployment/adaptation

Output:

- action distribution / action mean

Important rule:

- privileged factors must influence the action path only through `z`

That makes `z` a true bottleneck rather than an optional side channel.

### 3. Adaptation module

`phi`

Input:

- recent deployable history

Recommended contents:

- current history design reused from `V1`/`V2`
- recent proprio signals
- recent commands
- recent actions

Output:

- `z_hat_t`

## Required Contract Changes Relative To V1/V2

To count as a true RMA-style architecture, `V3` must change these things:

### Teacher target

Current:

- penultimate actor feature from frozen teacher

Required for `V3`:

- explicit teacher extrinsics latent `z_t = mu(e_t)`

### Base-policy bottleneck

Current:

- teacher path can still be interpreted as a full privileged policy

Required for `V3`:

- `pi` must be trained as a latent-conditioned base policy
- not as a policy that simply happens to expose a useful hidden feature later

### Student supervision target

Current:

- useful teacher hidden feature

Required for `V3`:

- the same explicit extrinsics latent used by the base policy

## Proposed Training Phases

### Phase 1: Train latent-conditioned privileged base policy

Train:

- `mu`
- `pi`

using privileged simulation data.

Teacher-side path:

```text
e_t -> mu -> z_t
x_t, z_t -> pi -> a_t
```

Optimization:

- PPO

Optional:

- light regularization on `z`
- latent norm control
- latent dimensionality constraint

But avoid making the first version too complicated.

### Phase 2: Train adaptation module

Freeze:

- `mu`
- `pi`

Train:

- `phi`

with target:

- `z_t = mu(e_t)`

Primary loss:

- latent regression loss

Optional auxiliary:

- weak action imitation from frozen `pi(x_t, z_t)`

### Phase 3: Joint deployable fine-tuning

Optional later step:

- fine-tune `phi + pi` together under PPO

using:

- deployable observations only
- `phi(history) -> z_hat_t`
- `pi(x_t, z_hat_t) -> a_t`

This is optional because the first real milestone is to get the clean
two-phase contract working at all.

## Latent Design Choices

These need to be decided deliberately before implementation.

### Choice 1: Fused latent vs structured latent

Option A:

- single fused latent `z`

Option B:

- structured latent:
  - `z_terrain`
  - `z_dynamics`

Recommendation for first `V3`:

- single fused latent

Reason:

- simpler contract
- easier deployment path
- closer to the original “extrinsics latent” spirit

### Choice 2: Latent dimension

Possible starting points:

- `8`
- `16`
- `32`

Recommendation:

- start with `16`

Reason:

- `8` may be too aggressive for fused terrain+dynamics
- `32` may be unnecessarily loose
- `16` is a reasonable first bottleneck

### Choice 3: Latent normalization

Recommendation:

- keep latent scale stable
- consider simple normalization or bounded regularization

Reason:

- easier adaptation regression
- easier interpretation
- less fragile deployment

## Deployment Contract

The intended deployable runtime should eventually be:

1. collect recent history
2. update `z_hat_t = phi(history)` at a slower rate
3. compute `a_t = pi(x_t, z_hat_t)` at a faster rate

This means `V3` should be designed to support:

- cached latent state
- explicit separation between adaptation updates and action updates

The first implementation does not need full asynchronous threading, but the
module boundaries should make that possible.

## File-Level Design

Recommended new file family:

- `rma_go2_lab/models/adaptation/extrinsics_encoder.py`
  - explicit `mu`
- `rma_go2_lab/models/adaptation/base_policy_actor_critic.py`
  - latent-conditioned `pi`
- `rma_go2_lab/models/adaptation/rma_v3_actor_critic.py`
  - wrapper/composition of `phi + pi`
- `rma_go2_lab/models/adaptation/ppo_with_explicit_extrinsics.py`
  - training logic for explicit `mu/pi/phi` contract
- `rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py`
  - runner config

Documentation:

- `docs/archive/adapt_v3/ADAPT_V3_TRUE_RMA_PLAN.md`

Task registration:

- `RMA-Go2-Adaptation-Student-Rough-History-V3`

## Success Criteria

Minimum success:

- explicit `mu(e_t) -> z_t` exists
- `pi` consumes only `z` as the privilege bottleneck
- `phi` is trained against that same `z`
- deployable path uses `z_hat`

Stronger success:

- `V3` matches or exceeds `V1`
- `V3` matches or exceeds `V2`
- modular deployment path is cleaner
- adaptation story becomes more faithful to original RMA

## What V3 Still Would Not Automatically Solve

Even with a complete original-style latent contract, `V3` would still not
automatically solve:

- hardware deployment
- body-stability quality
- exteroceptive foresight for stairs/debris
- all reward-design problems

So `V3` should be understood as:

- architectural completion of the RMA-style adaptation contract

not:

- the final answer to every locomotion problem in the repo

## Recommended Timing

Do not start `Adapt-V3` as an active training branch until:

- `V1` is frozen
- `V2` is either frozen or clearly judged
- current adaptation closeout is complete

But the design should be considered locked now so the next branch starts from a
crisp target instead of vague discussion.
