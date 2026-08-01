# Adapt-V3 Execution Spec

This note converts the `Adapt-V3` idea into an implementation-ready spec.

Use this file when we actually start the final original-style RMA branch.

Read this after:

- `docs/archive/adapt_v3/ADAPT_V3_TRUE_RMA_PLAN.md`
- `docs/PROJECT_GUIDE.md`

## Active Reset

The original terrain-plus-dynamics `Adapt-V3` line produced important barrier
records and frozen historical artifacts, but it did not survive the later
closed-loop adaptation stages cleanly enough to remain the active contract.

The active reboot narrows `mu` to hidden dynamics only:

- friction
- mass / payload
- motor strength

Terrain geometry is now treated as a deferred question rather than something
the blind history student is assumed to recover reliably from the outset.

Current frozen artifact for this reboot:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt`

## Faithfulness Rule

`V3` should be paper-faithful to original RMA wherever that contract can be
carried into this repo without distortion.

That means:

- preserve the two-phase training structure
- preserve the explicit `mu`, `pi`, `phi` decomposition
- preserve the actor-side latent bottleneck
- preserve explicit use of current observation, previous action, and latent
- preserve the history-to-latent adaptation pathway
- preserve the deployment idea of slower adaptation and faster control

At the same time, `V3` does not need to be literally identical in every detail,
because this repo differs in:

- robot platform
- simulator stack
- observation naming
- training regime
- available privileged factors

Allowed deviations are only acceptable when all of these are true:

1. they are forced by the repo's different regime or interfaces
2. they do not violate the core RMA contract
3. they are explicitly named in docs as a repo-specific adaptation
4. they have a defensible reason, not just convenience

So the right target is:

- as close to original RMA as possible
- without pretending this repo is a byte-for-byte reproduction

## Goal

`Adapt-V3` is the first branch that should satisfy the full intended
teacher-to-student latent contract:

```text
Phase 1:
e_t -> mu -> z_t
x_t, z_t -> pi -> a_t

Phase 2:
history_t -> phi -> z_hat_t
x_t, z_hat_t -> pi -> a_t
```

The purpose of this branch is to stop approximating that contract and implement
it explicitly.

## Major Barrier Record

`Adapt-V3` did not become viable in one step. The branch crossed multiple
distinct bottlenecks, and those barriers are part of the implementation
knowledge now.

### Barrier 1: latent bypass under strong actor priors

Observed failure:

- early `V3` Phase 1 could achieve superficially healthy reward while
  `mu(e_t)` remained dead or nearly constant
- debug checks showed the policy could ignore latent variation almost
  completely

Diagnosis:

- a strong actor-side locomotion shortcut allowed `pi` to solve the task
  without depending meaningfully on `z`
- this made the architecture look RMA-like structurally while violating the
  real latent contract functionally

What changed:

- remove actor warm-start for true `V3` Phase 1
- keep explicit latent supervision and latent-usage checks

### Barrier 2: latent collapse even after removing the easy bypass

Observed failure:

- raw privileged inputs varied correctly across envs
- but `mu(e_t)` still collapsed many different envs into nearly the same latent

Diagnosis:

- plain PPO and weak auxiliary shaping were not enough to force a
  discriminative extrinsics code

What changed:

- add direct latent anchoring and structured auxiliary supervision
- add pairwise structure pressure and latent-variation pressure
- use `debug_adapt_v3.py` as a required acceptance check rather than trusting
  reward alone

### Barrier 3: locomotion bootstrap failure under true load-bearing latent use

Observed failure:

- once `mu` became genuinely load-bearing, pure-switch and early mixed-regime
  Phase 1 runs often failed to form a real gait
- visually, robots frequently stayed in place, pitched forward/backward, and
  failed to progress terrain curriculum

Diagnosis:

- the first fully faithful `mu + pi` problem was too hard as a from-scratch
  locomotion task in this repo's switched adaptation regime
- this was not just a latent problem anymore; it was a locomotion bootstrap
  problem

What changed:

- split Phase 1 into a stationary `Stage A` and later adaptation-heavy `Stage B`
- `Stage A` keeps startup/reset randomization but removes within-episode
  switches
- use critic-only warm-start rather than actor warm-start
- add temporary action imitation from the frozen `B2` blind policy as an early
  locomotion scaffold

### Stage A breakthrough

The key `V3` breakthrough was the first strong `Stage A` run:

- locomotion became real again
- terrain curriculum rose strongly instead of collapsing
- latent usage remained real under debug checks

This is the point where `V3` stopped being only a structural ambition and
became a workable RMA-style branch foundation.

The decisive combination was:

1. actor fresh, not actor warm-started
2. critic-only warm-start from `B2`
3. stationary randomized episodes for locomotion bootstrap
4. temporary action imitation scaffold from the frozen blind baseline
5. explicit latent debugging to ensure success was not coming from another
   hidden bypass

This sequence should be treated as a core repo lesson, not an incidental tuning
detail.

## Why This Exists

`V3` is not a complexity-for-its-own-sake branch.

It exists because the adaptation problem itself has an awkward structure:

- the controller needs hidden environment information
- that information is available in simulation during training
- that same information is not available at deployment
- deployment still requires online adaptation from recent behavior

Simpler approaches each fail on at least one of those constraints:

- direct privileged conditioning works in simulation but is not deployable
- blind policies are deployable but cannot rely on hidden-factor inference
- bundled history policies can adapt, but the learned representation becomes
  hard to interpret and hard to align with the original RMA contract
- imitation-only or teacher-feature matching can help adaptation, but they do
  not by themselves guarantee a clean privileged-to-deployable latent interface

The RMA decomposition is the clean response to those constraints:

1. during training, use privileged factors to define a compact hidden state of
   the world
2. train the motor policy to depend on that hidden state
3. at deployment, replace privileged access with a history-based estimator of
   that same hidden state

So the purpose of `V3` is not merely to be more elaborate than `V1` or `V2`.
The purpose is to make the information pathway explicit enough that:

- the policy depends on a meaningful latent world description
- that description can be learned during training using privileged factors
- the same interface can later be supplied by a deployable adaptation module

This is also why `V3` must be careful about collapse and bypasses:

- if the actor can solve locomotion without using `z`, then `mu` can collapse
- if `mu` collapses, the architecture may look like RMA on paper while failing
  to carry meaningful environment state in practice

So a major part of `V3` implementation is not just drawing the `mu / pi / phi`
boxes. It is making the latent genuinely load-bearing.

## Locked Design Choices

These choices are intentionally fixed now so we do not keep re-deciding them
mid-implementation.

### Latent shape

- use a single fused latent `z`
- latent dimension: `32`

Why:

- smaller than `V1`/`V2`'s current `128`, which should reduce target hardness
- larger than the original tiny-latent spirit, so we keep enough capacity for
  terrain + dynamics together
- simple to reason about and debug

### Privileged teacher factors `e_t`

Faithful default:

- keep `e_t` as close as practical to the original RMA extrinsics notion
- only include factors we believe belong to the hidden environment / robot
  condition state

First `V3` default:

- terrain geometry signal
- friction signal
- mass / payload signal
- motor strength signal

Repo-specific note:

- because our current env exposes richer privileged channels than the paper, we
  should begin with the closest faithful subset first and only widen `e_t`
  later if there is a documented reason
Implementation mapping in this repo should start from:

- `terrain_privileged`
  - height-derived terrain signal
- `dynamics_privileged`
  - friction
  - base-mass ratio
  - motor-strength-related signal

Avoid adding stiffness / damping scales to the first faithful `V3` unless we
explicitly decide they are part of our repo-native extension beyond original
RMA.

No other privileged side channel is allowed into the action path.

### Deployable current observation `x_t`

`x_t` remains the current deployable proprioceptive observation, and `a_{t-1}`
must be treated as an explicit part of the actor contract.

For this repo, the actor contract should be documented as:

```text
pi(x_t, a_{t-1}, z_t)
```

where `a_{t-1}` may be carried inside the current `policy` group as long as the
implementation and docs make that explicit.

Current repo mapping:

- base lin vel
- base ang vel
- projected gravity
- velocity commands
- joint pos
- joint vel
- previous actions

### History input

Faithful default:

- `phi(x_{t-k:t}, a_{t-k-1:t-1})`

For the first faithful `V3` pass, prefer a history length closer to the paper
contract rather than simply inheriting the current shorter repo window.

Default target:

- `50` steps of history if the current training stack tolerates it
- if not, use the nearest stable repo-supported window and document the
  deviation explicitly

So:

- do not silently inherit the current `20`-frame window just for convenience
- either move toward `50`, or name the deviation and justify it

### Training order

`V3` is a two-phase branch with an optional third step:

1. train `mu + pi`
2. freeze `mu + pi`, train `phi`
3. optional joint deployable fine-tune

Do not collapse these phases into one monolithic training recipe on the first
pass.

### Deployment rate semantics

Faithful target:

- adaptation path conceptually at `10 Hz`
- base policy conceptually at `100 Hz`

For the first `V3` implementation in this repo:

- keep training synchronous
- but implement the model so deployment can call:
  - `phi` at a slower external rate
  - `pi` at every control step

So the code should expose the split cleanly even if the first training run does
not yet execute it asynchronously inside PPO.

## Exact Module Contract

### 1. Extrinsics encoder `mu`

Input:

- concatenated privileged factor vector `e_t`

Output:

- `z_t` with shape `(batch, 32)`

Rules:

- `mu` is the only module allowed to read `e_t`
- `z_t` is the only latent target for `phi`

### 2. Base policy `pi`

Input:

- current deployable observation `x_t`
- previous action `a_{t-1}`
- latent `z_t` during Phase 1
- latent `z_hat_t` during Phase 2 and deployment

Output:

- action mean / distribution

Rules:

- no direct privileged factor bypass
- no direct history input to the actor in `V3`
- the actor should depend on history only through `z_hat_t`

### 3. Adaptation module `phi`

Input:

- history of deployable observations and actions
- faithful contract:
  - `phi(x_{t-k:t}, a_{t-k-1:t-1})`

Output:

- `z_hat_t`

Rules:

- `phi` never reads privileged terms
- `phi` is supervised against `z_t = mu(e_t)`

### 4. Critic

For the first `V3` version, the critic may remain pragmatic:

- critic input can include `x_t`
- critic input can include `z_t` / `z_hat_t`
- critic can optionally still consume history if that keeps PPO stable

Important:

- the actor path must obey the clean bottleneck
- critic strictness is secondary on the first pass

## Planned Files

These are the intended new files for the first `V3` implementation.

### Models

- `rma_go2_lab/models/adaptation/rma_v3_actor_critic.py`
  - defines `mu`, `pi`, `phi`
  - explicit helper methods:
    - `encode_extrinsics(e_t) -> z_t`
    - `adapt_from_history(history) -> z_hat_t`
    - `act_with_latent(x_t, z) -> action`

- `rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py`
  - config for the final RMA-style branch

- `rma_go2_lab/models/adaptation/ppo_rma_v3_phase1.py`
  - PPO path for `mu + pi`

- `rma_go2_lab/models/adaptation/ppo_rma_v3_phase2.py`
  - frozen `mu + pi`, train `phi`

Optional later:

- `rma_go2_lab/models/adaptation/ppo_rma_v3_joint.py`
  - optional fine-tune path

### Environments

Reuse current envs first:

- `rma_go2_lab/envs/adaptation/rough_history_cfg.py`

If needed later, add:

- `rma_go2_lab/envs/adaptation/rough_history_v3_cfg.py`

Only do that if `V3` needs a genuinely different observation contract.

But if the current env cannot express the faithful actor/history contract
cleanly, then adding a `V3`-specific env config is the correct move.

### Task registration

Add future tasks to:

- `rma_go2_lab/__init__.py`

Planned task names:

- `RMA-Go2-Adapt-V3-Phase1-StageA`

Optional later:

- `RMA-Go2-Adapt-V3-Phase2`
- `RMA-Go2-Adapt-V3-Phase1-StageB`
- `RMA-Go2-Adapt-V3-Joint`

Historical note:
- only the current live task ids in [rma_go2_lab/__init__.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/__init__.py)
  should be treated as launchable
- the optional names above are design-era placeholders, not active registered
  tasks

### Debug / validation

- `scripts/eval/debug_adapt_v3.py`
  - shape checks
  - latent path checks
  - pre/post-switch sanity
  - latent target non-degeneracy

## Phase 1 Spec: Train `mu + pi`

### Objective

Train a privileged latent-conditioned base policy where all privileged
information enters through `z_t`.

### Data path

```text
e_t -> mu -> z_t
x_t, a_{t-1}, z_t -> pi -> a_t
```

### Losses

Use PPO only for the first version.

Allowed light regularizers:

- latent L2 penalty
- latent norm clamp

Do not add:

- reconstruction losses
- contrastive losses
- VAE objectives

on the first pass.

### Acceptance criteria

Phase 1 is good enough to freeze when:

- policy is clearly competent on the privileged task
- `z_t` is non-degenerate
- actor depends on `z_t` in ablations
- no privileged bypass exists in the actor input path

## Phase 2 Spec: Train `phi`

### Objective

Train deployable latent inference against the exact latent used by the base
policy.

### Data path

```text
history_t = (x_{t-k:t}, a_{t-k-1:t-1})
history_t -> phi -> z_hat_t
x_t, a_{t-1}, z_hat_t -> pi -> a_t
target: z_t = mu(e_t)
```

### Frozen modules

Freeze:

- `mu`
- `pi`

Train:

- `phi`

Repo-specific Phase 2 freeze contract:

- initialize from the frozen `Stage A` base:
  - `rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt`
- freeze:
  - actor `pi`
  - extrinsics encoder `mu`
  - Phase 1 auxiliary decoder heads
- keep trainable:
  - adaptation module `phi`
  - critic/value side for PPO fitting

### Losses

Primary:

- latent regression loss

Allowed auxiliary:

- weak action imitation against `pi(x_t, z_t)`

Recommendation:

- start with latent regression only
- add weak action imitation only if optimization is clearly too brittle

### Acceptance criteria

Phase 2 is good enough to freeze when:

- `phi` predicts a non-degenerate `z_hat`
- `V3` adaptation survives and reaches switch scenarios reliably
- performance is competitive with `V1`/`V2`

## Optional Phase 3 Spec: Joint Fine-Tune

This phase is optional.

Only attempt it if Phase 2 is already healthy and we believe end-to-end
adaptation can refine deployment performance.

Train:

- `phi`
- optionally `pi`

Freeze:

- likely `mu`

Do not start here.

## Implementation Order

Use this exact order.

1. add `rma_v3_actor_critic.py`
2. add `adapt_v3_ppo_cfg.py`
3. register Phase 1 task
4. build `debug_adapt_v3.py`
5. smoke test Phase 1 for one iteration
6. run a short Phase 1 validation job
7. freeze Phase 1 checkpoint
8. add Phase 2 PPO path
9. smoke test Phase 2 for one iteration
10. run a short Phase 2 validation job
11. launch full Phase 2 run

## Guardrails

These are non-negotiable for `V3`.

### No actor-side privileged leak

The actor must never directly read:

- `terrain_privileged`
- `dynamics_privileged`

except through `mu(e_t) -> z_t`.

### No history direct-into-actor shortcut

The actor must not take `policy_history` directly.

History must affect the actor only through `phi(history) -> z_hat`.

### No silent repo-specific deviations

If we choose a non-paper-faithful detail for `V3`, it must be written down
explicitly as one of:

- forced by simulator / interface difference
- deliberate repo-native improvement
- temporary implementation compromise

### No retroactive relabeling

Do not call `V3` complete original-style RMA unless:

- `mu` exists explicitly
- `pi` consumes explicit `z`
- `phi` predicts that same `z`

## Validation Checklist

Before any long `V3` run, confirm all of these:

- shapes of `x_t`, `e_t`, `history`, `z_t`, `z_hat_t` are correct
- zeroing `phi` changes `z_hat_t` as expected
- changing privileged factors changes `z_t`
- changing `z_t` changes actor output
- actor input does not contain privileged groups directly
- actor input does not contain `policy_history` directly
- Phase 1 and Phase 2 both survive a one-iteration smoke test

## Success Criteria

`V3` is successful if it gives us all of these:

- a clean original-style RMA contract in this repo
- a trainable teacher-side extrinsics encoder
- a deployable adaptation module that targets the same latent
- competitive adaptation performance against `V1` and `V2`
- a final architecture we can justify carrying into deployment work
- a documented accounting of which details are paper-faithful and which are
  repo-adapted

## Non-Goals For First V3

Do not expand scope by default into:

- vision
- exteroceptive cameras
- recurrent policies
- multi-latent hierarchical decompositions
- body-stability reward redesign
- fancy latent disentanglement tricks

Those can come later.

The first `V3` goal is simple:

- implement the clean RMA contract
- verify it trains
- compare it honestly
