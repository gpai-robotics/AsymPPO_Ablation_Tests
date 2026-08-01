# OG RMA vs This Repo

This note exists to answer one question clearly:

> where are we still following original RMA, where are we different, and why?

The repo already contains detailed design and execution notes. What was missing
was one short, explicit document that separates:

- the original RMA contract
- our faithful interpretation of that contract
- the deliberate departures we made in this repo
- the concrete reasons those departures were necessary

This file should be the first thing to link when someone asks:

- "is this still RMA?"
- "how are we different from the paper?"
- "why didn't we just reproduce the original setup exactly?"

## Short Answer

We are **not** doing a literal reproduction of original RMA.

We are doing:

- an RMA-faithful architecture in the important structural sense
- adapted to a different robot, simulator stack, training regime, and deployment
  pipeline
- with additional repo-specific changes that were forced by observed failure
  modes

The core thing we preserved is:

```text
privileged factors e_t -> encoder mu -> latent z_t
history -> adaptation module phi -> latent z_hat_t
current observation + latent -> base policy pi -> action
```

The biggest things we changed are:

- we currently use a dynamics-only latent rather than terrain-plus-dynamics
- we had to split training into a more staged bootstrap than the paper-level
  summary suggests
- we evaluate heavily against export, MuJoCo, and deployment-contract behavior,
  not only in-sim reward

Those changes were not aesthetic. They were responses to concrete failures in
this repo.

## Original RMA, In Clean Form

At the highest level, original RMA is easiest to think of as a three-part
system:

### 1. Privileged teacher-side world encoding

During training in simulation, the policy has access to hidden extrinsic
information:

- terrain
- friction
- payload / dynamics
- other environment or robot-condition factors

Those hidden factors are compressed into a latent:

```text
e_t -> mu -> z_t
```

The point is not just to use privileged information directly. The point is to
force privileged information through a compact latent bottleneck.

### 2. Latent-conditioned motor policy

The action policy is trained to depend on:

- deployable current observation
- previous action
- extrinsics latent

```text
x_t, a_t-1, z_t -> pi -> a_t
```

This matters because it makes the latent load-bearing. If the actor can ignore
the latent, the architecture may look like RMA on paper while functionally not
being RMA.

### 3. Deployable adaptation module

At deployment, privileged factors are gone.

So the policy replaces direct access to `z_t` with a history-based estimate:

```text
history_t -> phi -> z_hat_t
x_t, a_t-1, z_hat_t -> pi -> a_t
```

That is the essential RMA idea:

- learn a hidden world representation using privileged information in sim
- train the policy to need that representation
- later infer the same representation from recent experience only

## The OG RMA Phase Structure

In clean conceptual form, the original RMA flow is:

### Phase 1: Train the latent-conditioned base policy

Train:

- `mu`
- `pi`

Path:

```text
e_t -> mu -> z_t
x_t, a_t-1, z_t -> pi -> a_t
```

Goal:

- make `z_t` meaningful
- make `pi` actually depend on `z_t`

### Phase 2: Train the adaptation module

Freeze:

- `mu`
- `pi`

Train:

- `phi`

Path:

```text
history_t -> phi -> z_hat_t
```

Supervision target:

```text
z_t = mu(e_t)
```

Goal:

- infer the same latent from recent deployable history

### Phase 3: Deployment

Use only:

```text
history_t -> phi -> z_hat_t
x_t, a_t-1, z_hat_t -> pi -> a_t
```

This is the deployment promise of RMA:

- no privileged factors at runtime
- only recent experience
- still adapt online to hidden conditions

## What We Preserved Faithfully

These are the parts of original RMA that the repo is intentionally preserving.

### 1. Explicit `mu / pi / phi` decomposition

We moved away from bundled history-to-action adaptation and into an explicit:

- `mu`: extrinsics encoder
- `pi`: latent-conditioned base policy
- `phi`: history-to-latent adaptation module

That is the most important structural inheritance from RMA.

### 2. Latent bottleneck contract

We explicitly require privileged factors to influence the actor path through the
latent, not through a side channel.

That is why so much of `Adapt-V3` work focused on:

- latent collapse
- actor bypass
- proving latent usage

### 3. Two-stage teacher-to-student logic

Our active `Adapt-V3` line still follows the same conceptual decomposition:

- privileged phase first
- deployable adaptation phase second

### 4. Deployment through history, not privileged runtime access

Our deployable path still uses:

- current deployable observation
- history
- no privileged runtime extrinsics

That is fully aligned with the original RMA idea.

## Where We Deliberately Diverged

These are the main ways the repo differs from original RMA today.

### Divergence 1: dynamics-only latent first, not full terrain-plus-dynamics

Original-RMA-style ideal:

- a single latent covering the hidden world factors the policy needs

Our current active path:

- start with hidden dynamics only
- defer terrain geometry in the adaptive latent

Current active hidden factors:

- friction
- mass / payload
- motor strength / actuation scaling

Why we diverged:

- the terrain-plus-dynamics `Adapt-V3` line was valuable, but too hard to keep
  cleanly aligned in later closed-loop adaptation stages
- the simpler scientific question was:
  - can the blind history student reliably infer hidden dynamics at all in this
    stack?
- narrowing the latent reduced ambiguity about whether failure came from:
  - adaptation itself
  - terrain inference hardness
  - locomotion collapse under too much hidden-state burden

Interpretation:

- this is a deliberate simplification
- it is not "more faithful" than full RMA
- it is the repo's controlled path toward a survivable version of the contract

### Divergence 2: more staged locomotion bootstrap than the clean paper story

Paper-level RMA descriptions can read like:

- train `mu + pi`
- then train `phi`

In this repo, we needed a more careful bootstrap:

- stationary `Stage A`
- later recovery / switch-aware branches
- critic-only warm start
- temporary imitation scaffold from the blind baseline

Why we diverged:

- when the latent became genuinely load-bearing, from-scratch locomotion
  bootstrap often failed
- the actor would either:
  - ignore the latent
  - collapse the latent
  - fail to form a gait at all

So the extra staging was not just convenience.
It was required to make the architecture viable in this training stack.

### Divergence 3: stronger emphasis on proving latent usage

In this repo, we treat it as a failure if:

- reward looks good
- but the actor can ignore the latent

That led to explicit debugging and acceptance checks around:

- latent variation
- latent dependence
- actor bypass
- adaptation-truth metrics

Why we diverged:

- we saw exactly this failure mode in practice
- without those checks, we could have claimed "RMA success" too early

This is not a departure from RMA's spirit.
It is a repo-specific increase in verification discipline.

### Divergence 4: stronger Sim2Sim and deployment-contract focus

Original RMA is primarily a locomotion-and-adaptation architecture.

This repo adds a much stronger deployment audit layer:

- packaging/export
- source vs exported parity
- MuJoCo Sim2Sim runtime traces
- unclamped vs clamped latent behavior
- Isaac vs MuJoCo contact-pattern comparison

Why we diverged:

- we are not only asking whether the adaptive policy works in Isaac
- we are also asking whether the deployable contract survives another engine
- this became especially important once the adaptive branch showed cross-engine
  latent brittleness

This is a repo-level extension, not part of original RMA itself.

### Divergence 5: continued exploration of blind-reactive alternatives

The repo does not assume explicit latent RMA must be the final winner.

We also preserve a live comparison against:

- strong blind baselines
- terrain-aware blind students
- robustness-first ideas influenced by ETH-style reactive locomotion

Why we diverged:

- the scientific question is not only:
  - "can we build something RMA-like?"
- it is also:
  - "is the explicit latent worth its optimization and transfer cost relative
    to simpler robust deployable policies?"

That comparison pressure is stronger here than in a pure paper reproduction.

## Why These Divergences Were Necessary

The cleanest answer is:

we diverged where literal faithfulness stopped being informative and started
causing ambiguity or failure.

More specifically:

### 1. Different robot and simulator regime

This repo is not the original robot or simulator.

That affects:

- observation interfaces
- contact behavior
- curriculum behavior
- locomotion bootstrap difficulty
- cross-engine transfer behavior

So some implementation choices cannot be copied blindly.

### 2. We hit real functional failure modes

The repo did not merely "prefer" a different design.
We observed concrete failures:

- latent bypass
- latent collapse
- locomotion collapse under true load-bearing latent use
- adaptive latent drift in MuJoCo

Those failures forced the architecture and training recipe to become more
disciplined than a naive paper copy.

### 3. We care about deployable truth, not just structural similarity

A branch does not count as success here just because it has:

- `mu`
- `pi`
- `phi`

It has to also show:

- real adaptation pressure
- preserved locomotion
- exported/runtime consistency
- acceptable behavior in MuJoCo

That criterion naturally pushes the repo toward additional engineering and
evaluation layers beyond the paper abstraction.

### 4. We are separating scientific questions instead of mixing them

The current dynamics-only reboot is a good example.

It is less ambitious than full terrain-plus-dynamics RMA.
But it isolates a cleaner question:

- can a blind history student recover hidden dynamics reliably in this stack
  without behavior collapse?

That separation is useful because it avoids failing on five things at once and
learning nothing from the outcome.

## The Most Important Representation

If someone asks for the cleanest one-paragraph answer, use this:

> We are preserving the core RMA contract, namely `mu(e) -> z`, `phi(history)
> -> z_hat`, and `pi(obs, z)` with privileged training and deployable
> adaptation. We are not doing a literal reproduction because this repo uses a
> different robot, simulator, and deployment pipeline, and because several
> naive paper-faithful attempts failed in practice through latent bypass,
> latent collapse, locomotion bootstrap failure, and MuJoCo latent drift. So
> the repo deliberately simplifies some pieces, stages training more carefully,
> and evaluates much more aggressively at deployment time.

## Current Repo Position

Right now the repo should be described as:

- RMA-faithful in architecture
- repo-specific in training recipe
- narrower than full OG RMA in the active dynamics-only branch
- stricter than OG RMA in deployment and runtime verification

That is the most accurate framing.

It is more honest than saying:

- "this is just RMA"

and more precise than saying:

- "this is not RMA anymore"

The truth is:

- we are carrying the core contract forward
- while deliberately modifying the path around that contract so it survives in
  this repo

## Read Next

For the detailed companion notes, read:

1. `docs/archive/adapt_v3/ADAPT_V3_TRUE_RMA_PLAN.md`
2. `docs/ADAPT_V3_EXECUTION_SPEC.md`
3. `docs/ETH_ANYMAL_GAP_NOTES.md`
4. `docs/ARCHITECTURE_FLOW_FROM_FLAT_TO_ADAPTV3.md`
