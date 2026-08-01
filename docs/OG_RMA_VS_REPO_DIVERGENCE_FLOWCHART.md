# OG RMA vs Repo Flowchart

This is the presentation-friendly companion to:

- `docs/OG_RMA_VS_REPO_DIVERGENCE.md`

Use this file when the goal is:

- one-page explanation
- slide-friendly architecture summary
- quick answer to "how are we different from original RMA, and why?"

## Short Framing

```text
Original RMA:
  full privileged extrinsics -> latent z
  latent-conditioned base policy
  history-based adaptation recovers z at deployment

This repo:
  same core mu / pi / phi contract
  but narrower active latent, more staged training, and much stricter
  deployment validation

Why:
  literal faithfulness kept failing here through latent bypass, latent collapse,
  locomotion bootstrap failure, and Sim2Sim latent drift
```

## Clean Flowchart

```text
ORIGINAL RMA
============

Training in simulation
----------------------
hidden extrinsics e_t
  -> mu
  -> latent z_t

current observation x_t
previous action a_t-1
latent z_t
  -> pi
  -> action a_t

Then adaptation training
------------------------
recent deployable history
  -> phi
  -> z_hat_t

target:
  z_hat_t should match z_t = mu(e_t)

Deployment
----------
recent history
  -> phi
  -> z_hat_t

current observation + previous action + z_hat_t
  -> pi
  -> action
```

```text
THIS REPO
=========

What we preserved
-----------------
privileged factors e_t
  -> mu
  -> latent z_t

history
  -> phi
  -> latent z_hat_t

current deployable observation + latent
  -> pi
  -> action

What changed
------------
full terrain+dynamics latent
  -> replaced for the active branch by dynamics-only latent first

simple clean two-phase story
  -> replaced by more staged bootstrap:
     Stage A
     recovery branch
     Sim2Sim refinement branches

paper-style in-sim success criterion
  -> replaced by stricter multi-layer gate:
     Isaac
     export parity
     MuJoCo
     runtime traces

Why
---
paper-faithful structure alone was not enough in this repo because we hit:
  - latent bypass
  - latent collapse
  - locomotion bootstrap failure
  - MuJoCo latent drift
```

## The Most Important Comparison

```text
Original RMA asks:
  can a latent learned from privileged information be recovered from history?

This repo asks:
  can that still work on this robot, in this simulator stack, with this
  deployment path, without the latent collapsing or transfer breaking?
```

## Faithful vs Different

### Faithful

- explicit `mu / pi / phi` decomposition
- latent bottleneck between privileged information and actor
- history-based deployable adaptation
- teacher-to-student latent recovery logic

### Different

- active branch is dynamics-only first, not full terrain-plus-dynamics
- training is staged more aggressively than a clean paper diagram suggests
- deployment validation is much stricter and includes MuJoCo
- blind-reactive alternatives remain active comparison branches

## Why The Differences Exist

```text
If we had copied the paper structure literally, the repo repeatedly produced:
  structurally correct boxes
  but functionally weak adaptation

So we changed the path, not the core contract.
```

## One-Paragraph Version

> We are keeping the core original-RMA contract, namely `mu(e) -> z`,
> `phi(history) -> z_hat`, and `pi(obs, z)` with privileged training and blind
> deployable adaptation. We are not doing a literal reproduction because the
> repo uses a different robot, simulator, and deployment path, and because
> naive paper-faithful versions failed here through latent bypass, latent
> collapse, locomotion bootstrap failure, and MuJoCo latent drift. So the repo
> narrows the active latent, stages training more carefully, and validates much
> more aggressively at deployment time.

## Read Next

- `docs/OG_RMA_VS_REPO_DIVERGENCE.md`
- `docs/archive/adapt_v3/ADAPT_V3_TRUE_RMA_PLAN.md`
- `docs/ADAPT_V3_EXECUTION_SPEC.md`
- `docs/ARCHITECTURE_FLOW_FROM_FLAT_TO_ADAPTV3.md`
