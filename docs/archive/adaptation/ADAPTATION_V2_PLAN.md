# Adaptation V2 Plan

Status:

- historical planning note
- preserved for architectural lineage
- canonical current interpretation now lives in:
  - `docs/ADAPTATION_PHASE_SYNTHESIS.md`
  - `docs/V1_V2_CLOSEOUT_CHECKLIST.md`

This note defines `Adapt-V2` as the first explicitly modular, RMA-like
implementation in the repo.

## Why V2 exists

The current progression is:

- `Adapt-V0`
  - adaptation works
  - teacher action imitation
  - architecture still bundled
- `Adapt-V1`
  - explicit latent prediction
  - better scientific structure
  - still implemented as one integrated history-encoder actor-critic

`Adapt-V2` is the next step because we want the **architecture itself** to
match the intended deployment contract:

```text
phi(history) -> z_hat_t
pi(current_obs, z_hat_t) -> action
```

That is the first version in our repo where we can honestly say the code path
is structurally RMA-like rather than merely moving in that direction.

## V2 Contract

### Adaptation module

`phi`

- input:
  - flattened proprio/action history
- output:
  - `z_hat_t`

### Base policy

`pi`

- input:
  - current deployable observation
  - `z_hat_t`
- output:
  - action distribution / action mean

### Teacher target

For the first `V2` scaffold, the teacher-side latent target remains the same as
`V1`:

- frozen teacher actor penultimate feature
- dimension `128`

This keeps the supervision target stable while we change the student
architecture.

## Initial Implementation Choice

The first `V2` scaffold does **not** yet introduce a brand-new optimizer or
training algorithm.

Instead, it reuses the `V1` latent-regression training contract while changing
the student architecture to make the module split explicit.

This is deliberate:

- first isolate architectural change
- then later decide whether training/deployment frequency needs a deeper split

## Current File Layout

`V2` now has separate files:

- policy scaffold:
  - `rma_go2_lab/models/adaptation/modular_actor_critic.py`
- runner config:
  - `rma_go2_lab/models/adaptation/adapt_v2_ppo_cfg.py`

Registered task:

- `RMA-Go2-Adaptation-Student-Rough-History-V2`

## What is explicitly better in V2

Even before asynchronous execution, `V2` makes these things explicit:

- where `phi` lives
- where `pi` lives
- what `z_hat_t` is
- what the deployment-time data flow should be

That makes later real deployment work easier because:

- the adaptation path can be rate-limited independently
- the base policy path can remain fast
- logging can capture both:
  - `z_hat_t`
  - action outputs

## What V2 still does not yet claim

The first scaffold does **not** yet guarantee:

- lower adaptation update rate
- separate deployment process/thread for `phi`
- a new teacher latent design
- better performance than `V1`

It only establishes the correct architectural ownership.

## Success criteria for V2

At minimum:

- trains successfully under the same switched env
- keeps latent supervision active
- matches `V1` wiring integrity

Stronger success:

- matches or exceeds `V1`
- makes deployment decomposition cleaner
- improves interpretability of latent inference

## Recommended next steps

1. keep `V1` training to completion
2. preserve `V0` and `V1` as completed baselines
3. smoke-test `V2` wiring
4. only then decide whether to fully train `V2`
