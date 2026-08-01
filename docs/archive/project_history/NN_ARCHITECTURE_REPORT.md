# NN Architecture Report

This note is the architecture reference for the main policy lineage in this repo, from the flat locomotion prior through the blind baselines, teacher line, C1 line, and the adaptation/C2 line up to the latest structured `Z27` run.

It is written to answer questions like:

- "What network does your policy use?"
- "How many parameters does it have?"
- "What is actually deployed at inference time?"
- "What weights are warm-started versus newly learned?"

## One Important Distinction

When people ask about "the PPO model", there are really two different things they may mean:

1. **Full train-time actor-critic**
   - actor
   - critic
   - history encoder / latent encoder
   - auxiliary heads
   - action-noise parameters

2. **Deploy-time policy path**
   - only the modules needed to produce the action mean at inference

For this project, those are often different.

- C1 deploys only the history encoder plus actor.
- C2 Phase 2 deploys only `phi(history)` plus `pi(x, z)`.
- Critics and most auxiliary heads are training-only.

## Shared Building Blocks

Across almost every family in this repo:

- action space is **12-D** joint-position actions
- current deployable proprio observation is **48-D**
- actor trunk is usually:
  - `512 -> 256 -> 128 -> 12`
- critic trunk is usually:
  - `512 -> 256 -> 128 -> 1`
- activation is usually `ELU`
- action noise is a learned **12-D std vector**

The main changes across generations are:

- whether a terrain / dynamics latent is used
- whether history is encoded by MLP or temporal Conv1D
- whether the latent is unconstrained or structured

## Observation Dimensions

These are the dimensions that matter most for parameter counts.

### Shared current policy observation

- current deployable observation: **48**

This is the standard proprioceptive policy group used by the blind baseline and carried into C1/C2.

### History dimensions

- blind rough history: **20** steps
  - flattened history = `20 * 48 = 960`
- C1 ETH-like history: **100** steps
  - flattened history = `100 * 48 = 4800`

### Privileged dimensions

- dynamics privilege: **27**
  - `1` static friction
  - `1` dynamic friction
  - `1` base-mass ratio
  - `12` joint stiffness scales
  - `12` joint damping scales

- terrain privilege:
  - teacher terrain scan comes from a `1.6 x 1.0` grid with `0.1` resolution
  - this implies **160** scan values

Note:
- the `160` terrain-scan size is derived from the sensor grid geometry in config, not from a live instantiated log line

## Parameter Counting Convention

For a linear layer:

- `params = in_dim * out_dim + out_dim`

All counts below include biases.

## Lineage Summary Table

| Model Family | Core Idea | Full Train-Time Params | Deploy-Time Params |
|---|---|---:|---:|
| Flat expert / blind warm-start trunk | plain MLP actor-critic | 380,313 | 190,872 |
| Teacher V0 / V1 | terrain encoder + MLP actor/critic | 444,025 | 238,200 |
| Teacher V2 | compressed terrain latent | 401,153 | 207,616 |
| Teacher V3 | compressed terrain latent + raw dynamics privilege | 428,801 | 221,440 |
| Teacher V4 | V3 plus terrain-target head | 430,222 | 221,440 |
| Adaptation V0 | history MLP -> 8-D latent | 668,449 | 474,912 |
| Adaptation V1 | history MLP -> 128-D latent | 806,809 | 551,832 |
| Adaptation V2 | modular `phi + pi`, still 128-D latent | 806,809 | 551,832 |
| C1 ETH-like V1/V2/V3 | temporal Conv1D history encoder | 500,569 | 253,528 |
| Adapt-V3 dyn-only 32-D | `mu / phi / pi` with 32-D latent | 713,908 | 490,296 |
| Adapt-V3 structured `Z27` | direct 27-D dynamics-shaped latent | 691,868 | 487,091 |

Notes:

- "Deploy-time params" means the action-producing path only.
- For privileged teachers and C2 Phase 1 roots, deploy-time means the policy path used in sim with privileged inputs.
- For C2 Phase 2, deploy-time means the history-based student path.

## 1. Flat Expert / Blind Warm-Start Trunk

Config:
- [flat_prior_runner_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/priors/flat_prior_runner_cfg.py)

Env:
- [flat_forward_prior_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/priors/flat_forward_prior_cfg.py)

This is the original locomotion prior.

### Architecture

- actor input: `48`
- actor MLP:
  - `48 -> 512 -> 256 -> 128 -> 12`
- critic input: `48`
- critic MLP:
  - `48 -> 512 -> 256 -> 128 -> 1`
- learned action std:
  - `12`

### Parameter count

- actor: `190,860`
- critic: `189,441`
- action std: `12`

**Full train-time params = 380,313**

Deploy-time:
- actor + std only

**Deploy-time params = 190,872**

### What to say out loud

"The flat prior is just a plain proprioceptive MLP actor-critic. It takes a 48-D observation and uses a 3-layer 512-256-128 trunk."

## 2. Blind Rough Baselines

Configs:
- [blind_rough_runner_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/blind/blind_rough_runner_cfg.py)

Classes:
- [actor_critic.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/blind/actor_critic.py)

The warm-start blind baselines keep the same actor-critic architecture as the flat prior.

Difference from flat prior:

- same MLP size
- different environment
- often actor warm-started from `flat1499.pt`
- rough-terrain training instead of flat terrain

So architecturally:

- **same network family as flat expert**
- same parameter count:
  - **380,313 full**
  - **190,872 deploy**

## 3. Teacher Line

Main class:
- [teacher/actor_critic.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/teacher/actor_critic.py)

The teacher family adds a dedicated terrain encoder and, from V3 onward, raw dynamics privilege.

### 3.1 Teacher V0 / V1

Configs:
- [ppo_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/teacher/ppo_cfg.py)
- [ppo_v1_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/teacher/ppo_v1_cfg.py)

Privileged input:
- terrain scan: `160`

Terrain encoder:
- `160 -> 128 -> 64 -> 32`

Actor input:
- current proprio `48`
- terrain latent `32`
- total = `80`

Actor MLP:
- `80 -> 512 -> 256 -> 128 -> 12`

Critic MLP:
- `80 -> 512 -> 256 -> 128 -> 1`

Parameter count:

- terrain encoder: `30,944`
- actor: `207,244`
- critic: `205,825`
- std: `12`

**Full params = 444,025**

Deploy-time privileged action path:
- terrain encoder + actor + std

**Deploy-time params = 238,200**

### 3.2 Teacher V2

Config:
- [ppo_v2_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/teacher/ppo_v2_cfg.py)

Change from V0/V1:

- compress terrain privilege much harder

Terrain encoder:
- `160 -> 64 -> 32 -> 8`

Actor input:
- `48 + 8 = 56`

Actor MLP:
- `56 -> 512 -> 256 -> 128 -> 12`

Critic MLP:
- `56 -> 512 -> 256 -> 128 -> 1`

Parameter count:

- terrain encoder: `12,648`
- actor: `194,956`
- critic: `193,537`

Exact totals:

- **Full params = 401,153**
- **Deploy-time params = 207,616**

### 3.3 Teacher V3

Config:
- [ppo_v3_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/teacher/ppo_v3_cfg.py)

Env privilege split:
- [rough_v3_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/teacher/rough_v3_cfg.py)

Important change:

- terrain privilege still goes through an encoder
- dynamics privilege is passed in raw

Terrain encoder:
- `160 -> 64 -> 32 -> 8`

Raw dynamics privilege:
- `27`

Actor input:
- current proprio `48`
- raw dynamics `27`
- terrain latent `8`
- total = `83`

Actor MLP:
- `83 -> 512 -> 256 -> 128 -> 12`

Critic MLP:
- `83 -> 512 -> 256 -> 128 -> 1`

Exact totals:

- **Full params = 428,801**
- **Deploy-time params = 221,440**

### 3.4 Teacher V4

Config:
- [ppo_v4_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/teacher/ppo_v4_cfg.py)

Same policy path as V3, but adds a terrain-target auxiliary head.

Auxiliary head:
- `8 -> 64 -> 13`

So:

- policy path is unchanged from V3
- train-time model gets slightly larger

Exact totals:

- **Full params = 430,222**
- **Deploy-time params = 221,440**

### What to say out loud

"The teacher family is an actor-critic with a dedicated terrain encoder. V0/V1 use a 32-D terrain latent, V2 compresses that to 8-D, and V3/V4 add raw 27-D dynamics privilege alongside the terrain latent."

## 4. Early Adaptation Student Line

These are the earlier history-student generations before Adapt-V3.

### 4.1 Adaptation V0

Config:
- [adapt_ppo_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/adaptation/adapt_ppo_cfg.py)

Class:
- [actor_critic.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/adaptation/actor_critic.py)

History:
- `20 * 48 = 960`

History encoder:
- `960 -> 256 -> 128 -> 8`

Actor input:
- `48 + 8 = 56`

Actor MLP:
- `56 -> 512 -> 256 -> 128 -> 12`

Critic MLP:
- `56 -> 512 -> 256 -> 128 -> 1`

Exact totals:

- **Full params = 668,449**
- **Deploy-time params = 474,912**

### 4.2 Adaptation V1

Config:
- [adapt_v1_ppo_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/adaptation/adapt_v1_ppo_cfg.py)

Same idea as V0, but latent becomes much larger.

History encoder:
- `960 -> 256 -> 128 -> 128`

Actor input:
- `48 + 128 = 176`

Actor MLP:
- `176 -> 512 -> 256 -> 128 -> 12`

Critic MLP:
- `176 -> 512 -> 256 -> 128 -> 1`

Exact totals:

- **Full params = 806,809**
- **Deploy-time params = 551,832**

### 4.3 Adaptation V2

Config:
- [adapt_v2_ppo_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/adaptation/adapt_v2_ppo_cfg.py)

Class:
- [modular_actor_critic.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/adaptation/modular_actor_critic.py)

V2 keeps about the same size as V1 but makes the split explicit:

- `phi(history) -> z_hat`
- `pi(current, z_hat) -> action`

Architecturally:

- same latent size: `128`
- same actor input: `176`
- same total parameter count as V1

Exact totals:

- **Full params = 806,809**
- **Deploy-time params = 551,832**

### What to say out loud

"The early adaptation students were history-to-latent actor-critics. V0 used a small 8-D history latent, while V1/V2 jumped to 128-D, which made the model much larger and more expressive."

## 5. C1 ETH-Like Line

Active config:
- [blind_rough_runner_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/blind/blind_rough_runner_cfg.py#L191)

Active class:
- [history_actor_critic.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/blind/history_actor_critic.py)

Env:
- [c1_blind_rough_teacher_history_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/c1_blind_rough_teacher_history_cfg.py)

This is your blind history student with a temporal encoder.

### C1 architecture

History:
- length = `100`
- flattened = `4800`

Temporal encoder:
- `Conv1d(48 -> 64, kernel=3)`
- `ELU`
- `Conv1d(64 -> 64, kernel=3, dilation=2)`
- `ELU`

History projection:
- concatenate:
  - latest temporal feature
  - mean pooled temporal feature
- `Linear(128 -> 64)`
- `ELU`

History target head:
- `64 -> 128 -> 128`

Actor input:
- current proprio `48`
- history feature `64`
- total = `112`

Actor MLP:
- `112 -> 512 -> 256 -> 128 -> 12`

Critic MLP:
- `112 -> 512 -> 256 -> 128 -> 1`

### C1 exact totals

- **Full train-time params = 500,569**
- **Deploy-time params = 253,528**

Deploy-time includes:

- temporal encoder
- history projection
- actor
- action std

The V1/V2/V3 C1 variants all use the same network architecture.

What changes across them is mainly:

- the frozen teacher checkpoint used for supervision
- the latent/imitation coefficients

## 6. Adapt-V3 Family

Main class:
- [rma_v3_actor_critic.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/adaptation/rma_v3_actor_critic.py)

Core decomposition:

- `mu`: privileged encoder
- `phi`: deployable history encoder
- `pi`: control policy

This is the clearest RMA-style family in the repo.

### 6.1 Adapt-V3 dyn-only 32-D

Configs:
- [adapt_v3_ppo_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py)

History:
- `20 * 48 = 960`

Privilege:
- dynamics only = `27`

`mu`:
- `27 -> 128 -> 64 -> 32`

Dynamics decoder:
- `32 -> 64 -> 27`

`phi`:
- `960 -> 256 -> 128 -> 32`

Actor input:
- `48 + 32 = 80`

Actor MLP:
- `80 -> 512 -> 256 -> 128 -> 12`

Critic MLP:
- `80 -> 512 -> 256 -> 128 -> 1`

Exact totals:

- **Full train-time params = 713,908**

Deploy-time:

- privileged Phase 1 action path:
  - `mu + actor + std`
- student Phase 2 action path:
  - `phi + actor + std`

Student deploy-time path is the more relevant one operationally:

- **Deploy-time student path = 490,296**

### 6.2 Adapt-V3 terrain-lite 32-D

This is the archived terrain-inclusive branch.

Key difference:

- `mu` input becomes terrain-lite `13` + dynamics `27` = `40`

Approximate architecture:

- `mu: 40 -> 128 -> 64 -> 32`
- terrain summary decoder:
  - `32 -> 32 -> 4`
- dynamics decoder:
  - `32 -> 64 -> 27`
- `phi: 960 -> 256 -> 128 -> 32`
- actor input remains `48 + 32 = 80`

This branch was retired because of the crouch / posture issue, but the architecture is worth knowing because it explains the terrain-aware attempt.

### 6.3 Adapt-V3 structured `Z27`

Active rebuild config:
- [adapt_v3_ppo_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py#L543)

Frozen Phase 1 root:
- [adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.pt](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.pt)

This is the newest and currently most important C2 root.

The design goal is:

- make latent `z` directly match the 27-D hidden dynamics contract
- avoid the looser 32-D free-form latent geometry

### Structured `Z27` architecture

`mu`:
- `Linear(27 -> 27)`
- identity initialized

Dynamics decoder:
- `Linear(27 -> 27)`
- identity initialized

`phi`:
- `960 -> 256 -> 128 -> 27`

Actor input:
- `48 + 27 = 75`

Actor MLP:
- `75 -> 512 -> 256 -> 128 -> 12`

Critic MLP:
- `75 -> 512 -> 256 -> 128 -> 1`

### Structured `Z27` exact totals

- **Full train-time params = 691,868**

Phase 1 privileged deploy path:
- `mu + actor + std`
- **205,452**

Phase 2 deployable student path:
- `phi + actor + std`
- **487,091**

### Why this matters

This branch is the first one where:

- locomotion stayed strong
- privileged latent stayed alive
- dynamics prediction converged cleanly

So if someone asks what your latest C2 model is, this is the one to describe.

## Warm-Start and Initialization Story

This is the other thing people often mean when they ask about "weights."

### Flat / blind backbone

- flat expert is trained from scratch on flat terrain
- blind rough baselines warm-start from the flat expert

### Teacher line

- teacher models warm-start actor/critic from the blind baseline
- terrain encoder is newly initialized

### C1 line

- actor warm-starts from flat expert
- temporal history pathway is small non-zero initialized
- teacher imitation / target regression are added during PPO

### Early adaptation V0/V1/V2

- actor warm-starts from blind baseline
- history latent path is newly initialized

### Adapt-V3 dyn-only 32-D

- actor/critic trunk warm-starts from blind baseline
- `mu` and `phi` are learned around that trunk

### Adapt-V3 structured `Z27`

- actor/critic warm-start from blind baseline
- `mu` is explicitly identity initialized
- dynamics decoder is identity initialized
- `phi` starts small Xavier

That identity-initialized structured root is one of the key reasons the latest line worked better than the earlier collapsed-latent attempts.

## PPO / Training Weights by Family

These are not network weights. These are training coefficients.

### Flat expert

- learning rate: `1e-3`
- entropy coef: `0.01`

### Blind rough / C1 / most adaptation lines

Usually:

- learning rate: `1e-4`
- clip param: `0.2`
- entropy coef: `0.002`
- PPO epochs: `5`
- mini-batches: `4`
- gamma: `0.99`
- lambda: `0.95`

### C1 teacher-supervised extras

- imitation losses
- history target regression

### Adapt-V3 Phase 1 extras

- latent anchor
- dynamics prediction
- latent variation
- pairwise latent shaping
- optional flat-expert imitation

### Adapt-V3 Phase 2 extras

- latent regression to frozen Phase 1 root
- optional teacher imitation

## Best Short Answers

If someone asks:

### "How many parameters does your policy have?"

Good answer:

- flat / blind backbone:
  - "about **380k** full actor-critic params"
- C1:
  - "about **500k** full train-time params, about **254k** in the deploy-time action path"
- latest C2 structured `Z27`:
  - "about **692k** full train-time params, about **487k** in the deployable history-to-action path"

### "What architecture are you using?"

Good answer:

- C1:
  - "a temporal Conv1D history encoder feeding a 3-layer MLP actor-critic"
- latest C2:
  - "an RMA-style `mu / phi / pi` decomposition with a structured 27-D latent aligned to hidden dynamics"

### "What weights are your networks using?"

Good answer:

- "The actor/critic trunks are mostly 512-256-128 ELU MLPs. Depending on the line, they are warm-started either from the flat prior or the blind rough baseline, while the history or privileged latent modules are initialized separately and trained with PPO plus auxiliary losses."

## Final Practical Rule

When you talk about a model externally, always decide which of these you mean:

1. **full train-time actor-critic size**
2. **deploy-time inference path size**
3. **architecture family**
4. **training loss weights**

If you answer with the wrong category, it sounds vague. If you answer with the right one, you sound very clear.
