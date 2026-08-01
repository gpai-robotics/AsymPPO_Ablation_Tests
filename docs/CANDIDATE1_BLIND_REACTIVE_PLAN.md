# Candidate 1 Blind Reactive Plan

Historical note:
- this file is no longer the canonical Candidate 1 definition
- the single source of truth for Candidate 1 is now
  [C1_STAGEA_MODEL400_DEPLOY_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_STAGEA_MODEL400_DEPLOY_CARD.md)

Use this file for:

- design lineage
- rejected alternatives
- historical branch reasoning

This note defines the concrete forward path for Candidate 1.

Candidate 1 is **not** "any good blind policy."
It is specifically:

- a blind reactive deployable policy
- trained with privileged teacher-student structure
- trained in an environment that exposes terrain + dynamics privileged groups
  to the teacher during training
- with no privileged inputs required at deployment

Teacher-side qualification:

- the frozen `Teacher V3` checkpoint is architecturally given both
  `terrain_privileged` and `dynamics_privileged`
- but current dependency audits show the frozen checkpoint is materially using
  `dynamics_privileged` and effectively ignoring `terrain_privileged` on the
  tested forward probes
- so C1 should not be described as inheriting a teacher that is already
  validated as using both privileged sources equally well

## Current Frozen Candidate

This file still contains important historical V1 notes, but the active frozen
Candidate 1 artifact has moved forward. Read the section below as the current
truth and treat older V1-specific details later in the file as historical
context rather than the live winner.

The current active C1 finalist is:

- `rma_go2_lab/policies/exported/c1_ethlike_v3_model_400_candidate/`

Source checkpoint:

- `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_c1_ethlike_v3_v4teacher300/2026-05-11_13-10-12/model_400.pt`

Why this is the current C1 winner:

- inherited the repaired `Teacher V4 model_300` root
- history is behaviorally load-bearing on the canonical Isaac switch/push suites
- exported cleanly as a deployable `blind_history_policy`
- reached exact source/export parity after export-wrapper fixes
- completed Isaac deployment rehearsal
- completed MuJoCo nominal, hidden-env, moderate-disturbance, and
  continuous-corridor evaluation

Current C1 truth:

- `terrain-lite` remains the important historical bridge artifact
- `C1-ETHLike-V1 model_700` remains an important historical finalist
- `C1-ETHLike-V3 StageA model_400` is now the active blind-reactive finalist

Important qualification:

- the earlier V1 frozen finalist was exported and evaluated as a
  `blind_history_policy`
- but later ablation work showed that its history pathway was not meaningfully
  load-bearing in the originally frozen training line
- so it should be described as a strong blind deployable baseline, not as
  already-proven temporal inference success

Current qualification:

- the active StageA `model_400` candidate does clear the intended history-use
  bar on Isaac switch/push evaluation
- MuJoCo moderate-disturbance ablations also show a clean
  `normal > frozen > zero` pattern
- MuJoCo continuous-corridor ablations are mixed rather than uniformly
  monotonic
- its current main remaining deployment weakness is lateral push recovery under
  continuous corridor geometry in MuJoCo

See also:

- `docs/C1_STAGEA_MODEL400_DEPLOY_CARD.md`

## Candidate 1 Identity

The intended final contract is:

```text
training:
  privileged teacher is given terrain + dynamics privileged groups
  blind student learns from teacher/reference supervision

deployment:
  deployable observation
    -> blind reactive policy
    -> action
```

This means Candidate 1 should stay:

- blind at inference
- robustness-first
- simple at runtime

It should **not** carry the explicit RMA adaptation burden.

Current upstream truth:

- the frozen V3 teacher is a valid historical dynamics-privileged supervision
  source
- the active canonical upstream teacher is now `V4 model_300`
- `V4 model_300` is the current best overall teacher candidate that is
  validated as using both terrain and dynamics privilege on its audited
  general rough-terrain probes
- stair-specialized teacher evidence exists in archived `V4.1`, but it is not
  the canonical upstream teacher because the project contract is overall rough
  robustness rather than terrain-family specialization

## What Counts As The Real Base

If we interpret Candidate 1 strictly, the closest current repo base is:

- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase2_stage_a_final.pt`

Why:

- it is a blind student at inference
- its teacher/reference latent includes both:
  - `terrain_lite_privileged`
  - `dynamics_privileged`
- it is the current repo artifact that most closely matches:
  privileged teacher -> blind reactive student

Important distinction:

- `blind_baseline2_warmstart_final.pt`
- `blind_baseline3_warmstart_imitation_final.pt`

are still extremely important, but they are better treated as:

- comparison anchors
- deployment simplicity anchors
- behavior-quality references

They are not the cleanest match to the full Candidate 1 definition because
they do not represent the terrain+dynamics privileged teacher-student story we
want to preserve.

## Current Candidate 1 Reality

Current strongest terrain+dynamics-privileged blind-student artifact:

- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase2_stage_a_final.pt`

Current known strengths:

- lower velocity and yaw error than dyn-only in many suite families
- real proof that compact terrain privilege can survive into a blind student
- blind at inference

Current known weakness:

- more posture/composure failures than the dyn-only winner
- more crouched, lower-base, more failure-prone forward behavior
- not yet the current deployment-side winner

So Candidate 1 became **real** through the terrain-lite bridge, but the active
finalist has now moved to the cleaner blind-history successor line.

## Current C1 Successor Direction

The current bridge artifact is still useful, but the intended final-form C1 has
shifted closer to the ETH-style blind reactive line.

### Why the shift happened

The bridge candidate taught us something important:

- a terrain+dynamics privileged teacher can produce a blind student
- but the `terrain-lite` bridge still carries too much `Adapt-V3` ancestry
- and visually it remained posture-fragile

So the cleaner C1 target is now:

- direct blind reactive deployment policy
- recent proprioceptive history as the deployable temporal signal
- privileged teacher support during training only
- no explicit adaptive latent burden at deployment

### What stays

- rough-terrain env family
- reward family unless evidence forces a specific change
- domain-randomization plumbing
- gait screen, isolated suite, OOD suite
- clip recording, export bundles, MuJoCo preflight/runtime bridge

### What changes

- student becomes direct history-conditioned blind control
- C1 is judged by robustness, posture/composure, Sim2Sim, and deployment
- C1 no longer carries the burden of proving explicit online adaptation

### What gets dropped from C1 expectations

- deploy-time latent interpretation
- `phi(history)`-style adaptation claims
- adaptation metrics as the main success criteria

## Current Executable C1 Branch

The first serious successor branch is now:

- `C1-ETHLike-V1`

Working task:

- `RMA-Go2-C1-ETHLike-V1-StageA`

### V1 deployment contract

```text
policy observation
recent history
  -> blind reactive policy
  -> action
```

No privileged inputs at deployment.

### V1 training contract

```text
privileged teacher
  -> terrain privilege
  -> dynamics privilege

blind student
  -> policy
  -> policy_history
  -> action

training signal
  -> PPO reward
  -> teacher imitation during bootstrap
```

### V1 architecture

- temporal encoder over `policy_history`
- compact history feature
- actor MLP over `policy + history_feature`

### V1 current implementation status

Current status:

- implemented
- compiles
- task registration resolves
- training smoke cleared interface bugs
- training cleared into real behavior validation
- `model_700` is now the selected checkpoint
- deploy bundle export is complete
- Isaac runtime trace is complete
- MuJoCo runtime rehearsal is complete

## What C1 Must Now Prove

The reference-repo sweep and the later history debugging both sharpened the
standard for this candidate.

C1 should not be considered a successful history-conditioned blind controller
merely because:

- `policy_history` exists in the architecture
- the export bundle includes a history dimension
- MuJoCo and Isaac runners can feed a history tensor

Instead, a future C1 history-bearing line should prove three separate things:

1. the temporal path is alive
   - `normal`, `zero`, and `frozen` history ablations produce materially
     different actions
2. the temporal path is useful
   - `normal` meaningfully outperforms `zero` or `frozen` on trusted
     switch/disturbance probes
3. the temporal path survives deployment semantics honestly
   - history handling in Isaac, MuJoCo, and hardware is consistent with the
     export contract

This standard is directly motivated by the reference repos:

- `walk-these-ways-go2` supervises the adaptation/history path explicitly
- `unitree_rl_gym` keeps temporal policies explicit as recurrent contracts
- `unitree_rl_lab` makes deploy-time observation/history semantics explicit in
  config

So the repo should no longer accept "history is present" as evidence that
history matters.

## C1 V2 Corrective Retrain

Archived note:
- `RMA-Go2-C1-ETHLike-V2-StageA` was an intermediate corrective branch.
- It is kept here for historical context only and is no longer part of the
  live task registry.

The next active C1 training adjustment is now a clean successor task:

- `RMA-Go2-C1-ETHLike-V2-StageA`

This exists because the V1 history line reached an unsatisfying middle state:

- the history path is now alive after the warm-start fix
- but later checkpoints still do not make live-updating history consistently
  better than `zero` or `frozen` ablations

So V2 adds an explicit supervision signal for the temporal path itself.

### V2 idea

Keep the same basic deploy contract:

- `policy`
- `policy_history`
- action

But during training, add a second teacher-driven loss:

- student history encoder predicts a teacher-aligned internal target
- frozen V3 teacher exports its penultimate actor feature
- PPO now optimizes:
  - reward
  - weak action imitation
  - explicit history-target regression

### V2 implementation

- task:
  - `RMA-Go2-C1-ETHLike-V2-StageA`
- runner:
  - `rma_go2_lab.models.blind.variants_ppo_cfg:Go2C1EthLikeV2PPORunnerCfg`
- policy:
  - `TemporalBlindActorCritic`
  - adds `history_target_head(history_feature) -> 128-dim teacher target`
- algorithm:
  - `BlindPPOWithV3Teacher`
  - now supports:
    - `history_target_regression`
    - `history_target_active_frac`
    - `history_target_cosine`

### V2 intent

This is the first C1 line that is trying to make the temporal path useful on
purpose, instead of relying on action imitation alone to somehow force that
behavior to emerge.

### V1 training recipe summary

The frozen C1 finalist should currently be understood as having been trained
with this contract:

- task:
  - `RMA-Go2-C1-ETHLike-V1-StageA`
- student environment:
  - blind rough terrain
  - deployable `policy_history_length = 100`
- student runtime inputs:
  - `policy`
  - `policy_history`
- teacher-only privileged groups during training:
  - `terrain_privileged`
  - `dynamics_privileged`
- teacher source:
  - frozen V3 privileged teacher
- optimization:
  - PPO reward optimization
  - plus temporary teacher imitation on action mean

Current teacher-imitation schedule in code:

- stage 0:
  - coefficient `0.20`
  - active before update `300`
- stage 1:
  - coefficient `0.05`
  - active from update `300` to `799`
- after update `800`:
  - coefficient `0.00`

Imitation is also command-gated:

- imitation is applied only when command magnitude is above `0.1`

So the intended training story is:

- early privileged teacher guidance for moving commands
- then reduced teacher influence
- then pure PPO consolidation with the blind history student alone

## Current C1 Training Lesson

The first frozen C1 line surfaced a real repo lesson:

- adding a new temporal branch on top of a warm-started blind policy can leave
  that branch effectively decorative if initialization is careless

That bug has now been fixed in the model code, but the process lesson should
stay attached to C1:

- any future C1 retrain that claims temporal benefit should be accompanied by
  explicit history ablation checks during training, not only after final
  freezing

In practice, the next trustworthy C1 temporal line should include:

1. early checkpoint ablation
2. switch/push recovery ablation
3. a decision note explaining whether history is:
   - unnecessary
   - mildly useful
   - or genuinely important

### V1 training provenance and reproducibility

The frozen C1 finalist is not just a conceptual recipe. Its concrete training
lineage is:

- task registration:
  - `RMA-Go2-C1-ETHLike-V1-StageA`
  - registered in `rma_go2_lab/__init__.py`
- environment config entry point:
  - `rma_go2_lab.envs.blind.c1_ethlike_v1_cfg:Go2C1EthLikeV1EnvCfg`
- history/env base:
  - `rma_go2_lab.envs.blind.rough_history_cfg:Go2BlindBaselineHistoryRoughEnvCfg`
- runner config entry point:
  - `rma_go2_lab.models.blind.variants_ppo_cfg:Go2C1EthLikeV1PPORunnerCfg`
- policy class:
  - `TemporalBlindActorCritic`
- PPO / teacher-imitation algorithm class:
  - `BlindPPOWithV3Teacher`
  - implemented in `rma_go2_lab/models/blind/ppo_with_v3_teacher.py`

Important frozen training sources:

- actor warm-start prior:
  - `rma_go2_lab/policies/flat1499.pt`
- teacher checkpoint:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v3/2026-04-21_15-35-03/model_1999.pt`
- runner experiment name:
  - `go2_c1_ethlike_v1`
- selected winning run directory:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_c1_ethlike_v1/2026-05-07_10-51-33/`
- selected winning checkpoint:
  - `model_700.pt`

Key runner settings from the registered config:

- `num_steps_per_env = 32`
- `max_iterations = 2000`
- `save_interval = 20`
- actor / critic hidden dims:
  - `[512, 256, 128]`
- history encoder:
  - channels `[64, 64]`
  - kernel size `3`
  - history feature dim `64`
- PPO core:
  - `learning_rate = 1e-4`
  - `num_learning_epochs = 5`
  - `num_mini_batches = 4`
  - `entropy_coef = 0.002`
  - `gamma = 0.99`
  - `lam = 0.95`

Canonical launch form for reproducing this branch:

```bash
env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/tools/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RMA-Go2-C1-ETHLike-V1-StageA
```

So if someone asks how the frozen C1 finalist was actually trained, the short
answer is:

- registered task `RMA-Go2-C1-ETHLike-V1-StageA`
- env config `Go2C1EthLikeV1EnvCfg`
- runner config `Go2C1EthLikeV1PPORunnerCfg`
- warm-start from `flat1499.pt`
- imitate frozen V3 teacher early, then anneal imitation away
- winner selected from run `2026-05-07_10-51-33`, checkpoint `model_700.pt`

### V1 intentional simplifications

These are deliberate V1 simplifications, not accidents:

- shorter and cleaner repo-native observation design than the ETH supplementary
  code
- no explicit motion-prior redesign yet
- no C1-native teacher yet
- no actuator-model redesign yet

### V1 non-negotiable guarantees

These are the checks that keep V1 honest as a blind deployable candidate:

- student resolved obs groups must stay:
  - `policy`
  - `policy_history`
- student actor input must not include privileged dims
- privileged groups may exist only for teacher-side supervision during training

## Teacher Structure For C1

The current C1 teacher is appropriate for V1, but it is important to be
precise about what it is.

Current teacher source:

- frozen V3 privileged teacher

Teacher structure:

- blind B2-derived locomotion prior
- encoded terrain privilege
- raw dynamics privilege
- anti-crouch reward shaping inherited from the teacher line

What this means:

- good enough for V1 teacher-student training
- not necessarily the final ideal C1 teacher
- if C1 later inherits the wrong style, a C1-native teacher is a plausible
  refinement

## Reference Alignment

Relative to the local ETH supplementary code, current `C1-ETHLike-V1` now
matches the right **family** of design:

- direct blind history-conditioned control
- privileged teacher during training
- no explicit runtime latent burden

Still intentionally simpler than ETH in:

- history richness and observation payload
- teacher architecture specificity
- actuator-side realism

That is acceptable for V1 as long as we remember it is:

- structurally aligned
- not yet a faithful endpoint

## Current Evaluation Outcome

Selected checkpoint:

- `model_700.pt`

Evaluation outcome:

- best overall C1 checkpoint on gait + blind suite + OOD switch suite
- cleaner standstill/composure than `model_400`
- better overall blind and OOD robustness than `model_1200`
- canonical MuJoCo limit rerun completed with:
  - `23 / 23` scenarios
  - `5` rollouts each
  - `moderate` reset diversity preset

Canonical MuJoCo findings:

- strongest MuJoCo cases:
  - `flat_command_step_up`
  - `locomotion_scene_nominal`
  - `low_friction_flat`
- hidden-env robustness on flat is strong:
  - low friction
  - heavy payload
  - high friction
- hardest cases remain:
  - `flat_lateral_push`
  - `continuous_corridor_lateral_push`
  - `continuous_corridor_low_friction`
  - `hidden_triple_combo_continuous_corridor`

Sim2Sim outcome:

- Isaac runtime trace completed successfully
- MuJoCo runtime rehearsal completed successfully
- transfer is promising rather than perfect:
  - MuJoCo forward tracking is worse than Isaac
  - posture and tilt remain close
  - no latent instability exists because C1 has no deploy-time adaptive latent

Current interpretation:

- this is a valid deployable blind-reactive finalist
- it is strong enough to serve as the fixed no-adaptation reference while C2
  continues
- the final canonical MuJoCo story is now clear:
  - C1 is good at persistent forward locomotion and command reactivity
  - C1 handles hidden environment mismatch better than lateral shove recovery
  - lateral disturbance recovery is the main deployment-side weakness to carry
    into C1 vs C2 comparison

## ETH Training Methodology Gap Map

The local ETH supplementary repo does not expose a large trainer-side staged
curriculum script. Most of the methodology is embedded directly in the
environment contract.

This is useful because it tells us which ingredients matter most before we add
more moving parts to `C1`.

### What `C1-ETHLike-V1` already matches

- blind student with deployable history-conditioned control
- long history window:
  - current `C1` uses `100`
  - ETH test apps also use `100`
- privileged teacher exists only during training
- student runtime path is still blind at deployment

### What `C1-ETHLike-V1` partially matches

- rough terrain rather than flat-only locomotion
- per-episode hidden dynamics randomization
- rough-terrain curriculum through terrain levels
- teacher-student structure

These are directionally correct, but still simpler than the ETH recipe.

### Important ETH ingredients still missing or simplified

#### 1. Terrain family diversity

ETH clearly trains across multiple terrain families in the environment itself:

- hills
- steps
- stairs
- uniform slope

Current `C1` still uses a simpler rough-terrain mix.

Interpretation:

- current `C1` is good for proving branch viability
- it is not yet an equally broad terrain-generalization claim

#### 2. Terrain-aware command sampling

ETH command generation is not just a fixed velocity box.
It samples goals and derives commands relative to terrain and task mode.

Current `C1` training is simpler:

- forward-only command distribution
- no lateral command diversity
- no yaw-command diversity

Interpretation:

- this is one of the biggest simplifications in the current C1 regime

#### 3. Richer student observation payload

ETH blind history uses a richer observation/history payload than our current
`48`-dim student observation.

Examples visible in the repo:

- gait-phase-like signals
- joint-target error structure
- richer contact/foot-context-related privileged machinery on the teacher side

Interpretation:

- our current C1 branch is structurally aligned
- but still lighter in what the student is asked to encode over time

#### 4. Stronger reset and initial-condition diversity

ETH reset logic visibly randomizes:

- terrain-aligned base pose
- heading/orientation
- joint state
- gait phase
- initial height on terrain

Current `C1` benefits from IsaacLab reset/randomization, but not yet with that
same explicit terrain-aware richness.

#### 5. Strong observation noise modeling

ETH injects noise directly into:

- body velocity
- angular velocity
- joint position
- joint velocity
- target-related terms

Current `C1` is not yet making the same level of observation-noise assumptions.

### What this means for current C1 interpretation

Current `C1-ETHLike-V1` should be interpreted as:

- a valid first ETH-like branch
- strongly aligned in deployment contract and temporal structure
- still simpler in command richness, terrain diversity, and observation realism

That is acceptable for V1.

It means:

- success here proves the branch family is viable
- it does not yet prove we have matched the full ETH robustness recipe

### Recommended upgrade order after V1 viability

If `C1-ETHLike-V1` remains healthy through training and early eval, the most
useful progression is:

1. broaden command distribution
   - add lateral and yaw diversity first
2. broaden terrain-family diversity
   - move closer to explicit hills / steps / stairs / slope coverage
3. enrich student observation/history payload only if needed
4. add stronger observation-noise realism

This order matters.

It keeps us from destabilizing the branch by changing too many axes at once,
while still moving toward the stronger ETH-style robustness recipe.

## Candidate 1 Comparison Anchors

These should stay in the evaluation story:

### B2

- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`

Role:

- clean canonical blind baseline
- motion-quality anchor
- simplest deployment-side reference

### B3

- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`

Role:

- robustness-leaning blind baseline
- geometry/push comparison anchor
- useful ETH-style robustness sanity check

### Dyn-only Stage A

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`

Role:

- current deployment-side winner overall
- nearest head-to-head reference for whether terrain+dynamics teacher-student
  training is actually buying us something

## Candidate 1 Success Criteria

Candidate 1 should eventually be defensible as:

- blind at deployment
- privileged-teacher-trained during learning
- competitive or superior to the simpler blind baselines
- competitive with the current dyn-only deployment winner
- visually convincing in Isaac and MuJoCo
- simpler to explain and deploy than the explicit adaptive RMA line

It does **not** need to prove:

- explicit online latent adaptation
- `phi(history)` as a changing hidden-factor estimate
- deploy-time adaptation semantics

## Immediate Work Order

### Step 1: Treat terrain-lite as the official Candidate 1 base

Do not let Candidate 1 drift into ambiguity.

For now, the working base should be:

- `adapt_v3_terrain_lite_phase2_stage_a_final.pt`

with:

- B2 as the clean blind reference
- B3 as the robustness blind reference

### Step 2: Push the current Candidate 1 base through deployment-style checks

The next concrete work is not new training first.
It is to make the current terrain-lite blind student legible as a deployment
candidate.

That means:

1. package/export the terrain-lite candidate cleanly
2. run the same Sim2Sim path used for dyn-only
3. generate canonical visual clips for Isaac and MuJoCo
4. compare directly against:
   - B2
   - B3
   - dyn-only Stage A

### Step 3: Decide if terrain-lite is already good enough

After deployment-style evaluation, decide which of these is true:

1. terrain-lite is already strong enough to freeze as Candidate 1
2. terrain-lite is real but needs one refinement pass
3. terrain-lite is too posture-fragile, and Candidate 1 should be reframed
   around a simpler blind reactive branch instead

### Step 4: If refinement is needed, keep it narrow

If Candidate 1 needs more work, the next refinement should stay focused on the
known problem:

- preserve the tracking benefit
- reduce the conservative low-base / posture-failure behavior

This should be framed as:

- terrain-lite refinement

not as:

- a return to dense privilege
- a detour into explicit adaptive latent repair

## What Not To Do

- do not redefine Candidate 1 around the dyn-only winner just because it wins
  today
- do not collapse Candidate 1 into generic blind baselines only
- do not mix Candidate 1 goals with RMA adaptation goals
- do not add new complexity before the current terrain-lite base has gone
  through Sim2Sim and visual comparison

- do not let C1 planning drift back across multiple overlapping notes
- do not treat archived C1 design notes as the main source of truth when this
  file already covers the topic

## Recommended Next Concrete Action

The next practical move for Candidate 1 is:

1. package/export
   `adapt_v3_terrain_lite_phase2_stage_a_final.pt`
2. run MuJoCo Sim2Sim rehearsal
3. record canonical Isaac and MuJoCo visual artifacts
4. compare it against:
   - `blind_baseline2_warmstart_final.pt`
   - `blind_baseline3_warmstart_imitation_final.pt`
   - `adapt_v3_dyn_only_phase2_stage_a_final.pt`

That will tell us whether Candidate 1 is already a serious finalist or whether
it needs one focused refinement pass.

## Read Next

- `docs/EXPLORATION_LEDGER.md`
- `docs/TWO_FINAL_CANDIDATES_ROADMAP.md`
