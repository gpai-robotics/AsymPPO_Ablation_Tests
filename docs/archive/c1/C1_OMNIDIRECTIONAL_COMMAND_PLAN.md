# C1 Omnidirectional Command Plan

This file now serves two purposes:

- preserve the lineage of the omni command expansion work
- record the current cleaned omni stack and the active next step

This note defines the next active development step for Candidate 1 after the
public `C1` baseline split.

The goal is simple:

- keep the current `C1` blind-history architecture
- keep the current teacher-derived training story
- expand the command regime from forward-only to true planar/yaw tracking

This work stays in the private `rma_go2_lab` repo first. The public repo should
only be updated later if the result is clearly better and stable.

## Why This Is Next

Current `C1` is already the best deployable blind-history artifact in the repo,
but the active command regime is still narrow:

- `lin_vel_x` forward only
- `lin_vel_y = 0`
- `ang_vel_z = 0`
- `heading_command = False`

That makes the next expansion path obvious:

- preserve the baseline
- broaden the command family
- measure what breaks

This is a better next step than adding more infrastructure, because it improves
practical locomotion capability without reopening the architecture question.

## Current Source Of Truth

Current active omni stack:

- flat prior:
  - `rma_go2_lab/policies/flat_omni_v1.pt`
- frozen omni teacher:
  - `rma_go2_lab/policies/rough_omni_teacher_v1.pt`
- frozen deployable omni student:
  - `rma_go2_lab/policies/c1_blind_rough_omni_usable_v1_final.pt`

For code navigation, the active files are:

- `rma_go2_lab/envs/teacher/rough_omni_v1_cfg.py`
- `rma_go2_lab/models/teacher/ppo_omni_v1_cfg.py`
- `rma_go2_lab/envs/blind/c1_blind_rough_omni_usable_cfg.py`
- `rma_go2_lab/models/blind/blind_rough_runner_cfg.py`

The earlier omni student branches should now be read as provisional probes, not
as the canonical omni training definition.

## Frozen Findings

What is already settled:

- `flat_omni_v1` is the preferred standalone flat omni prior
- contact-only and contact+phase flat omni branches were useful ablations, but
  did not replace `flat_omni_v1`
- the privileged rough omni teacher now exists and is strong enough to freeze
- omni-student work should no longer use the old forward-only rough teacher

What is still active:

- export and deployment of the frozen deployable blind rough omni student
- sim2real gap study around the frozen deployment candidate

What is now also settled on the deployment side:

- export parity for the frozen student is clean
- Isaac deploy rehearsal is clean
- nominal flat-surface MuJoCo Sim2Sim sanity check is clean
- the first concrete deployment-facing failure case is low friction
- added mass and moderate motor weakening are not the dominant immediate gaps

Task:

- `RMA-Go2-C1-ETHLike-V3-StageA`

Env chain:

- `rma_go2_lab/envs/blind/blind_rough_forward_cfg.py`
- `rma_go2_lab/envs/blind/blind_rough_forward_history_cfg.py`
- `rma_go2_lab/envs/blind/c1_blind_rough_teacher_history_cfg.py`

Runner config:

- `rma_go2_lab/models/blind/blind_rough_runner_cfg.py`
  - `Go2C1BlindRoughTeacherHistoryV3PPORunnerCfg`

Current command bottleneck is in:

- `rma_go2_lab/envs/blind/blind_rough_forward_cfg.py`

Specifically:

- `cmd.ranges.lin_vel_y = (0.0, 0.0)`
- `cmd.ranges.ang_vel_z = (0.0, 0.0)`
- `cmd.heading_command = False`

## Development Strategy

Do this as a staged fork of `C1`, not as an in-place mutation of the current
canonical command baseline.

Recommended new line:

- `usable omni student` supervised by a true `omni rough teacher`

Meaning:

- reuse current `C1` architecture
- reuse current StageA teacher lineage
- only change command distribution and whatever stabilization is needed

## Proposed Implementation Order

### Phase 1: Yaw And Lateral Unlock

Create a new env variant derived from the current `C1` env and expand commands
conservatively.

Suggested first ranges:

- `lin_vel_x = (-0.5, 1.0)`
- `lin_vel_y = (-0.4, 0.4)`
- `ang_vel_z = (-0.8, 0.8)`
- keep `heading_command = False` at first

Reason:

- this adds real omnidirectional tracking pressure
- but avoids immediately coupling in heading-command logic

This should be the first implementation slice.

### Phase 2: Reward And Failure Rebalance

After command unlock, expect these failure modes:

- sideways drift instability
- turning-induced body tilt
- poor standstill behavior during yaw commands
- low-progress termination firing too early on turning maneuvers

Likely reward/termination updates:

- inspect `track_lin_vel_xy_exp`
- inspect `track_ang_vel_z_exp`
- retune standstill penalties
- retune `low_progress_termination`

Do not redesign rewards broadly at first. Only adjust what the wider command
family clearly breaks.

### Phase 3: First Training Sweep

Run a short sweep aimed at stability, not final quality.

Questions to answer:

- does the current architecture train at all under omni commands?
- does history remain useful?
- do sideways and yaw commands degrade terrain robustness too much?

Initial sweep suggestions:

- current `C1` runner, unchanged optimizer schedule
- 2 to 3 short runs with command-range variants
- no export work yet

### Phase 4: Evaluation Gate

Before any longer run, require a clear answer on:

- forward command quality regression
- lateral command tracking quality
- yaw tracking quality
- rough-terrain recovery under omni commands

If omni support destroys the original forward-strength baseline, fix that before
moving to longer training.

## Historical First-Slice Record

The sections below remain as historical implementation context for how the omni
line was opened up. They should not override the cleaned active stack above.

## Recommended New Files

To keep the current `C1` line clean, add new variants instead of mutating the
existing ones immediately.

Suggested additions:

- `rma_go2_lab/envs/blind/c1_blind_rough_omni_cfg.py`
- `rma_go2_lab/models/blind/c1_omni_ppo_cfg.py`

Suggested task name:

- `RMA-Go2-C1-Omni-V1-StageA`

This keeps:

- current `C1` deploy card stable
- current public story stable
- new work isolated and auditable

## First Concrete Implementation Slice

The first slice should be intentionally small:

1. add a new C1 omni env config derived from the current blind-history rough env
2. widen `lin_vel_y` and `ang_vel_z`
3. keep heading-command mode off
4. register a new task
5. point it at a runner config cloned from `Go2C1BlindRoughTeacherHistoryV3PPORunnerCfg`
6. run a short training smoke test

That is enough to answer whether the current `C1` architecture can survive the
expanded command family.

## Success Criteria For C1 Omni V1

Early success is not "perfect omni locomotion."

Early success is:

- training remains stable
- forward performance does not collapse
- lateral and yaw commands are visibly nontrivial and trackable
- rough-terrain stability remains recognizably `C1`-like

## What Not To Do Yet

- do not merge omni changes directly into the current public `C1` path
- do not reopen the explicit adaptation question here
- do not add heading-command complexity in the first slice
- do not start with export/deploy packaging

That phase is now complete for the current omni line.

The active next step is deployment rehearsal, trusted flat-surface Sim2Sim
sanity checking, and friction-focused gap confirmation, not further
omni-architecture expansion.

## Recommended Immediate Next Step

Implement the smallest working `C1 omni v1` branch:

- new env config
- new registered task
- cloned runner config
- first short training launch

That is the highest-value next move.

## Current Status

`C1 omni v1` now exists and has already answered the first feasibility
question.

Implemented:

- provisional omni student probes were explored first
- the cleaned active student branch is now:
  - `rma_go2_lab/envs/blind/c1_blind_rough_omni_usable_cfg.py`
  - task `RMA-Go2-C1-Omni-Usable-V1-StageA`
- the cleaned active teacher branch is now:
  - `rma_go2_lab/envs/teacher/rough_omni_v1_cfg.py`
  - task `RMA-Go2-Privileged-Teacher-Rough-Omni-V1`

Observed so far:

- training is stable
- command curriculum now widens correctly
- the policy survives broader planar and yaw commands
- command breadth can reach beyond the initial narrow `+-0.1` regime
- tracking quality degrades as the command envelope broadens

This means the current core question is:

- how well can the deployable blind omni student inherit from the frozen
  omnidirectional rough teacher?

## Warm-Start Note

The current `C1 omni v1` line originally inherited a forward-biased flat
warm start. That is useful for stability, but it is also a real source of bias.

To reduce that bias, add and train a flat omni prior first:

- task `RMA-Go2-Flat-Omni-V1`
- env `rma_go2_lab/envs/priors/flat_omni_prior_cfg.py`
- runner `Go2FlatOmniPriorPPORunnerCfg`

Once a stable flat omni checkpoint exists, it should become the preferred
warm start for both:

- the privileged rough omni teacher
- the deployable blind rough omni student

## Reference-Backed V2 Patch Order

The next omni patch should stay disciplined and close to what the reference
repos already suggest.

### Best-First Patch

Add richer gait-state cues to the `C1 omni` observation contract.

Especially:

- contact-state observation
- gait/phase observation

Why:

- the ETH blind reference is history-conditioned, but it is also more
  gait-aware at each instant
- long history alone is likely not enough for clean broad omni tracking

If only one idea is adopted first, this is the one.

### Second Patch

Import locomotion-structure rewards from `unitree_rl_lab`.

Especially:

- air-time variance style penalty
- joint-position penalty with moving vs standstill sensitivity

Why:

- these are more grounded than broad reward retuning
- they directly target locomotion quality rather than only punishing failure

### Third Patch

Adopt the command deadzone idea from `unitree_rl_gym`.

Specifically:

- zero tiny planar commands

Why:

- avoid spending training effort on low-signal near-zero lateral requests
- keep the curriculum, but make the command distribution cleaner

## What Not To Do First

Even if these become useful later, they should not be the first reaction:

- do not shorten history first
- do not add heading-command mode first
- do not do broad reward retuning everywhere

These are less grounded than the three patches above.

## Recommended V2 Scope

Keep the current omni branch and make one focused `v2` patch with:

- contact-state observation
- phase observation
- optionally one locomotion-structure reward such as air-time variance

That is the cleanest next step after the flat omni warm-start work.

## Freeze: Flat Omni Prior Findings

The flat-prior ablation is now clear enough to freeze.

Completed priors:

- `flat_omni_v1`
  - task `RMA-Go2-Flat-Omni-V1`
  - env `rma_go2_lab/envs/priors/flat_omni_prior_cfg.py`
  - promoted policy `rma_go2_lab/policies/flat_omni_v1.pt`
- `flat_omni_contact_v1`
  - task `RMA-Go2-Flat-Omni-Contact-V1`
  - env `rma_go2_lab/envs/priors/flat_omni_contact_v1_cfg.py`
  - kept as a valid contact-only ablation result
- `flat_omni_contact_phase_v1`
  - task `RMA-Go2-Flat-Omni-Contact-Phase-V1`
  - env `rma_go2_lab/envs/priors/flat_omni_contact_phase_v1_cfg.py`
  - kept as a valid contact+phase ablation result

Comparison outcome:

- `flat_omni_v1` remains the preferred standalone flat omni prior
- `flat_omni_contact_v1` converges well and is valid, but does not produce a
  global improvement over `flat_omni_v1`
- `flat_omni_contact_phase_v1` also converges well and is valid, but still
  does not beat `flat_omni_v1` on the fixed-command comparison
- contact-only observation helps some specific cases such as backward and some
  yaw commands
- contact+phase observation also helps some local command corners, but the
  simpler `flat_omni_v1` remains better overall
- extra observation structure did not produce a decisive standalone flat-prior
  improvement in this round

Evidence used for this freeze:

- TensorBoard comparison of the two full runs
- fixed-schedule evaluator:
  - `scripts/eval/eval_flat_omni_schedule.py`
  - `scripts/eval/compare_flat_omni_priors.py`
- artifacts:
  - `artifacts/evaluations/flat_omni_prior_compare/comparison.md`
  - `artifacts/evaluations/flat_omni_prior_compare/comparison.json`
  - `artifacts/evaluations/flat_omni_v1_vs_contact_phase_v1/comparison.md`
  - `artifacts/evaluations/flat_omni_v1_vs_contact_phase_v1/comparison.json`

Practical conclusion:

- keep `flat_omni_v1.pt` as the preferred flat omni warm start
- freeze `flat_omni_contact_v1` as an observation ablation, not as the new
  default prior
- freeze `flat_omni_contact_phase_v1` as an observation ablation, not as the
  new default prior

## Recommended V3 Scope

`v3` has now been completed and frozen.

Implemented `v3` target:

- `contact + phase` flat expert first
- then a matching `C1 omni` branch that uses the same observation contract

Outcome:

- `v1` established that an omni flat prior materially improves the rough omni
  startup story
- contact-only `v2` showed that explicit contact state is compatible and can
  help some command families
- contact+phase `v3` showed that the richer observation contract is trainable
  and can produce a strong omni prior
- however, it still did not beat `flat_omni_v1` in the final fixed-schedule
  A/B comparison

So the current freeze point is:

- `v1`: omni flat prior only, preferred default
- `v2`: contact-only ablation, frozen
- `v3`: contact + phase ablation, frozen
