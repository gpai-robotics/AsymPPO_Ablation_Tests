# Teacher Design V0

This is the first concrete privileged-teacher design for the post-baseline
phase.

It intentionally starts smaller than the full long-term teacher/student vision.

## Purpose

Teacher V0 is meant to answer:

> how much headroom is available above the frozen blind baselines when the
> policy is allowed to see local terrain geometry on the same rough-locomotion
> task?

This is a teacher upper-bound question, not yet a student-distillation or
adaptation question.

## Comparison Anchor

Primary blind reference:

- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`

Secondary comparison:

- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`

Baseline 2 remains the canonical blind anchor because it is the cleanest
in-distribution controller and the safest static-mismatch baseline.

Initialization rule:

- Teacher V0 should warm-start from the frozen B2 checkpoint in a shape-aware
  way
- the terrain encoder starts neutral
- the downstream actor/critic layers inherit the blind rough controller
- this makes the teacher comparison "B2 plus terrain privilege" rather than a
  fresh scratch policy

## Fairness Rules

Teacher V0 must inherit the frozen blind regime as closely as possible.

Keep fixed:

- rough terrain family
- command distribution
- action space
- reward structure
- terminations
- evaluation stack

Change only:

- actor/critic observation access to local terrain geometry

Do not change in Teacher V0:

- terrain training mix
- stair or box training proportions
- broader command space
- reward weights
- perception stack
- adaptation latent or student architecture

## Teacher V0 Choice

The first implementation is a terrain-privileged teacher:

- actor sees:
  - the same proprio observation group as the blind baselines
  - a privileged local terrain height scan through a small terrain encoder
- critic sees:
  - the same combined actor observation set

This is intentionally narrower than a full terrain-plus-dynamics privileged
teacher.

Why start here:

- the height scanner is already a native part of the upstream rough task
- it is easy to define precisely
- it cleanly isolates geometric privilege
- it avoids guessing at internal dynamics randomization state for the first
  scaffold

## Why Not Dynamics Privilege First

We do want explicit dynamics privilege later.

But for the first scaffold, terrain privilege is the safest choice because:

- it is already represented explicitly in the environment through a ray-caster
- it does not require custom bookkeeping for startup-randomized hidden
  parameters
- it gives us a clean, interpretable first teacher

The later extension path is:

1. Teacher V0
   - terrain privilege only
2. Teacher V1
   - terrain + explicit dynamics privilege
3. later
   - student / adaptation branch

## Registered Task

Historical note:
- `RMA-Go2-Privileged-Teacher-Rough` is an archived task id.
- It is documented here as the original V0 scaffold, not as a live registered
  entry in the current repo surface.

Teacher V0 task:

- `RMA-Go2-Privileged-Teacher-Rough`

Environment config:

- `rma_go2_lab/envs/teacher/rough_cfg.py`

Runner config:

- `rma_go2_lab/models/teacher/ppo_cfg.py`

## Observation Mapping

Environment groups:

- `policy`
  - same proprio terms as the frozen blind ladder
- `privileged`
  - height scan only

Runner mapping:

- actor `policy` set:
  - `["policy", "privileged"]`
- critic `critic` set:
  - `["policy", "privileged"]`

This keeps the definition explicit and avoids hiding privileged terrain inside
the nominal blind observation group.

Representation choice:

- the height scan remains its own environment group
- the policy does not consume the raw `187` terrain values by direct flat
  concatenation anymore
- instead, a small terrain encoder compresses the scan into a lower-dimensional
  latent before fusing it with proprio

This is still a fair Teacher V0 change because the new capacity is only there
to make the single privileged channel usable. The task, reward, and evaluation
frame stay fixed.

## PPO Choice

Teacher V0 should stay close to the blind PPO recipe.

Initial choice:

- same hidden dimensions as B1/B2/B3
- same rollout horizon
- same PPO update schedule
- same entropy and KL settings
- B2-aware warm-start for actor/critic initialization

If the teacher later shows a clear optimization pathology, we can adjust the
recipe then. The default should be parity, not aggressive retuning.

## Qualification Plan

Teacher V0 should be evaluated against the same frozen baseline frame where
possible:

- gait checks
- canonical isolated suite
- canonical recording manifest

And later, the exploratory OOD suites:

- geometry OOD
- dynamics OOD
- push OOD
- switch OOD

The point is not to prove the teacher wins every axis immediately.

The point is to measure:

- nominal headroom over B2
- which OOD axes improve with terrain privilege alone
- which remaining failures still motivate adaptation or later dynamics
  privilege

## Outcome

Teacher V0 was implemented and trained to convergence with:

- a terrain encoder for the `187`-dim height scan
- a shape-aware warm-start from frozen B2

This fixed the earlier failed scratch-teacher attempt and produced a stable
privileged run that trained at high terrain curriculum levels.

However, the frozen evaluation result was not a clean win over B2.

Main observations from the valid Teacher V0 eval pass:

- standstill behavior was acceptable and close to B2
- forward-motion behavior was weaker than B2
- the policy ran with a visibly lowered base during locomotion
- nominal forward tracking was substantially worse than B2
- isolated-suite performance did not show a persuasive nominal advantage over
  the blind anchor

Interpretation:

- Teacher V0 did learn a usable terrain-aware controller
- but it appears to have converted terrain privilege into a conservative,
  crouched stability strategy rather than clearly better locomotion quality

So Teacher V0 is scientifically useful, but not the teacher recipe to carry
forward unchanged.

It should be treated as:

- a valid first privileged probe
- a documented negative-or-mixed result
- a design lesson for the next teacher iteration

## Locked Takeaway

What Teacher V0 tells us:

- terrain privilege is usable under the current stack
- warm-start from B2 is important
- raw scratch teacher training is not the right path
- terrain encoder plus warm-start is necessary, but not sufficient
- without additional pressure, the teacher can settle into low-base,
  conservative locomotion

Decision:

- lock Teacher V0 as complete
- do not keep tuning it indefinitely
- move forward to a next teacher iteration with the V0 lesson preserved

## Expected Interpretation

If Teacher V0 improves mostly on geometry-linked failures, that supports the
claim that local terrain visibility is a major missing ingredient.

If Teacher V0 does not improve much on static dynamics mismatch or abrupt
switches, that would be a useful result too. It would tell us terrain
privilege alone is not the whole story.

## Bottom Line

Teacher V0 is the smallest fair next step:

- same rough problem
- same benchmark frame
- one explicit new advantage:
  local terrain geometry

That is the right place to restart teacher work without destabilizing the
project story.
