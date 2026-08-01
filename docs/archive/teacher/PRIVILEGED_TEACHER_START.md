# Privileged Teacher Start

Status:

- historical phase-entry note
- kept for phase-transition context
- canonical teacher truth now lives in:
  - `docs/TEACHER_V4_MODEL300_CARD.md`

This note is the careful starting point for the post-baseline phase.

The baseline regime is closed. Teacher work should now begin on top of the
frozen baseline package rather than by reopening baseline design.

Primary baseline references:

- `docs/BASELINE_COMPARISON_FINAL.md`
- `docs/OOD_FINDINGS_B1_B2_B3.md`
- `docs/FROZEN_BASELINE_SYNTHESIS.md`
- `docs/BASELINE_REGIME_CLOSED.md`
- `rma_go2_lab/policies/blind_baseline_protocol.md`

## What We Already Have

The active branch now gives us:

- a frozen proprio-only flat prior
- frozen blind baselines B1, B2, and B3
- canonical in-distribution evaluation
- exploratory OOD evaluation across:
  - geometry shifts
  - static dynamics shifts
  - push recovery
  - abrupt mid-episode switches

This means the next phase does not need to ask:

- can a blind controller work at all?
- does warm-start matter?
- does temporary imitation change the robustness profile?

Those questions are already answered well enough to support the next branch.

## Governing Principle

The first privileged teacher should be built as a clean upper-bound comparison
on the same locomotion problem, not as a broad redesign of the project.

The teacher phase should answer:

> what does privileged information buy us on the same rough-locomotion task
> once the blind baseline regime is frozen?

This is intentionally narrower than:

- full RMA adaptation
- perception-enabled locomotion
- obstacle-specialist locomotion
- deployment-ready student distillation

Those may follow later, but the first teacher should not try to solve all of
them at once.

## Fairness Envelope

To keep the teacher phase interpretable, the first privileged teacher should
inherit as much as possible from the frozen blind benchmark.

Keep fixed:

- base terrain family from `rma_go2_lab/envs/blind/rough_cfg.py`
- command distribution
- reward structure unless a concrete teacher-only pathology forces a change
- evaluation stack
- canonical frozen baseline set used for comparison

Change deliberately:

- observation interface for the teacher
- possibly critic interface if needed
- later, teacher-specific distillation or adaptation machinery

Do not change all of these at once:

- terrain family
- command regime
- reward recipe
- privileged observations
- adaptation architecture
- perception stack

If we do that, we lose the clean comparison frame we just earned.

## What The First Teacher Should Be

Recommended first teacher:

- same rough locomotion task
- same action space
- same command regime
- same terrain curriculum family
- privileged actor observations enabled
- privileged critic observations allowed
- no student yet
- no adaptation latent yet
- no camera or perception stack yet

In other words:

- first build a strong privileged rough controller
- then decide how to use it

That gives us a clean answer to:

- how much headroom is still available above the frozen blind baselines?

## Recommended Starting Reference

Use Baseline 2 as the main reference baseline for the first teacher phase.

Why Baseline 2:

- it is the clean canonical blind winner
- it has the strongest overall static mismatch profile
- it is the most defensible deployable-style baseline

Keep B3 in view as a secondary comparison because:

- it is stronger on geometry OOD
- it is stronger on push recovery

But if we need one blind anchor for the teacher phase, it should be B2.

## Candidate Privileged Inputs

The first teacher should get only privilege that we can justify clearly.

Good first candidates:

- terrain height samples or height-scan style local terrain structure
- explicit friction parameters
- explicit mass offset / payload parameters
- explicit actuator strength scales

These are attractive because they match the mismatch axes already explored in
the frozen OOD study.

This means the first teacher can be framed as:

- a controller that sees the hidden variables the blind baselines do not

That is much cleaner than jumping directly to a complex latent architecture.

## Candidate Teacher Variants

The clean ordering is:

1. `Teacher V0: terrain + dynamics privileged teacher`
   - same rough task
   - same commands
   - same action space
   - actor gets local terrain and dynamics privilege
   - critic can also stay privileged

2. `Teacher V1: terrain-only privileged teacher`
   - isolates geometric privilege

3. `Teacher V2: dynamics-only privileged teacher`
   - isolates hidden dynamics privilege

The repo does not need all three immediately.

Recommended first move:

- start with `Teacher V0`
- then ablate later if needed

## Why Not Jump Straight To The Student

Because we still need to separate three questions:

1. how strong can the privileged policy become?
2. which privileged information matters most?
3. how much of that can later be transferred to a student?

If we skip question 1, later student results will be harder to interpret.

## What The Teacher Should Not Quietly Change

The first teacher should not quietly add:

- a new terrain curriculum family
- stair and box training by default
- a broader command manifold
- more reward shaping just because the policy can exploit it
- perception sensors and privileged terrain together in the same first step

Each of those would turn the teacher phase into a different experiment.

## Minimal SOP For Teacher Phase

Before training:

1. define the exact teacher observation groups
2. define which hidden variables are privileged and why
3. define the primary blind comparison baseline
4. define the qualification evaluation stack
5. define what counts as a fair teacher gain

During training:

- keep the task otherwise matched to the frozen blind regime
- log teacher-specific metrics explicitly
- avoid informal reward changes

Before freezing:

- run the same canonical evaluation stack as the blind baselines where
  applicable
- run the same OOD suites where meaningful
- add a clearly labeled teacher artifact family rather than mixing it into the
  baseline folders

## Recommended Immediate Next Deliverables

The next concrete deliverables should be:

1. `teacher_design_v0.md`
   - exact privileged observation specification
   - fairness statement
   - comparison baseline

2. `teacher_rough_cfg.py`
   - derived carefully from the frozen blind rough task

3. `teacher_ppo_cfg.py`
   - initially close to the blind PPO recipe unless a teacher-specific reason
     appears

4. one registered task
   - single teacher entry point, not a family explosion

## Bottom Line

We do have something solid "in the bag" for privileged teacher now:

- a stable blind benchmark
- a clear canonical baseline anchor
- a map of which robustness axes remain hard

The right next move is not to rebuild RMA immediately.

The right next move is:

- first train one fair privileged teacher on the same rough problem
- measure its headroom over B2 and B3
- only then decide how to turn that into adaptation or student work
