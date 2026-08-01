# Teacher Phase Synthesis

Historical note:
- this file is no longer the canonical teacher definition
- the single source of truth for the active teacher is now
  [TEACHER_V4_MODEL300_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/TEACHER_V4_MODEL300_CARD.md)

Use this file for:

- teacher lineage
- earlier branch outcomes
- historical phase synthesis

This note consolidates the first privileged-teacher phase after the frozen
blind baseline regime.

The teacher question was:

> does privileged local terrain information produce a meaningfully better rough
> locomotion controller than the frozen blind anchor B2?

## Post-Freeze Recovery Update

The original teacher chapter froze around `Teacher V3 final`, but later
dependency-audit work changed the repo's understanding of what was actually
load-bearing in the frozen checkpoints.

Current active teacher truth:

- `Teacher V3 final` remains a useful historical reference
- it is validated as using `dynamics_privileged`
- it is not validated as using `terrain_privileged`
- `Teacher V4 model_300` is the single current canonical active teacher
  candidate
- `Teacher V4.1`, `Teacher V5`, and `Teacher V6` are archived exploratory
  branches, not active teacher candidates

Current honest limitation:

- we still do not have a single teacher checkpoint that is the best validated
  terrain+dynamics user across every terrain family
- the project contract still requires one overall robust rough-terrain teacher
- until a better overall teacher exists, the repo keeps the best overall
  candidate active and archives specialized detours

## Teacher Ladder

### V0

- terrain privilege only
- terrain encoder latent `32`
- B2-aware warm-start
- no anti-crouch correction

Outcome:

- technically successful
- behavior remained clearly crouch-biased
- not a persuasive all-around improvement over B2

### V1

- same terrain privilege as V0
- same terrain encoder latent `32`
- same B2-aware warm-start
- plus motion-gated terrain-aware base-height penalty

Outcome:

- better than V0
- less pathological than V0
- geometry-linked usefulness became more visible
- still not a clean decisive win over B2

### V2

- same environment intervention as V1
- same B2-aware warm-start
- compressed terrain latent `8`

Outcome:

- matched or slightly exceeded V1 in training and frozen eval
- suggests the larger `32`-dim terrain latent was not necessary
- currently the best teacher recipe of the first phase

### V3

- same anti-crouch environment as V2
- same compressed terrain latent `8`
- adds explicit raw dynamics privilege
  - static friction
  - dynamic friction
  - base-mass ratio
  - joint stiffness scales
  - joint damping scales

Outcome:

- strongest teacher in the full ladder
- materially stronger than terrain-only teachers on geometry and dynamics OOD
- first teacher that makes a real case that richer privilege improves
  robustness
- still not a universal replacement for B2 on every nominal axis

Later dependency-audit qualification:

- the frozen `model_1999` checkpoint does not appear to use the
  `terrain_privileged` channel on the audited forward probes
- the same checkpoint does depend materially on `dynamics_privileged`
- so V3 should now be described as:
  - a stronger teacher line with both privileged groups exposed
  - a frozen checkpoint that is validated as dynamics-privileged
  - not yet a checkpoint that is validated as terrain-using

## What We Learned

### 1. Terrain privilege was exposed, but the frozen V3 checkpoint does not validate its use

The teacher ladder proved that terrain privilege could be wired into training
and did not make optimization impossible. But the later root audits matter
more than the earlier architectural intent:

- on `random_rough l5`
- on `boxes l5`
- on `pyramid_stairs l5`

the frozen `V3` checkpoint is unchanged under `zero_terrain`.

So the current honest claim is:

- terrain privilege was exposed successfully
- but the frozen `V3` checkpoint does not presently prove terrain dependence

### 2. Raw privilege does not automatically produce good behavior

V0 showed that simply adding terrain information can lead to a conservative
low-base strategy rather than clearly better locomotion.

This means:

- privilege availability is not the same as privilege being used well

### 3. Anti-crouch shaping helped, but did not fully solve the style issue

V1 improved over V0 and made the teacher more plausible, but visual inspection
still suggested a similar crouched motion style.

So V1 should be read as:

- a corrective step
- not a final explanation of the teacher behavior

### 4. Compressed terrain privilege is sufficient

V2 kept the anti-crouch intervention and reduced the terrain latent from `32`
to `8`.

This did not collapse performance. Instead, V2 remained competitive and often
looked slightly cleaner numerically.

That suggests:

- the privileged terrain information can be represented compactly
- the larger terrain code was not obviously load-bearing

This is the strongest positive result of the teacher phase so far.

### 5. Richer privilege matters more than terrain-only privilege, but the frozen V3 gain is being carried by dynamics

V3 kept the V2 terrain recipe fixed and only added explicit hidden-dynamics
privilege.

That changed the result meaningfully:

- canonical robustness improved
- geometry OOD improved
- dynamics OOD improved
- switch behavior improved

This is still the clearest evidence from the teacher phase that privileged
information was not merely decorative. But the later dependency audits sharpen
the interpretation:

- the frozen gains are being carried by hidden-dynamics privilege
- the same frozen checkpoint does not show measurable terrain dependence on the
  audited forward probes

## What Privilege Seems To Help

The full teacher phase gives the strongest evidence for:

- geometry-linked usefulness
- hidden-dynamics robustness
- switch robustness under mid-episode changes

This is most visible in:

- geometry OOD cases
- dynamics OOD cases
- some switch OOD cases
- some canonical rough-tracking metrics

The teacher phase does **not** support a strong claim that privilege alone
creates a clearly superior all-around controller under every evaluation axis.

## What Privilege Does Not Yet Prove

The current teacher results do **not** justify claiming:

- privilege is strictly necessary in the original-RMA sense
- the privileged teacher dominates the frozen blind baseline on every metric
- privilege alone solves weak-motor or all discrete-obstacle failure modes
- teacher-only PPO is sufficient to make the full adaptation argument

So the most honest current conclusion is:

- terrain-only privilege helped somewhat
- terrain + dynamics privilege helped materially more
- V3 is the first teacher that clearly outperformed earlier teachers in the
  places where hidden factors matter
- the frozen V3 checkpoint is a trustworthy dynamics-privileged teacher
- the same frozen checkpoint is not yet a trustworthy terrain-using teacher
- but the teacher phase still falls short of a full original-RMA-style case,
  because deployable adaptation has not yet been demonstrated

## Frozen Teacher Reference

The original teacher chapter is now considered historically frozen.

Historic frozen reference:

- `V3 final`
- checkpoint:
  `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v3/2026-04-21_15-35-03/model_1999.pt`

Why this is the frozen reference:

- best overall robustness profile
- best evidence that hidden-dynamics privilege is genuinely useful
- best upper bound for the adaptation phase

Supporting references that remain useful:

- `V2 final`
  - best terrain-only teacher
- `V0 final`
  - first proof that the privileged path worked technically
- `B2 final`
  - frozen blind anchor

## Current Active Teacher Candidate

The repo should no longer present every post-`V3` branch as active.

Current validated active teacher candidate:

- `V4 model_300`
  - checkpoint:
    `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v4_terrain_aux/2026-05-09_10-34-56/model_300.pt`
  - role:
    canonical overall teacher candidate
  - current evidence:
    validated as using both terrain and dynamics privilege on the audited
    `random_rough` and `boxes` probes

Archived specialized evidence:

- `V4.1 model_1999`
  - checkpoint:
    `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v41_stair_bias/2026-05-09_16-20-41/model_1999.pt`
  - role:
    stair-specialized diagnostic branch
  - current evidence:
    validated as using both terrain and dynamics privilege on the audited
    `pyramid_stairs` and `pyramid_stairs_inv` probes
  - why it is archived:
    it improves stairs but regresses the general rough-terrain story, so it
    does not match the project's overall-teacher contract

Archived exploratory branches:

- `V5`
  - useful for proving terrain-family inequality and exposing stability risk
  - not kept as an active teacher line
- `V6`
  - useful as an IsaacLab-style direct-input idea
  - not pursued as an active teacher line

## Next Research Direction

The next step is **not** another teacher variant.

The next phase is:

- adaptation

Why:

- `V3` answered the main teacher question well enough
- the remaining gap to original RMA is no longer about "more privilege"
- it is about making the latent/adaptation mechanism structurally important

The active handoff document is:

- [ADAPTATION_PHASE_PLAN.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/archive/adaptation/ADAPTATION_PHASE_PLAN.md)

## Bottom Line

The teacher phase did not prove an original-RMA-strength adaptation story.
It did prove something narrower and still valuable:

- privileged information can help materially
- hidden-dynamics privilege is clearly load-bearing in the frozen V3 checkpoint
- terrain privilege was exposed but is not validated as load-bearing in the
  same checkpoint
- terrain-only privilege was not the whole story
- adding hidden-dynamics privilege produced the strongest teacher we observed
- `V3 final` is a credible dynamics-privileged upper bound for the next phase,
  but not a fully validated terrain+dynamics upper bound
