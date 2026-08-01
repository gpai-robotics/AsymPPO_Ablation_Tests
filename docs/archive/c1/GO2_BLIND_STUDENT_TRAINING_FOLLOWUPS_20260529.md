# GO2 Blind Student Training Follow-Ups 2026-05-29

This note translates the current real-robot deployment issues of
`c1_blind_rough_omni_usable_v1_final` into training-side follow-up items.

It does **not** assume we should retrain immediately.

The current working stance is:

- first extract the best possible real deployment behavior from the frozen
  policy
- then retrain only after we are confident the remaining gap is genuinely
  policy-side or sim-to-real-side, not just deploy-side sloppiness

## Why This Note Exists

Current real deployment issues observed on the old robot:

- small commanded velocity can still create aggressive body motion
- releasing the joystick does not reliably produce a crisp stop
- lateral and yaw commands can leak into forward/backward motion
- command-range reduction alone does not reliably bound real body speed

At the same time:

- these issues are not showing up clearly in IsaacLab simulation
- they are also not showing up clearly in MuJoCo rehearsal

That strongly suggests the remaining gap is not just “the policy is bad in
everywhere.” It is more likely a sim-to-real mismatch in one or more of:

- actuator response
- target tracking dynamics
- friction/contact details
- real sensor/odometry behavior
- small-carryover locomotion dynamics after command release

## What This Means Practically

We should separate two questions:

1. What can still be improved in deployment right now without retraining?
2. Which deployment patches are symptoms of missing training structure?

This note focuses on question 2, but keeps question 1 in view.

## Training-Side Hypotheses

### 1. Stop-on-release is under-taught

Observed on hardware:

- command returns near zero
- robot continues moving longer than expected

Likely training-side reason:

- the student is trained to track moving commands across a practical omni
  envelope
- but “release means settle immediately” is probably not emphasized strongly
  enough as a distinct behavior

Evidence:

- standstill rewards exist, but are thresholded and fairly soft
- no explicit “hard neutral pulse and arrest motion rapidly” objective is
  obvious in the current blind student setup

Possible training follow-ups:

- stronger penalty on planar velocity and yaw rate when command norm is near
  zero
- explicit command pulse / release curriculum
- more frequent command changes with abrupt neutral segments
- stronger penalty on residual action magnitude or foot motion after command
  release

### 2. Off-axis leakage is under-penalized

Observed on hardware:

- lateral command can induce backward/forward motion
- yaw command can induce translation

Likely training-side reason:

- tracking reward mainly encourages getting the commanded component right
- it may not punish non-commanded components strongly enough in the real regime

Possible training follow-ups:

- add cross-axis penalties:
  - penalize `vx` during pure `vy`
  - penalize `vy` during pure `vx`
  - penalize planar translation during pure yaw
  - penalize yaw during pure translation when inappropriate
- bias curriculum toward isolated-axis teleop segments, not only mixed omni
  motion

### 3. Action/target stream is too aggressive for real hardware

Observed on hardware:

- small filtered commands still produce large joint targets and large
  `tau_est`
- deployment-side action scale reduction helps somewhat
- deployment-side joint target slew limiting is plausible and common in other
  Go2 deploys

Likely training-side reason:

- action-rate and effort regularization exist, but are weak relative to
  tracking terms
- the student may have learned a target stream that is acceptable in sim but
  too abrupt for the real robot and PD loop

Possible training follow-ups:

- increase `action_rate_l2` magnitude
- moderately strengthen torque and/or acceleration regularization
- consider training-time joint target slew limiting if we conclude it is a
  permanent deploy mechanism
- consider explicit target smoothness penalties over consecutive actions

### 4. Actuator-domain realism is not rich enough

Observed on hardware:

- nominal deploy gains match training/export assumptions on paper
- yet the real response is still more coupled and more aggressive than sim

Likely training-side reason:

- the student may be overfit to the nominal actuator model even if stiffness and
  damping randomization exist
- the real actuator-plus-mechanics response may differ in ways not covered by
  current randomization

Possible training follow-ups:

- broaden actuator gain randomization if safe and justified
- include more latency / target lag / target slew effects in training if these
  are the deploy-time safeguards we keep needing
- revisit friction/contact realism and low-speed stance-transition behavior

## What This Suggests About The Current Setup

Because the bad behavior is not clearly reproduced in IsaacLab or MuJoCo:

- we should be cautious about assuming immediate retraining is the best next
  move
- the current deploy stack may still be missing the right minimal real-world
  safeguard

In other words:

- if simulation looks good but hardware does not, the next best deploy-side
  question is often:
  - “what real actuator/target shaping is still missing?”
- not immediately:
  - “which reward weight should we change first?”

## Deploy-Side Levers Still Worth Exploring Before Retraining

These are the most defensible remaining deployment levers for the current
policy:

1. Joint-target slew shaping
   - this directly attacks the `policy target -> PD response` abruptness
   - other Go2 deployments use this
   - it is more actuator-proximal than command caps

2. Slight PD retuning or runtime gain scaling
   - only if done carefully and logged clearly
   - especially if one axis or one phase of gait looks over-stiff on real
     hardware

3. Real-speed safety shaping rather than command shaping alone
   - if actual body speed exceeds intended slow teleop behavior, command caps
     are not enough

4. Better real-robot diagnostics
   - segment logs by pure `vx`, pure `vy`, pure `wz`, release
   - compare joint target error and `tau_est` by phase

## Best Current Working Principle

Before retraining, we should try to answer:

“Is there a small, principled, actuator-proximal deployment safeguard that
closes most of the real gap while leaving the frozen policy intact?”

If the answer becomes:

- “yes, and it makes deployment stable”

then retraining can focus on removing that safeguard or internalizing it.

If the answer becomes:

- “no, even the cleanest deploy shaping still leaves obvious stop/coupling
  issues”

then we have strong evidence that retraining is justified and well-scoped.

## Recommended Training Patch Themes Later

When retraining time comes, the clearest candidate themes are:

1. stronger zero-command stabilization
2. stronger cross-axis decoupling
3. stronger action smoothness
4. actuator-aware realism around target tracking and low-speed control

These should be implemented as explicit, documented follow-ups to the
deployment patch log rather than as vague “try different rewards” work.
