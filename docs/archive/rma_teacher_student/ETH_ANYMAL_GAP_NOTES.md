# ETH ANYmal Blind Locomotion Gap Notes

This note records what the ETH/Intel ANYmal rough-terrain locomotion line
appears to do differently from the current project, and which parts are worth
testing as a future branch.

Primary reference:

- Lee, Hwangbo, Wellhausen, Koltun, Hutter
- "Learning Quadrupedal Locomotion over Challenging Terrain"
- Science Robotics 2020

This is not a claim that their setting is identical to ours.
It is a structured gap map:

- what they optimized for
- what we optimized for
- what they may be getting "for free" from those choices
- which branch ideas are worth testing later

## High-Level Interpretation

The ETH controller is not best understood as "a better explicit RMA latent."

It is better understood as:

- a very strong blind reactive locomotion stack
- trained with privileged teacher-student supervision
- driven by proprioceptive history through a temporal model
- stabilized by motion priors and actuator modeling
- strengthened by an adaptive terrain curriculum

That is a different optimization target from our explicit `mu(e) -> z`,
`phi(history) -> z_hat`, `pi(x, z)` decomposition.

Our current project has spent more effort on:

- explicit latent structure
- explicit online hidden-factor adaptation
- deployment-contract clarity

Their project appears to spend more effort on:

- raw blind reactive robustness
- sequence-model inductive bias
- terrain curriculum quality
- motion prior quality

This distinction matters.

If the goal is "best blind rough-terrain locomotion," their recipe may be
easier to optimize than a fragile explicit latent bottleneck.

If the goal is "RMA-style online adaptation under changing hidden dynamics,"
our explicit latent path is still scientifically meaningful, but it is a harder
optimization problem.

## What They Seem To Do Better

### 1. Strong temporal student architecture

Their student is not a flat MLP over stacked history.

It is a temporal convolutional network over proprioceptive history.

Likely advantage:

- better inductive bias for contact events
- better inductive bias for slip and disturbance reactions
- easier implicit state estimation from history

Current project gap:

- our current `Adapt-V3` student path still relies on an MLP over flattened
  history
- this is weaker as a sequence model, even if the total observation content is
  similar
- and the frozen `Teacher V3` checkpoint has now been audited to use hidden
  dynamics privilege much more clearly than terrain privilege, which weakens
  any claim that current teacher-student training is already inheriting
  terrain-reactive structure in the ETH sense

Future branch implication:

- test a TCN-based `phi(history)` encoder
- possibly also test a direct history-to-action student without explicit latent
  bottleneck as a robustness-oriented comparison branch

### 2. More deliberate terrain curriculum

Their terrain generation is not only "domain randomization plus terrain level."

It actively searches for terrains in a medium-difficulty band:

- hard enough to teach the policy
- not so hard that learning signal collapses

Likely advantage:

- less wasted rollout budget on impossible terrains
- smoother skill growth
- better robustness without brute-force curriculum escalation

Current project gap:

- our rough-terrain training uses useful curricula, but not the same degree of
  explicit difficulty-band maintenance
- recovery branches currently rely more on carefully chosen training settings
  than on a dynamic terrain-difficulty controller

Future branch implication:

- test a difficulty-tracking terrain curriculum for blind/recovery branches
- especially for branches where switch pressure and gait quality compete

### 3. Strong motion priors

Their controller is not synthesizing locomotion from a fully unconstrained
 action head.

It modulates structured motion primitives:

- FTG / PMTG-style stepping prior
- frequency and phase modulation
- residual foot placement

Likely advantage:

- easier optimization
- stronger gait regularity
- better transfer
- less burden on the policy to invent stepping structure

Current project gap:

- our current stack is stronger on policy-contract clarity than on explicit
  locomotion priors
- if gait quality and adaptation compete, stronger motion priors may reduce
  that tradeoff

Future branch implication:

- test stronger locomotion priors in the blind student
- especially if later recovery branches gain adaptation but lose composure

### 4. They optimize reactive robustness more than explicit latent cleanliness

Their student does not seem to depend on proving an interpretable latent that
changes online in a clean RMA sense.

Instead, the history policy itself becomes the main adaptive mechanism.

Likely advantage:

- fewer collapse modes
- fewer places where the actor can ignore or misuse a latent bottleneck
- better end-to-end optimization for "survive and move"

Current project gap:

- our explicit latent pathway has already shown several failure modes:
  - `phi` collapse
  - actor using latent as a fixed bias
  - switch pressure harming gait before adaptation becomes strong

Future branch implication:

- keep the explicit latent line for the scientific RMA objective
- but also consider a robustness-first blind-history branch that does not force
  the full explicit latent contract

### 5. Better actuator/control realism discipline

The paper stresses actuator modeling and deployment-faithful control structure.

Likely advantage:

- smoother sim-to-real
- more stable training
- less compensatory policy behavior

Current project status:

- we now have a serious deployment bridge and a real MuJoCo line
- but the architecture itself was not designed around the same kind of
  locomotion-specific motion-generator stack

Future branch implication:

- preserve the deployment path work we now have
- use it to evaluate later architecture changes instead of treating deployment
  as an afterthought

## What They May Not Be Solving That We Care About

It is important not to overread their result in the other direction.

Their paper is extremely strong on blind rough-terrain locomotion, but it is
not primarily a proof of explicit online hidden-factor adaptation in the RMA
sense.

So we should not casually conclude:

- "their result makes our explicit latent objective obsolete"

A better interpretation is:

- they likely demonstrate a stronger recipe for blind reactive rough-terrain
  locomotion
- we are trying to additionally preserve an explicit adaptation mechanism

Those are related but not identical goals.

## Current Repo Gap Map

From this paper, the most plausible future-branch gaps in our repo are:

1. sequence modeling
- test TCN-based history encoding

2. curriculum quality
- test active medium-difficulty terrain selection instead of only coarse
  curriculum escalation

3. motion priors
- test stronger locomotion priors / trajectory-generator modulation

4. objective framing
- separate "robust blind locomotion" branches from "explicit online adaptation"
  branches more cleanly

5. comparison discipline
- compare explicit-latent branches against robustness-first history policies,
  not only against weaker baselines

## Suggested Future Branches

These are the most justified follow-up experiments inspired by the ETH line.

### Branch A: TCN `phi(history)`

Keep the current explicit latent architecture, but replace the flattened-history
MLP encoder with a temporal convolutional encoder.

Question:

- can we preserve the RMA-style latent contract while gaining better sequence
  inductive bias?

### Branch B: Blind reactive history student

Train a teacher-student blind history controller that directly consumes
proprioceptive history and outputs action without the explicit `z_hat`
bottleneck.

Question:

- how much of the ETH-style robustness can we recover if we optimize directly
  for blind reactive performance instead of explicit adaptation structure?

This branch should be treated as:

- robustness-first comparison
- not a replacement for the explicit-latent line

### Branch C: Adaptive terrain curriculum refinement

Add a more deliberate terrain-difficulty tracking mechanism that keeps sampled
terrains in a "hard but survivable" band.

Question:

- can we improve robustness and adaptation pressure without breaking gait
  quality?

### Branch D: Stronger motion-prior student

Introduce a trajectory-generator or residual-foot-placement structure for
blind-student policies.

Question:

- can stronger priors preserve posture quality and reduce body-failure modes
  while keeping adaptation alive?

## Operational Rule

Use this note as a future-branch seed list, not as a retrospective criticism of
the current line.

The current repo already answered important questions:

- deployment path
- blind-student quality
- explicit latent failure modes
- recovery path for real switch-aware adaptation

This ETH comparison helps identify what a later robustness-first branch should
borrow or test more directly.
