# Known Limitations And Branch Follow-Ups

This note records the main limitations, recurring issues, and unresolved
questions observed across the repo so far.

Its job is not to diminish results. Its job is to preserve signal:

- what is already solid
- what keeps recurring
- what is only a hypothesis right now
- what later branches should decide whether to fix

Use this as a branch-planning note, not as a replacement for phase-specific
freeze docs.

Related future-branch comparison note:

- `docs/ETH_ANYMAL_GAP_NOTES.md`

## Active Follow-Up Order

The current follow-up order is now intentionally narrow:

1. keep `adapt_v3_dyn_only_phase2_stage_a_final.pt` as the canonical dyn-only
   student
2. build the matching terrain-aware student
3. compare both under the same eval battery
4. choose the deployable candidate from that comparison

Interpretation rule:

- frozen dyn-only `Stage A` artifacts are milestone anchors
- the failed mixed/switch dyn-only branch is retained only as negative evidence
  in docs, not as active code
- switched hidden-dynamics changes remain an evaluation stress test rather than
  the primary training contract
- Sim2Sim and deployment preparation should wait for a policy that proves both
  rough-terrain competence and strong per-episode randomized robustness

## Retired Adapt-V3 Exploration

We explicitly tried a dyn-only `Adapt-V3` Phase 2 continuation with
mid-episode hidden-dynamics switching during training.

Outcome:

- the recipe did not hold
- both `25%` and `10%` switch-probability continuations degraded global
  locomotion instead of converging to a stronger adaptive student
- this failure happened even before switched episodes became the dominant share
  of training

Current interpretation:

- this is useful negative evidence, not the active forward path
- the reference `rl_locomotion` recipe randomizes hidden dynamics at reset and
  holds them fixed through the episode
- the project should now return to the standard per-episode domain
  randomization contract for student training

## Scope

This note covers the project trajectory through:

- flat prior
- blind baselines
- privileged teacher phase
- adaptation `NA`
- adaptation `V0`
- adaptation `V1`
- adaptation `V2`
- early `Adapt-V3` Phase 1 bring-up

Truth status:

- findings tied to frozen checkpoints and canonical eval artifacts should be
  treated as repo truth
- `V3` observations are still provisional until that branch is frozen

## Cross-Cutting Limitations

### 1. Body stability is still not a solved axis

This is the cleanest recurring limitation in the repo so far.

Across multiple strong candidates, we still repeatedly see:

- nontrivial `base_height` failures
- some `base_orientation` failures
- policies that can survive and adapt without looking fully composed

What this means:

- robustness and adaptation improved
- posture quality and body composure did not fully keep pace

This should be treated as a distinct future branch, not folded casually into
 every adaptation discussion.

### 2. Robustness and gait quality may be in tension

The switched adaptation regime likely improves hidden-factor robustness, but it
may also bias the learned gait toward more conservative or compromised motion.

Observed concerns:

- stretched fore-aft stance pattern
- cautious or over-damped behavior
- plausible link between robustness-seeking gait changes and body-height issues

Current repo stance:

- this is a credible hypothesis
- not yet a proven causal result

Suggested future test:

- compare stationary-trained and switch-trained policies on nominal gait/body
  metrics before and after the switch point

### 3. The current project optimizes adaptation more than elegance

The current branch answers:

- does hidden-factor adaptation help?

It does **not** yet fully answer:

- what produces the cleanest, most stable, most natural locomotion style?

That distinction should remain explicit in public and research framing.

### 4. Sim-to-real is still not proved

The repo now supports a strong simulation-side story, but it does not yet prove:

- hardware success
- deployment robustness in the real world
- superiority over the stock Unitree controller on robot

Deployment remains a separate validation phase.

## Phase-Specific Lessons

### Flat Prior

What worked:

- useful initialization anchor for later blind and adaptation branches

Limitations:

- not itself the main research result
- should not be overinterpreted as a deployable rough-terrain controller

Future use:

- continue to treat it as a prior / initializer, not as a competing final
  locomotion result

### Blind Baselines

What worked:

- established strong fixed-controller baselines
- warm start and temporary imitation clearly helped baseline quality

Limitations:

- blind policies can become strong enough that adaptation gains are not
  guaranteed to be large
- strong blind baselines make later adaptation results scientifically stronger,
  but they also make wins harder to achieve
- blind baselines should not be turned into endlessly retuned obstacle
  specialists

Frozen lesson:

- baseline strength is a feature of the methodology, not a problem

### Privileged Teacher Phase

What worked:

- produced a meaningful privileged upper-bound lineage
- gave the adaptation phase a serious training target

Limitations:

- a strong privileged teacher does not automatically imply a clean deployable
  adaptation path
- later adaptation branches still had to prove themselves

Important conceptual limitation:

- teacher internal features are useful training targets, but they are not
  automatically equivalent to a principled environment-extrinsics latent

That limitation became central when moving from `V1`/`V2` toward true `V3`.

### Adaptation `NA`

What worked:

- proved that a deployable proprio-only student can still be strong under the
  switched regime
- gave the adaptation branch a serious no-adaptation baseline

Limitations:

- because `NA` is strong, adaptation must earn its gain honestly
- the project cannot assume history-based adaptation is necessary without
  evidence

Frozen lesson:

- `NA` is one of the most important controls in the whole repo

### Adaptation `V0`

What worked:

- first clear positive adaptation result over `NA`
- justified the adaptation branch with a completed empirical win

Limitations:

- adaptation path is effective, but still not architecturally clean in the
  original RMA sense
- result is positive but not a blowout
- the mechanism is still closer to guided imitation-based adaptation than to a
  clean latent-contract implementation

Frozen lesson:

- `V0` is the first empirical adaptation win
- `V0` is not the final architecture answer

### Adaptation `V1`

What worked:

- explicit latent prediction trains to completion
- became a real mature adaptation result
- strong survival / contact-quality profile

Limitations:

- translational tracking stayed less sharp than ideal for much of training
- the predicted latent is still a teacher-side internal feature target, not a
  fully explicit extrinsics encoder target
- architecture remained bundled even though the latent supervision was explicit

Frozen lesson:

- explicit latent regression is viable
- but explicit latent supervision alone does not yet equal full original-style
  RMA

### Adaptation `V2`

What worked:

- proved that the modular split can be implemented and trained cleanly
- established a real architectural milestone

Limitations:

- `V2` did not produce a new empirical result over `V1`
- canonical eval outputs are identical to `V1`
- modularization alone did not change the actual frozen outcome in this repo

Frozen lesson:

- modular decomposition is architecturally meaningful
- but modularization by itself is not enough to guarantee a better policy

### Adapt-V3 Phase 1

Current status:

- the earlier terrain-plus-dynamics `V3` line is now treated as exploratory
  lineage
- the active `V3` reboot now has a frozen dynamics-only `Stage A` base
- the older frozen `Stage A` and `Phase 2 Stage A` artifacts remain useful
  barrier-crossing records, but they are no longer the active implementation
  contract

What looks promising so far:

- the stricter `mu + pi` bottleneck is trainable
- a stationary `Stage A` with critic-only warm-start and temporary blind-policy
  imitation successfully recovered real locomotion
- `Stage A` also preserved real latent use under debug checks, so locomotion
  recovery did not come from a hidden latent bypass

Current limitations and open questions:

- the branch only became viable after introducing staged training and a
  locomotion scaffold, which means pure from-scratch switched `V3` Phase 1
  should currently be treated as too hard in this repo
- the biggest unresolved design question is whether a blind history student can
  realistically infer terrain geometry well enough to justify terrain privilege
  inside `mu`
- body stability still looks like the same slow-moving challenge seen earlier
- because of that mismatch concern, the active reboot now narrows `mu` to
  hidden dynamics only and treats terrain geometry as the deferred question

Branch importance:

- `V3` is the first branch likely to produce a genuinely different adaptation
  result, because the latent contract is finally explicit from the start

Frozen lesson to preserve already:

- the main `V3` bottleneck was crossed by staged training:
  - `Stage A` locomotion-first
  - critic-only warm-start
  - temporary blind-policy imitation
  - explicit latent validation
- this should be treated as a project evolution checkpoint, not as ad hoc
  tuning noise

### Adapt-V3 Phase 2

Current status:

- the earlier terrain-inclusive `Phase 2 Stage A` freeze is preserved as a
  historical bootstrap success
- it is not the active forward path anymore after the mixed continuation
  failed behaviorally

What looks promising so far:

- direct `Phase 2` training under full switched pressure was too harsh
- a stationary `Phase 2 Stage A` with temporary teacher-action scaffold
  preserved locomotion while training `phi(history)`
- the history-path bootstrap survived all the way until teacher imitation
  decayed to zero

Current limitations and open questions:

- mixed continuation showed that latent matching can improve while closed-loop
  behavior still collapses
- that failure suggests the remaining bottleneck may be target mismatch, not
  just optimization difficulty
- the active next step is therefore a dynamics-only reboot, not more tuning of
  the terrain-inclusive `Phase 2 Mixed` recipe

Frozen lesson to preserve already:

- `Phase 2` appears to need the same kind of staging that `Phase 1` needed
- behavior-preserving history bootstrap may be a necessary repo-specific
  adaptation of the original RMA training story

## Methodological Risks To Keep Visible

### 1. Mid-episode switch pressure may distort nominal gait

This is one of the most important active concerns.

Possible mechanism:

- policy expects future hidden change
- policy learns a safer compromise gait all the time
- compromise gait may help survival but hurt composure and body support

Potential downstream effects:

- stretched stance geometry
- conservative gait timing
- body-height issues
- less elegant nominal walking

Current status:

- plausible and important
- not yet conclusively isolated

### 2. Teacher-feature latents are useful, but not the same as explicit extrinsics

This limitation applies especially to `V1` and, indirectly, to the transition
before `V3`.

What we learned:

- teacher-side hidden features can be a practical target

What remains true:

- they are not automatically interpretable as environment state
- they are not a full substitute for `mu(e_t) -> z_t`

This is one of the main reasons `V3` became necessary.

### 3. Stronger architecture does not guarantee stronger results

The repo already showed this with `V2`:

- cleaner architecture
- real modular milestone
- no new frozen empirical gain over `V1`

This should keep us honest in later branches.

## Suggested Future Branches

These are the most justified follow-up directions after the current main line.

### 1. True RMA branch

Goal:

- complete the original-style latent contract cleanly

Current designated branch:

- `Adapt-V3`

### 2. Body-stability branch

Goal:

- directly target base-height, posture, and gait composure quality

Why it deserves its own branch:

- the issue is persistent
- it cuts across blind, adaptation, and now early `V3`
- mixing it into every branch muddies scientific interpretation

### 3. Switch-regime ablation branch

Goal:

- test whether the mid-episode switch regime is distorting gait and posture

Likely experiments:

- stationary-only vs switched-only vs mixed-regime training
- pre-switch vs post-switch behavior analysis
- gait geometry / slip / body-height comparison

Specific follow-up hypothesis to preserve:

- the current all-or-mostly-switched training regime may be encouraging a
  compromise gait that helps reward under hidden future disruption but hurts
  nominal body support
- a mixed training regime may preserve adaptation pressure while reducing
  stretched stance behavior and improving `base_height` / posture outcomes

Named candidate mixes:

- `40/60` stationary/switched
  - first balanced ablation to try
  - keeps switch pressure load-bearing while giving some stationary anchor
- `70/30` stationary/switched
  - stronger nominal-gait anchor
  - useful second test if `40/60` is still too compromise-heavy

Truth status:

- this is currently a branch-planning hunch, not a validated repo conclusion
- it is worth testing because it is a plausible explanation for the recurring
  `base_height` and posture issues

### 4. Deployment branch

Goal:

- shift focus from architecture search to hardware validation

This should happen after the architecture question is settled well enough.

## Branch-Priority Heuristic

When choosing what to fix next, use this order:

1. fix methodological ambiguity that threatens interpretation
2. fix architectural gaps that block the intended final contract
3. fix repeated behavior-quality issues that span many branches
4. defer aesthetic or marginal cleanup until the above are settled

Right now, that means:

1. finish `V3`
2. decide whether the switch regime is biasing gait in a harmful way
3. open a dedicated body-stability branch
4. move toward deployment

## Short Summary

The repo is now strong enough that the main unresolved issues are no longer
"can we build anything that works?"

They are:

- can we finish the true RMA contract cleanly?
- how much is the switch regime shaping strange gait behavior?
- how do we solve body stability without muddying the adaptation story?
- which issues belong to later dedicated branches rather than this one?
