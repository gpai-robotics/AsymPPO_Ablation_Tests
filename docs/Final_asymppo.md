# Blind Locomotion Paper Outline

## Working Title

Morphology-Agnostic Reward Shaping and Staged Training for Blind Stair-Climbing Locomotion

## One-Sentence Claim

A single blind locomotion policy can acquire robust stair-climbing behavior, including extrapolation beyond trained stair heights, through staged training and morphology-agnostic reward shaping without explicit terrain sensing at deployment.

## What Must Be True For This To Be Publishable

This project is only interesting as a paper if the contribution is sharper than "the robot climbs stairs."

The likely publishable claim is:

1. A blind policy can learn stair-climbing behavior from proprioception alone.
2. The behavior emerges from a staged training pipeline rather than stair-specific runtime heuristics.
3. The reward shaping is morphology-agnostic and does not encode stair geometry or limb identity.
4. The learned policy extrapolates beyond the stair heights seen during training.

If any of these do not hold under controlled evaluation, the contribution weakens significantly.

## Paper Story

### Problem

Blind locomotion on discontinuous terrain is hard because the policy must recover from failed forward progress and obstacle interaction without explicit terrain observations.

### Hypothesis

A policy does not need explicit stair perception if training shapes a generic recovery behavior:

- maintain commanded progress when locomotion is working
- detect failed progress implicitly through command-tracking mismatch
- prefer lifted forward swing over low-clearance forward bulldozing

### Method

Train a single blind policy with:

1. A staged curriculum:
   - flat prior stage
   - rough terrain stage
   - stair specialization stage
2. A temporal blind actor with history
3. Morphology-agnostic reward terms that encode recovery dynamics rather than terrain geometry

### Main Result

The learned blind policy climbs stairs higher than those seen in training, suggesting it learned a general recovery strategy rather than memorizing a terrain-specific behavior.

## Proposed Paper Structure

## 1. Introduction

Goals:

- Motivate blind locomotion as a deployable setting
- Explain why stair climbing is a strong test of blind robustness
- Position the work against terrain-aware and stair-specific approaches

Key points:

- Explicit terrain sensing is often unavailable, brittle, or expensive
- Stair climbing is a representative failure case for blind locomotion
- Existing solutions often rely on exteroception, scripted heuristics, or morphology-specific shaping
- We show that a single blind policy can acquire stair-climbing behavior using staged training and morphology-agnostic reward shaping

## 2. Related Work

Buckets to cover:

- blind legged locomotion
- stair climbing and rough terrain locomotion
- curriculum learning for locomotion
- morphology transfer / multi-embodiment RL
- reward shaping for recovery behaviors

Questions to answer:

- What is the closest blind stair-climbing baseline?
- Who uses exteroception or height maps?
- Who uses morphology- or terrain-specific reward design?

## 3. Method

### 3.1 Policy

Describe:

- actor input contract
- temporal history encoder
- critic privileged information
- why the actor remains blind while the critic can be privileged

Artifacts to document:

- actor observation groups
- history length and temporal encoder design
- action space and actuator model

### 3.2 Staged Training Pipeline

Document the exact stages:

1. Flat prior
2. Rough terrain adaptation
3. Stair specialization

For each stage, specify:

- initialization source
- terrain distribution
- command ranges
- randomization settings
- reward differences
- termination logic
- checkpoint used to warm-start the next stage

### 3.3 Reward Design Philosophy

State the design rule explicitly:

Morphology-agnostic rewards should use embodiment-neutral signals and avoid:

- limb identity assumptions
- fixed foot clearance targets tied to one robot
- stair height priors
- hand-designed gait templates

Then explain each important reward term:

- tracking rewards
- stable progress
- adaptive swing recovery
- feet slide
- feet air time
- stand-still penalties
- failure terminations

### 3.4 Stair Recovery Reward

This should likely be its own subsection.

Explain that `adaptive_swing_recovery` is not a stair detector. It is a failed-progress recovery term:

- gate on meaningful commanded motion
- gate on poor achieved progress
- penalize low-lift forward distal motion
- reward forward motion coupled with upward swing

This is likely the most novel conceptual piece and should be written carefully.

## 4. Experimental Setup

### 4.1 Training Setup

Include:

- simulator and version
- robot embodiment(s)
- PPO settings
- stage lengths
- hardware and training time

### 4.2 Evaluation Protocol

Must be explicit and reproducible.

Suggested evaluation axes:

- flat ground
- rough terrain
- trained stair range
- larger unseen stair heights
- ascending and descending
- different command magnitudes

### 4.3 Metrics

Core metrics:

- success rate
- timeout rate
- base-height failure rate
- orientation failure rate
- command-tracking error
- max traversable stair height
- traversal speed
- recovery frequency after failed progress

## 5. Main Results

### 5.1 Stair Climbing in Training Range

Show that the stage-3 policy succeeds on trained stair heights.

### 5.2 Extrapolation Beyond Training Range

This is likely the most important section.

If trained up to `12 cm` and successful at `17 cm`, quantify:

- success rate by stair height
- degradation curve
- whether failures are graceful or catastrophic

### 5.3 Behavior Analysis

Use videos and trajectory summaries to show:

- lifted stepping vs bulldozing
- recovery after failed initial contact
- improved body stability over training

## 6. Ablations

This section will determine whether the paper is convincing.

Minimum ablations:

1. Remove stage 3 entirely
   - test whether rough-stage warm start alone climbs stairs

2. Remove `stable_progress`
   - test whether progression becomes unstable or conservative

3. Remove `adaptive_swing_recovery`
   - test whether bulldozing returns

4. Train stairs from scratch
   - compare against staged warm-start

5. Replace morphology-agnostic recovery with a simpler fixed foot-clearance reward
   - compare generalization and extrapolation

6. Reduce history or remove temporal encoder
   - test whether memory is critical for blind stair recovery

Optional but strong:

7. Multi-seed evaluation
8. Different embodiments if available
9. Different stair textures / friction / lighting / disturbances

## 7. Limitations

Be explicit:

- likely evaluated on a limited embodiment set so far
- reward still depends on distal body selection like `.*_foot`
- current evidence may be stronger on ascent than descent
- extrapolation beyond `17 cm` may still fail abruptly
- single-policy claim should be scoped carefully if only one morphology is fully validated today

## 8. Conclusion

Likely message:

Blind stair climbing can emerge from staged training and morphology-agnostic recovery shaping, and the resulting policy exhibits extrapolative behavior beyond the trained stair regime.

## Evidence Checklist

Before writing the paper, collect and freeze:

- exact commit hashes for training code
- exact env and agent config files for every reported run
- stage checkpoints
- evaluation scripts
- videos of success and failure cases
- stair geometry used in training and testing
- seed list for all final results
- train/eval tables exported from logs

## Immediate Next Tasks

1. Freeze the exact stage-3 checkpoint(s) that demonstrate the claim.
2. Build a formal evaluation matrix over stair height, ascent/descent, and command speed.
3. Run ablations for:
   - no `adaptive_swing_recovery`
   - no `stable_progress`
   - no stage-3 fine-tune
4. Record matched videos for:
   - trained stair height
   - extrapolated stair height
   - failure cases
5. Write a concise method note for each training stage before the details get lost.

## Draft Figures

Suggested figures:

1. Training pipeline diagram:
   flat prior -> rough adaptation -> stair specialization

2. Policy architecture:
   blind actor with temporal history, privileged critic

3. Reward concept figure:
   progress failure activates recovery shaping

4. Success vs stair height plot:
   trained range and extrapolation range

5. Ablation bar chart:
   full method vs removed components

6. Frame sequence:
   bulldozing failure vs lifted recovery behavior

## Draft Tables

Suggested tables:

1. Training stages and config deltas
2. Main evaluation results across terrain families
3. Generalization across unseen stair heights
4. Ablation results

## Writing Notes

Avoid overstating the claim. Good wording:

- "blind"
- "single policy"
- "staged training"
- "morphology-agnostic reward shaping"
- "extrapolates beyond trained stair height range"

Avoid claiming without evidence:

- "foundation model"
- "general multi-embodiment policy"
- "universal transfer"

Those can be future directions unless the experiments fully support them.
