# Adaptive Sim2Sim Refinement Plan

This note defines the next refinement stage for the adaptive `Adapt-V3` line
after the first bounded-latent recovery freeze.

Its purpose is narrow:

- explain the remaining adaptive-branch MuJoCo gap clearly
- define the most justified next training-side fixes
- define the exact evaluation gate for deciding whether a new run is better

This note should be read together with:

- `docs/SIM2SIM_STAGEA_VS_ADAPTIVE_COMPARISON.md`
- `docs/ADAPT_V3_ACTIVE_ROADMAP.md`
- `docs/ETH_ANYMAL_GAP_NOTES.md`
- `docs/DEPLOYMENT_AUDIT_ADAPT_V3.md`

## Current Starting Point

We now have three important frozen dyn-only `Adapt-V3` artifacts:

### Deployment-side winner

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`

Meaning:

- strongest current deployment-side checkpoint
- keeps a clean alternating gait in MuJoCo
- proves the bridge is no longer the main blocker

### Adaptation-recovery anchor

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`

Meaning:

- first canonical recovery artifact that restored real online adaptation
- still too fragile in MuJoCo due to latent blow-up

### Bounded-latent recovery refinement base

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

Meaning:

- first frozen training-side repair that materially improves unclamped MuJoCo
  behavior for the adaptive branch
- best current refinement base for the next adaptive run

## What We Now Know

The Stage A vs adaptive runtime-trace comparison narrowed the problem a lot.

What is no longer the main hypothesis:

- generic export failure
- generic MuJoCo bridge failure
- generic deploy-config mismatch

Why:

- the stationary Stage A winner walks well in MuJoCo
- the adaptive bounded-latent branch is the one that still degrades

So the current bottleneck is more specific:

- the adaptive branch remains fragile under cross-engine history mismatch

## Working Failure Mechanism

The current best explanation is:

1. Isaac and MuJoCo produce meaningfully different proprioceptive history
   statistics
2. `phi(history)` in the adaptive branch is more sensitive to that shift than
   the stationary winner
3. latent drift then pushes the actor toward a conservative, sticky stepping
   pattern
4. the policy retains locomotion, but loses grace, diagonal regularity, and
   contact symmetry

The earlier recovery branch failed through catastrophic latent runaway.

The bounded-latent branch improved that story substantially:

- no longer immediate action explosion
- materially better unclamped MuJoCo behavior
- still too much latent drift and too much sticky stance support

So the next problem is not "prevent total blow-up" anymore.

The next problem is:

- reduce residual latent drift enough that the adaptive branch can keep a
  cleaner MuJoCo gait

## Evidence For That Mechanism

From `docs/SIM2SIM_STAGEA_VS_ADAPTIVE_COMPARISON.md`:

### Stage A MuJoCo

- `reward_proxy_mean = 0.411`
- `vel_err_step_mean = 0.155`
- `yaw_err_step_mean = 0.082`
- `base_height_mean = 0.332`
- diagonal support:
  - `FL+RR = 0.267`
  - `FR+RL = 0.262`
- all-4 contact:
  - `0.079`
- latent norm:
  - mean `7.007`
  - max `7.007`

### Bounded-latent adaptive MuJoCo

- `reward_proxy_mean = 0.302`
- `vel_err_step_mean = 0.267`
- `yaw_err_step_mean = 0.163`
- `base_height_mean = 0.267`
- diagonal support:
  - `FL+RR = 0.116`
  - `FR+RL = 0.049`
- all-4 contact:
  - `0.558`
- latent norm:
  - mean `25.767`
  - max `121.183`

Interpretation:

- the adaptive branch is not dead
- the adaptive branch is not merely slightly rougher
- the adaptive branch is still carrying a very specific latent-and-contact
  mismatch in MuJoCo

## What The Next Branch Should Optimize For

The next adaptive refinement branch should optimize for all of these at once:

1. preserve real online adaptation pressure
2. preserve Isaac locomotion quality
3. reduce MuJoCo latent drift
4. reduce sticky all-feet-down stance behavior
5. increase diagonal support occupancy in MuJoCo

This is important:

- the next branch should not be treated as "just make MuJoCo look calmer"

It must remain an adaptive branch, not drift back into a stationary winner that
happens to export nicely.

## Refinement Order

Use one-change-at-a-time refinement, not a bundle of speculative changes.

### Stage R1: Stronger latent boundedness

Start from:

- `adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

Most justified first interventions:

1. keep the current latent L2 penalty
2. add a coordinate-wise magnitude control term
3. optionally add a softly bounded latent output head

Concrete options:

- max-abs penalty on `z_hat`
- Huber-like penalty above a small coordinate threshold
- final `tanh` scaling on the adaptation output head

Why this comes first:

- the current observed failure is still latent drift
- this is the closest training-side intervention to the measured problem
- it changes less at once than moving immediately to a new sequence model

### Stage R2: Temporal smoothness of the latent

If stronger boundedness alone is not enough, add:

- penalty on `||z_hat_t - z_hat_t-1||`

Purpose:

- reduce jittery latent moves under slight simulator-history mismatch
- encourage smoother inferred hidden-state trajectories

Important constraint:

- this should be weak enough that real switch response remains possible

So this is not "make the latent constant."
It is:

- reduce spurious drift
- preserve meaningful change when hidden dynamics really shift

### Stage R3: Better sequence inductive bias in `phi(history)`

If R1 and R2 still leave a large MuJoCo gait gap, the next justified
architecture change is:

- replace flattened-history MLP `phi` with a TCN-style encoder

This is the most justified architecture change from
`docs/ETH_ANYMAL_GAP_NOTES.md`.

Why it is justified:

- the current adaptive branch is history-sensitive
- contact timing and slip response are sequence problems
- ETH-style temporal inductive bias is one of the clearest repo gap hypotheses

Why it should not be first:

- it is a larger change
- it makes attribution harder if done too early
- we have not yet exhausted simpler latent-stability repairs

### Stage R4: Curriculum or gait-retention help

Only after the latent-side fixes are understood should we widen to:

- difficulty-band terrain control
- stronger gait-retention shaping
- motion-prior style help

These may become important later, but they are not the first clean answer to
the current measured failure.

## Suggested Next Experimental Branches

### Branch 1: Latent boundedness plus max-abs control

Minimal change set:

- current latent L2 branch
- add a max-abs or thresholded coordinate penalty

Question:

- can we reduce MuJoCo latent drift while keeping the recovery branch's
  adaptation strength?

Result:

- no, not as a net improvement

Observed branch outcome:

- Isaac checkpoint selection within the branch favored `model_300.pt`
- the exported max-abs MuJoCo candidate was materially worse than the earlier
  bounded-latent `model_220.pt` challenger when run unclamped
- the clamp-5 diagnostic still helped, but the branch did not beat the earlier
  bounded-latent branch even there

Key MuJoCo comparison:

- earlier bounded-latent `model_220` unclamped:
  - `reward_proxy_mean = 0.305`
  - `vel_err_step_mean = 0.263`
  - `yaw_err_step_mean = 0.164`
  - `base_height_mean = 0.266`
  - `latent_norm_mean = 25.517`
  - `latent_norm_max = 121.183`
- max-abs `model_300` unclamped:
  - `reward_proxy_mean = 0.170`
  - `vel_err_step_mean = 0.564`
  - `yaw_err_step_mean = 0.661`
  - `base_height_mean = 0.088`
  - `latent_norm_mean = 149.931`
  - `latent_norm_max = 1196.138`

Interpretation:

- the coordinate-wise max-abs refinement did not stabilize the adaptive branch
  in MuJoCo
- the branch appears to have made the selected candidate more brittle under the
  true deployment-side history shift
- this branch should be recorded as a negative refinement result, not promoted
  as the new adaptive challenger

Operational decision:

- keep
  `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`
  as the active adaptive MuJoCo challenger
- retire the current max-abs branch as a non-winning refinement attempt
- move to the next refinement idea rather than polishing this branch further

### Branch 2: Latent boundedness plus temporal smoothness

After Branch 1 failed as a net MuJoCo improvement, the next justified low-to-mid
complexity step is:

- add a weak temporal latent-delta penalty

Question:

- can we reduce sticky MuJoCo stepping by calming short-timescale latent noise?

Result:

- partly, but not enough as a net MuJoCo improvement

Observed branch outcome:

- Isaac checkpoint selection within the branch favored `model_100.pt`
- the exported smooth-branch MuJoCo candidate was materially worse than the
  earlier bounded-latent `model_220.pt` challenger when run unclamped
- the clamp-5 diagnostic helped substantially and produced a usable policy
  again, but the branch still did not beat the earlier bounded-latent branch

Key MuJoCo comparison:

- earlier bounded-latent `model_220` unclamped:
  - `reward_proxy_mean = 0.305`
  - `vel_err_step_mean = 0.263`
  - `yaw_err_step_mean = 0.164`
  - `base_height_mean = 0.266`
  - `latent_norm_mean = 25.517`
  - `latent_norm_max = 121.183`
- smooth `model_100` unclamped:
  - `reward_proxy_mean = 0.112`
  - `vel_err_step_mean = 0.510`
  - `yaw_err_step_mean = 0.307`
  - `base_height_mean = 0.099`
  - `latent_norm_mean = 61.644`
  - `latent_norm_max = 1096.156`
- smooth `model_100` clamp-5:
  - `reward_proxy_mean = 0.337`
  - `vel_err_step_mean = 0.229`
  - `yaw_err_step_mean = 0.179`
  - `base_height_mean = 0.297`
  - `latent_norm_mean = 16.556`
  - `latent_norm_max = 27.236`

Interpretation:

- weak temporal smoothing improved the max-abs refinement outcome
- but it still did not solve the main problem, which is good unclamped adaptive
  behavior under deployment-side history mismatch
- this branch should be recorded as a non-winning refinement result, not
  promoted as the new adaptive challenger

Operational decision:

- keep
  `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`
  as the active adaptive MuJoCo challenger
- retire the current smoothness branch as a non-winning refinement attempt
- move to the next refinement idea rather than polishing this branch further

### Branch 3: TCN `phi(history)`

If the bounded-and-smoothed MLP path still underperforms badly in MuJoCo:

- switch the adaptation encoder to a temporal convolutional history model

Question:

- can sequence inductive bias close the gait-quality gap without giving up the
  explicit RMA-style latent contract?

## What Should Not Change Yet

To keep the next results interpretable, avoid changing all of these at once:

- reward stack
- terrain stack
- latent target definition
- policy observation definition
- full actor architecture
- deployment bridge contract

The bridge is now good enough to serve as an evaluator.
We should use that advantage.

## Exact Evaluation Gate

Every new adaptive refinement candidate should pass through the same four-part
gate.

### Gate A: Isaac training health

During training, require:

- non-collapsed student latent
- rising latent cosine
- non-trivial adaptation switch activity
- no obvious locomotion collapse

Track at minimum:

- `latent_cosine`
- `latent_regression`
- `student_latent_batch_std`
- `student_latent_l2`
- `student_latent_max_abs`
- reward
- episode length
- `adaptation_switch_applied_frac`

### Gate B: Isaac checkpoint selection

Do not assume the final checkpoint is best.

For each run:

- checkpoint-sweep the early-to-mid window
- use gait screen
- then `blind_baseline_v1`
- then `ood_switch_v1`

The current bounded-latent run showed that early-mid selection remains the
right discipline.

### Gate C: Isaac vs MuJoCo runtime trace comparison

For the best adaptive checkpoint, produce:

- Isaac full trace via `scripts/eval/trace_isaac_policy.py`
- MuJoCo full trace via `scripts/deploy/run_sim2sim.py`
- comparison report via `scripts/eval/compare_runtime_traces.py`

This comparison is now mandatory for adaptive branches.

### Gate D: Progress thresholds

Use the current bounded-latent artifact as the minimum baseline and the Stage A
winner as the stretch target.

#### Minimum improvement gate over current bounded-latent adaptive

A new adaptive branch should not be considered a real improvement unless it
beats the current bounded-latent artifact on most of these MuJoCo metrics:

- `reward_proxy_mean > 0.302`
- `vel_err_step_mean < 0.267`
- `yaw_err_step_mean < 0.163`
- `base_height_mean > 0.267`
- `base_tilt_projected_gravity_xy_mean < 0.103`
- `all_4_contact_fraction < 0.558`
- combined diagonal support fraction
  `FL+RR + FR+RL > 0.165`
- `latent_norm_mean < 25.767`
- `latent_norm_max < 121.183`

#### Practical promotion gate

To count as a serious adaptive promotion candidate, target roughly:

- `reward_proxy_mean >= 0.35`
- `vel_err_step_mean <= 0.22`
- `base_height_mean >= 0.30`
- `all_4_contact_fraction <= 0.30`
- combined diagonal support fraction `>= 0.30`
- `latent_norm_mean <= 15`
- `latent_norm_max <= 40`

Interpretation:

- this does not require matching Stage A immediately
- but it does require visibly closing the gap

#### Stretch target

The true long-term goal is to approach the Stage A MuJoCo pattern while
keeping the adaptive-branch gains:

- Stage A `reward_proxy_mean = 0.411`
- Stage A `vel_err_step_mean = 0.155`
- Stage A combined diagonal support `= 0.529`
- Stage A all-4 contact `= 0.079`

## Recommended Next Run

The cleanest next run is:

- warm-start from
  `adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`
- keep the current low-switch recovery task family
- add temporal smoothness before paying the complexity cost of a new sequence
  model
- do not change the bridge or the eval protocol

This gives the next branch the best chance of answering one question cleanly:

- can we close more of the MuJoCo adaptive gap with a smoother latent
  trajectory, before paying the complexity cost of a new sequence model?

## Repo-Level Interpretation

The current project story should now be:

- Stage A solved the deployment-side bridge question well enough
- the first recovery branch solved the "is adaptation real?" question
- the bounded-latent branch solved the "can training-side repair improve
  unclamped MuJoCo?" question
- the max-abs branch answered a narrower follow-up negatively:
  - "a stronger coordinate-wise max-abs penalty alone is not the right next
    repair"
- the next branch should solve:
  - "can the adaptive branch approach Stage A MuJoCo gait quality without
    losing its recovered adaptation identity?"
