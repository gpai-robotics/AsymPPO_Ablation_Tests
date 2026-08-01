# Adapt-V3 Active Roadmap

This is the active planning note for the current `Adapt-V3` work.

Use this file first when the question is:

- what is frozen already?
- what is the active branch right now?
- what comes next?
- which ideas are milestones versus research follow-ups?

This file is intentionally narrower than `docs/ADAPT_V3_EXECUTION_SPEC.md`.
The execution spec explains the architecture and barrier record. This roadmap
states the current repo contract.

Note:

- parts of this roadmap describe older pre-structured adaptive lanes for
  historical context
- for the current working structured C2 path, first read:
  - `docs/C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md`
  - `docs/C2_STRUCTURED_Z27_REBUILD_PLAN.md`
  - `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.md`
- for a future cleaner restart branch, read:
  - `docs/C2_RMA_RESTART_BLUEPRINT.md`

Companion active docs:

- `docs/FINAL_CANDIDATE_COMPARISON_RUBRIC.md`
- `docs/DEPLOYMENT_AUDIT_ADAPT_V3.md`
- `docs/ADAPTIVE_SIM2SIM_REFINEMENT_PLAN.md`
- `docs/ADAPTIVE_POLICY_EVAL_PROTOCOL.md`
- `docs/TWO_FINAL_CANDIDATES_ROADMAP.md`
- `docs/OG_RMA_VS_REPO_DIVERGENCE.md`
- `docs/OG_RMA_VS_REPO_DIVERGENCE_FLOWCHART.md`
- `docs/ARCHITECTURE_FLOW_FROM_FLAT_TO_ADAPTV3.md`

## Current Status

We now have:

- two finalized blind-student `Adapt-V3` comparison candidates
- one clear deployment-side winner from the first full head-to-head comparison
- one canonical low-switch recovery artifact that restores real adaptation
  pressure more faithfully than the stationary Stage A winner
- one canonical bounded-latent recovery challenger that materially improves
  unclamped MuJoCo behavior relative to the first recovery artifact
- one completed max-abs refinement attempt that did not beat the bounded-latent
  challenger in MuJoCo
- one completed temporal-smoothness refinement attempt that improved the
  max-abs result but still did not beat the bounded-latent challenger in
  MuJoCo
- one completed structured offline C2 pipeline that now serves as the canonical
  working path for structured adaptation

Frozen comparison candidates:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`
- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase2_stage_a_final.pt`

Current winner:

- `adapt_v3_dyn_only_phase2_stage_a_final.pt`

Canonical recovery artifact:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`

Canonical bounded-latent recovery challenger:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

Canonical working structured offline C2 artifact:

- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`

Supporting terrain-aware parent artifact:

- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.pt`

Meaning:

- dyn-only:
  - current best deployment-side candidate
- dyn-only recovery:
  - current best recovered adaptation-side checkpoint
  - not yet the automatic deployment replacement
  - canonical reference for future switch-aware continuation work
- dyn-only bounded-latent recovery:
  - current best Sim2Sim-oriented recovery refinement checkpoint
  - first frozen training-side fix for MuJoCo latent brittleness
  - still not the automatic deployment replacement
- dyn-only max-abs refinement:
  - evaluated and rejected as a new MuJoCo leader
  - retained only as a documented negative refinement result
- dyn-only smooth refinement:
  - evaluated and rejected as a new MuJoCo leader
  - retained only as a documented non-winning refinement result
- dyn-only TCN anchored refinement:
  - evaluated and retired as another non-winning follow-up
  - improved early stability relative to the first unanchored TCN branch
  - still reproduced the same late pattern:
    better latent alignment, worse locomotion
- dyn-only DAgger-style `phi` follow-up:
  - evaluated as a useful partial follow-up after the TCN retirements
  - freezes the locomotion trunk and removes PPO policy pressure on `phi`
  - improved the branch's mid-training window and produced a decent
    `model_300.pt` checkpoint
  - still kept the same broad gait family and still decayed later in training
  - did not replace the bounded-latent recovery baseline
- terrain-lite recovery DAgger-style follow-up:
  - evaluated as the first direct dyn+terrain reopening after the dyn-only
    DAgger-style partial success
  - keeps the more conservative Phase 2 training recipe from the dyn-only
    DAgger-style branch
  - expands the privileged latent back to terrain-lite + dynamics
  - confirmed that the old terrain-lite crouch / posture-fragility problem
    still returns
  - did not replace the bounded-latent dyn-only baseline
- dyn-only offline supervised `phi` restart:
  - evaluated after the DAgger-style branch and terrain-lite reopening
  - cleanly separated teacher-target collection from `phi` optimization
  - improved offline latent fit, but produced a null runtime result
  - did not replace the bounded-latent dyn-only baseline
- structured dyn-only offline pipeline:
  - rebuilt Candidate 2 around the structured `Z27` root and offline
    supervised Phase 2
  - produced the first end-to-end working structured C2 pipeline artifact
  - now serves as the winner-for-now for the mission-critical structured path
  - later `v2`, `v3`, bottleneck, and residual follow-ups remained informative
    alternates, but did not replace the canonical `v1` artifact
- terrain-lite:
  - real blind terrain-aware student
  - not yet the winner
  - retained as the terrain-aware refinement base

## What Is Proven

The active `Adapt-V3` line now proves:

- rough-terrain locomotion is still intact without terrain privilege in the
  latent target
- the blind history student can preserve locomotion through the explicit
  latent path
- compact terrain privilege can survive all the way into a blind student
- `Adapt-V3` is no longer only a structural aspiration in this repo

It does **not** yet prove:

- Sim2Sim readiness
- hardware readiness
- that the current terrain-lite summary is already optimal
- that the old terrain-lite root is still the right terrain-aware foundation
  for Candidate 2
- that cleaner `phi` optimization alone is enough to repair Candidate 2

## Active Branch Order

Treat the current forward path as five lanes:

### Lane 1: Milestone archive

These are already frozen and should remain stable:

- `adapt_v3_dyn_only_phase1_stage_a_final.pt`
- `adapt_v3_dyn_only_phase2_stage_a_final.pt`

Their role is:

- preserve a working stationary `Adapt-V3` base
- provide safe warm-start and comparison anchors

### Lane 2: Retired exploration

We explicitly tried a dyn-only Phase 2 continuation with mid-episode hidden
dynamics switches.

Outcome:

- it did not hold as a training recipe
- the policy degraded globally before switched episodes became a dominant part
  of training
- the mixed-switch task is now treated as failed exploration, not active path

Interpretation:

- mid-episode hidden-dynamics changes remain a useful evaluation stress test
- they are no longer the active training contract for final `Adapt-V3`

### Lane 3: Mainline winner

This is the current primary deployment-side branch:

- frozen canonical dyn-only student:
  - `adapt_v3_dyn_only_phase2_stage_a_final.pt`

Status:

- first-place candidate after the current final-candidate comparison
- should move first into Sim2Sim and deployment-path follow-up

### Lane 4: Recovery branch

This is the current adaptation-recovery anchor:

- `adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`

Status:

- first canonical checkpoint from the low-switch recovery line
- selected from `model_1200.pt` rather than the final `model_1999.pt`
- proves that non-collapsed online latent behavior can be recovered under
  active switch pressure
- does not yet replace the stationary Stage A winner as the default deployment
  artifact

Interpretation:

- the old Stage A winner remains the safer deployment-side baseline
- the recovery artifact is the stronger adaptation-side reference
- future work should now ask how to preserve the recovery branch's adaptation
  gains without reintroducing long-horizon posture collapse

### Lane 5: Bounded-latent recovery refinement

This is the current Sim2Sim-oriented recovery refinement base:

- `adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

Status:

- first canonical bounded-latent continuation of the low-switch recovery line
- selected from `model_220.pt` after gait, blind-suite, and `ood_switch_v1`
  checkpoint selection
- first frozen training-side fix that materially improves unclamped MuJoCo
  behavior relative to the earlier recovery artifact
- still not the automatic deployment replacement for the stationary Stage A
  winner

Interpretation:

- the earlier recovery artifact remains the canonical adaptation-truth anchor
- the bounded-latent continuation is the canonical refinement base for reducing
  cross-engine latent brittleness
- future work should now ask how to reduce the remaining latent drift without
  giving back the recovered adaptation gains

### Lane 5d: Retired restart attempts

Several deeper C2 restart attempts were evaluated and retired:

- offline supervised `phi(history)` restart
- offline dynamics-bridge restart

Interpretation:

- they were useful diagnostic experiments
- neither replaced the bounded-latent baseline at runtime
- current active work should treat their details as archive material, not a
  live branch surface

### Lane 5b: Max-abs refinement result

This branch was evaluated and should not replace Lane 5:

- `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg-MaxAbs`

Historical note:
- this task name is archived and is not part of the live task registry

Status:

- won its own Isaac checkpoint sweep with `model_300.pt`
- exported successfully
- failed to beat the earlier bounded-latent `model_220.pt` adaptive challenger
  in MuJoCo
- regressed badly in unclamped MuJoCo due to renewed latent blow-up

Interpretation:

- this was a useful negative result
- stronger coordinate-wise max-abs control by itself is not the right next
  adaptive Sim2Sim repair
- the active adaptive challenger remains:
  `adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

### Lane 5c: Temporal-smoothness refinement result

This branch was evaluated and should not replace Lane 5:

- `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg-Smooth`

Historical note:
- this task name is archived and is not part of the live task registry

Status:

- won its own Isaac checkpoint sweep with `model_100.pt`
- exported successfully
- improved over the max-abs branch when clamped
- failed to beat the earlier bounded-latent `model_220.pt` adaptive challenger
  in MuJoCo
- regressed badly in unclamped MuJoCo due to renewed latent blow-up

Interpretation:

- this was a more promising refinement than max-abs, but still not a new
  adaptive MuJoCo leader
- weak temporal smoothing alone is not enough to close the remaining adaptive
  Sim2Sim gap
- the active adaptive challenger remains:
  `adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

### Lane 6: Terrain-aware refinement

This is now the active research-follow-up branch:

- `terrain-lite`

Meaning:

- terrain-aware blind student is now proven real
- but it loses the current head-to-head to dyn-only
- future work should refine the terrain summary rather than return to dense
  terrain privilege

### Lane 7: Stronger-history-encoder adaptive branch

This branch has now been tried and retired:

- `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg-TCN-Anchored`

Meaning:

- an earlier unanchored TCN attempt has been archived
- the anchored continuation has also now been archived
- it keeps the explicit deployment-time latent contract
- it keeps the TCN `phi(history)` encoder
- it starts from the bounded-latent recovery family rather than the stationary
  Stage A trunk
- it uses a longer teacher-action scaffold so `phi` is less free to optimize
  latent matching while giving up gait quality
- it still failed to replace the bounded-latent baseline because locomotion
  decayed late in training

## Immediate Next Moves

The active order is:

1. keep `adapt_v3_dyn_only_phase2_stage_a_final.pt` as the current lead
   deployment candidate
2. preserve `adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt` as the
   canonical adaptation-recovery checkpoint
3. preserve
   `adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt` as the
   canonical Sim2Sim-oriented recovery refinement base
4. preserve `adapt_v3_terrain_lite_phase2_stage_a_final.pt` as the valid
   terrain-aware challenger and refinement base
5. move dyn-only Stage A into Sim2Sim / deployment-path follow-up first
6. treat the max-abs branch as a documented negative result, not the new
   adaptive leader
7. treat the temporal-smoothness branch as a documented non-winning result, not
   the new adaptive leader
8. treat the first TCN branch as an informative non-winning intermediate result
9. make the anchored TCN continuation the next serious adaptive training branch
10. use `docs/ADAPTIVE_SIM2SIM_REFINEMENT_PLAN.md` as the active guide for the
   TCN branch evaluation gate
11. revisit terrain-aware input refinement only after the current deployment-side
   winner path is clearer

While candidate comparison and deployment-path preparation continue, the active
non-training work is:

- define the final comparison rubric clearly
- audit the deployment/Sim2Sim path so the winning candidate can move forward
  without last-minute interface ambiguity

Do **not** jump straight from every milestone to deployment work.

Deployment-side effort should now begin from the current winner branch, while
keeping the terrain-aware branch as a documented challenger.

See also:

- `docs/ADAPT_V3_FINAL_CANDIDATE_COMPARISON.md`

The real repo question was:

- explicit latent path
- rough-terrain competence
- strong per-episode randomized robustness
- competitive final comparison results against the active alternative candidate

## Deployment Gate

The first policy that should earn serious Sim2Sim/deployment preparation must
satisfy all of these:

- blind at inference time
- strong rough-terrain locomotion
- no teacher imitation dependence at the end of training
- strong results under the final comparison rubric
- debug checks consistent with the claimed latent contract

Current answer:

- dyn-only satisfied that bar first
- terrain-lite became a credible but not winning finalist

## Repo Hygiene Rule

To reduce future doc sprawl:

- use this file for active branch order
- use `docs/ADAPT_V3_EXECUTION_SPEC.md` for architecture and barrier record
- use `docs/archive/adapt_v3/KNOWN_LIMITATIONS_AND_BRANCH_FOLLOWUPS.md` for unresolved issues
- treat older `PLAN`, `START`, and exploratory notes as lineage unless they are
  explicitly linked here
