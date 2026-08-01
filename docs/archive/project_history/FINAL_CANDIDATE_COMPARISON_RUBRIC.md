# Final Candidate Comparison Rubric

This note defines how the final `Adapt-V3` deployment candidate should be
chosen once the active candidate set is frozen.

Its purpose is to prevent the end-of-project comparison from collapsing into:

- one aggregate score
- one flattering scenario
- one anecdotal video
- or one last-minute subjective judgment

Use this as the final comparison contract for:

- the canonical dyn-only blind student
- the terrain-aware blind student
- any later explicitly versioned replacement candidate

## Active Intended Comparison

The current expected head-to-head is:

1. `adapt_v3_dyn_only_phase2_stage_a_final.pt`
2. `adapt_v3_terrain_lite_phase2_stage_a_final.pt`

The terrain-lite Phase 1 privileged/base artifact is not itself a final
deployable candidate. It is the teacher/base half that produced the
terrain-aware blind student.

## Decision Rule

The final winner should not be selected from one scalar score alone.

Instead:

1. establish hard pass/fail gates
2. compare core evaluation families
3. compare deployment-relevant failure modes
4. compare gait/composure quality
5. only then use aggregate score summaries as supporting evidence

## Hard Gates

A policy cannot be the final deployment winner unless all of these are true:

- blind at inference
- no privileged observations used at runtime
- no teacher imitation dependence at the end of training
- no structural checkpoint inconsistency
- clean policy lineage and freeze note
- same evaluation battery run under the same settings as its comparator

For `Adapt-V3`, deployment-time actor path must be:

- `policy + policy_history`
- `phi(history) -> z_hat`
- `pi(policy, z_hat) -> action`

Not acceptable:

- runtime `terrain_lite_privileged`
- runtime `dynamics_privileged`
- runtime `mu(e_t)`

## Primary Evaluation Families

These are the mandatory comparison families.

### 1. Blind suite

Purpose:

- measure nominal and controlled in-distribution robustness

Use:

- `scripts/eval/run_isolated_suite.py`
- suite:
  - `blind_baseline_v1`

Compare:

- scenario ranking
- mean suite score
- weakest-case scenario
- timeout behavior
- tracking error

### 2. Geometry OOD

Purpose:

- measure whether terrain-aware privilege actually improves geometry handling

Use:

- `scripts/eval_ood/run_ood_suite.py`
- suite:
  - `ood_geometry_v1`

This family should be treated as the most important differentiator between:

- dyn-only
- terrain-aware

### 3. Dynamics OOD

Purpose:

- measure hidden-factor robustness outside the in-distribution training range

Use:

- suite:
  - `ood_dynamics_v1`

### 4. Push OOD

Purpose:

- measure disturbance recovery and post-push stability

Use:

- suite:
  - `ood_push_v1`

### 5. Switch OOD

Purpose:

- treat within-episode hidden-factor changes as a held-out stress test

Use:

- suite:
  - `ood_switch_v1`

Important:

- switched hidden-dynamics performance is an evaluation stress family
- it is not the active training contract

## Metrics That Matter Most

These metrics should be treated as first-class decision signals.

### Survival / failure

- `timeout_fraction_of_terminals`
- `timeout_events_per_env`
- `base_contact_events_per_env`
- `terminal_base_height`
- `terminal_base_orientation`
- `low_progress`

### Tracking

- `vel_err_step_mean`
- `yaw_err_step_mean`

### Terrain competence

- `terrain_level_step_mean`
- geometry-suite scenario outcomes

### Gait / composure

From `scripts/eval/gait.py`:

- `gait_interpretation`
- `diagonal_trot_score`
- `forward_lateral_drift_per_meter_mean`
- `base_z_vel_abs_mean`
- standstill terminations
- forward terminations

## Metrics That Are Supporting, Not Primary

These are useful context but should not decide the winner by themselves:

- aggregate suite score
- reward alone
- one strong OOD scenario
- one clean replay clip

## Comparison Hierarchy

Use this order when two candidates disagree.

### Tier 1: Safety and survival

Prefer the candidate with:

- fewer `base_height` failures
- fewer `base_orientation` failures
- fewer contact collapses
- stronger timeout behavior

### Tier 2: Geometry handling

Prefer the candidate that is stronger on:

- stairs up
- stairs down
- boxes
- high-level rough terrain

This tier is where terrain-aware privilege should earn its keep.

### Tier 3: Tracking and composure

Prefer the candidate with:

- lower linear/yaw tracking error
- lower forward drift
- cleaner gait interpretation
- better body composure

### Tier 4: Aggregate suite score

Use this only after the higher-priority checks.

## Expected Outcomes By Branch

### Dyn-only candidate

Expected strengths:

- strong hidden-dynamics robustness
- cleaner blind-student simplicity
- lower deployment complexity

Expected weakness:

- may leave geometry-specific performance on the table

### Terrain-aware candidate

Expected strengths:

- better geometry OOD handling
- better rough-terrain composure if terrain-lite worked as intended

Expected risks:

- worse history-to-latent recovery
- drift toward more privileged teacher-side dependence
- weaker deployability if Phase 2 latent recovery is not clean

## Final Deliverables

Before final policy selection, produce:

1. one frozen eval bundle per candidate
2. one short result table across all mandatory suites
3. one narrative comparison note explaining:
   - where the winner is better
   - where the loser is still better
   - why the final winner was chosen

## Reporting Rule

The final project conclusion should be defensible from this rubric without
asking the reader to trust an opaque score.

That means the final write-up should clearly answer:

- which candidate survived better?
- which candidate handled geometry better?
- which candidate tracked commands better?
- which candidate looked cleaner and more deployable?
- why that combination justified the final winner
