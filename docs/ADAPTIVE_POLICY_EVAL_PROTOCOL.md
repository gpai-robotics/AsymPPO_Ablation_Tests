# Adaptive Policy Evaluation Protocol

This note defines the canonical evaluation and testing protocol for adaptive
`Adapt-V3` branches.

Its purpose is to answer a simple but important question:

- how do we rigorously decide whether an adaptive policy is actually moving the
  project forward?

This protocol is intentionally broader than one suite or one simulator.

It is designed to prevent four failure modes:

1. confusing locomotion quality with real online adaptation
2. confusing training-side latent metrics with deployable behavior
3. over-trusting Isaac-only results
4. promoting a more complex adaptive policy that is still worse than the
   stationary deployment winner where it matters

Read this together with:

- `docs/ADAPT_V3_ACTIVE_ROADMAP.md`
- `docs/ADAPTIVE_SIM2SIM_REFINEMENT_PLAN.md`
- `docs/SIM2SIM_STAGEA_VS_ADAPTIVE_COMPARISON.md`
- `docs/FINAL_CANDIDATE_COMPARISON_RUBRIC.md`
- `docs/DEPLOYMENT_AUDIT_ADAPT_V3.md`

## Current Status

The repo has already completed a meaningful fraction of this protocol.

Strongly established already:

- training-side adaptation truth checks
- checkpoint-swept Isaac evaluation
- Isaac vs MuJoCo runtime-trace comparison
- direct A/B against the stationary Stage A winner

Still only partially systematized:

- formal source-vs-export parity checks
- structured post-switch recovery analysis
- long-horizon multi-seed stability battery
- one mandatory pass/fail gate applied to every new adaptive branch

This note closes that organizational gap.

## Core Principle

Do not trust one metric family alone.

For adaptive policies, the minimum rigorous stack is:

1. prove adaptation is real in training
2. prove locomotion remains strong in Isaac
3. prove deploy/export contract is sound
4. prove gait logic survives in MuJoCo
5. prove the adaptive branch earns its complexity by helping on hidden-factor
   stress tests

If any one of those layers fails badly, the branch should not be promoted.

## Canonical Baselines

Every adaptive branch should be interpreted relative to these frozen anchors.

### Deployment-side winner

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`

Meaning:

- strongest current deployment-side baseline
- reference for MuJoCo gait quality and deploy-side composure

### Adaptation-recovery anchor

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`

Meaning:

- reference proving real adaptation was restored

### Bounded-latent adaptive refinement base

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

Meaning:

- current best adaptive Sim2Sim-oriented refinement baseline

## Evaluation Ladder

Each new adaptive branch should be judged through the following layers, in
order.

## Layer 1: Training-Side Adaptation Truth

Purpose:

- prove that the adaptive branch is actually learning online hidden-factor
  inference rather than collapsing into a fixed bias

Track at minimum:

- `latent_cosine`
- `latent_regression`
- `student_latent_batch_std`
- `teacher_latent_batch_std`
- `student_latent_l2`
- `student_latent_max_abs`
- any branch-specific latent stability metric such as:
  - `student_latent_max_abs_excess`
- `adaptation_switch_applied_frac`

Strong signals:

- non-trivial student latent variation
- improving latent cosine
- decreasing latent regression loss
- non-zero switch application fraction

Warning signs:

- student latent variance collapsing toward zero
- latent cosine stagnating at weak values
- adaptation switch fraction never activating
- latent control penalty dominating and crushing useful latent movement

Layer-1 verdict:

- a branch should not move forward if it fails to show real adaptation health
  in training

## Layer 2: Checkpoint-Swept Isaac Evaluation

Purpose:

- verify that locomotion remains strong while adaptation improves

Discipline:

- do not assume the final checkpoint is best
- sweep an early-to-mid candidate window

Minimum recommended sequence:

1. quick gait screen
2. `blind_baseline_v1`
3. `ood_switch_v1`

Optional but useful:

- `ood_dynamics_v1`
- longer scenario-specific probes if a branch targets a known weakness

Current repo tools:

- `scripts/eval/gait.py`
- `scripts/eval/run_isolated_suite.py`
- `scripts/eval_ood/run_ood_suite.py`

Important metrics:

- mean suite score
- scenario ranking
- `vel_err_step_mean`
- `yaw_err_step_mean`
- `timeout_fraction_of_terminals`
- `base_height` and `base_orientation` failures
- gait/composure metrics from the gait screen

Layer-2 verdict:

- a branch should not move forward if adaptation improves but usable Isaac
  locomotion clearly regresses versus the current adaptive baseline

### Interpreting `normal` vs `zero` vs `frozen`

For deployable adaptive students, one of the most useful runtime truth checks
is now a latent-mode ladder:

- `normal`
- `zero`
- `frozen`

Interpretation:

- `normal`
  - standard runtime path
  - `phi(history)` updates every step
- `zero`
  - remove latent information entirely
- `frozen`
  - capture `phi(history)` once after reset
  - hold that latent fixed until the env resets

This split matters because it distinguishes three different stories:

- `normal >> zero` and `normal >> frozen`
  - online latent updating is genuinely load-bearing
- `normal ~= frozen >> zero`
  - the latent matters, but mostly as a coarse episode-level context code
- `normal ~= frozen ~= zero`
  - the actor backbone is carrying nearly everything

Recent structured C2 diagnostics currently support:

- weak motor:
  - `normal ~= frozen`
- low friction:
  - `normal ~= frozen >> zero`

So current C2 behavior should be interpreted as:

- some latent/context usefulness is real
- but strong online latent updating is not yet the main source of deployment
  success

This check should be preferred over additional history-length sweeps once
`10`, `20`, and `40` step windows all tell the same story.

## Layer 3: Deployment-Contract and Parity Checks

Purpose:

- verify that the policy survives packaging/export without hidden contract
  damage

Required steps:

1. package candidate bundle
2. export deploy artifact
3. validate manifest / metadata
4. confirm runtime path uses only deployable groups

Current repo tools:

- `scripts/deploy/package_candidate.py`
- `scripts/deploy/export_policy.py`
- `scripts/deploy/validate_bundle.py`

Required contract:

- `policy_kind = blind_adaptive_student`
- deployable groups:
  - `policy`
  - `policy_history`
- runtime update semantics:
  - `phi(history) -> z_hat`

Current status:

- structural bundle checks are real
- parity checks are still less formal than they should be

Recommended parity comparison:

- same checkpoint
- same fixed command
- same initial condition when possible
- compare:
  - action mean
  - `q_target`
  - latent norm trend
  - base pose trend

Layer-3 verdict:

- no branch should be promoted if exported behavior is structurally unclear or
  materially inconsistent with the intended runtime contract

## Layer 4: Isaac vs MuJoCo Runtime-Trace Comparison

Purpose:

- test whether the policy keeps the same gait logic outside Isaac

Current repo tools:

- Isaac trace:
  - `scripts/eval/trace_isaac_policy.py`
- MuJoCo trace:
  - `scripts/deploy/run_sim2sim.py`
- comparison:
  - `scripts/eval/compare_runtime_traces.py`

Required outputs:

- Isaac full trace JSON
- MuJoCo full trace JSON
- comparison JSON / markdown
- plots when useful

Mandatory metrics:

- `reward_proxy_mean`
- `vel_err_step_mean`
- `yaw_err_step_mean`
- `base_height_mean`
- `base_tilt_projected_gravity_xy_mean`
- `latent_norm_mean`
- `latent_norm_max`
- `action_abs_mean`
- all-4 contact fraction
- diagonal support fractions
- per-foot contact fraction
- per-foot mean foot height

Interpretation rule:

- a branch that looks good in Isaac but degrades into sticky,
  asymmetric, over-supported stepping in MuJoCo has not yet solved the
  deployment-side adaptive problem

Layer-4 verdict:

- adaptive branches must now pass through runtime-trace comparison before they
  can be treated as serious promotion candidates

## Layer 5: A/B Against The Stationary Winner

Purpose:

- verify that the adaptive branch is not merely “interesting,” but actually
  earning its additional complexity

Mandatory comparator:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`

Interpretation:

- if the adaptive branch is worse than Stage A everywhere, its adaptation story
  is not yet paying for itself
- if it loses on MuJoCo gait quality but wins on hidden-dynamics stress tests,
  it can still be a valid refinement base

This distinction matters:

- deployment winner
- adaptive research leader

are not yet automatically the same artifact in this repo

## Layer 6: Adaptation-Specific Stress Tests

Purpose:

- isolate whether online adaptation is actually useful under hidden-factor
  changes

Minimum required family:

- `ood_switch_v1`

Preferred additional analysis:

- compare adaptive branch against Stage A on the same hidden-dynamics switches
- inspect post-switch latent movement
- inspect post-switch tracking recovery
- inspect whether recovery is faster or more stable than the stationary winner

Current repo status:

- switch OOD and low-switch recovery branches are real
- structured post-switch recovery analysis is still not as formal as it should
  be

Recommended future extension:

- add a focused post-switch recovery report that measures:
  - pre-switch vs post-switch tracking error
  - time-to-recover after switch
  - latent shift magnitude after switch

Layer-6 verdict:

- a branch should not be called an adaptive improvement unless it helps on
  hidden-factor stress tests in a way that a stationary baseline does not

## Layer 7: Long-Horizon Stability

Purpose:

- catch policies that pass short demos but degrade over longer horizons

Recommended checks:

- long fixed-command rollouts
- long mixed-command rollouts
- repeated reset/restart tests
- multi-seed repeat runs for important candidates

Current repo status:

- long-horizon intuition exists
- but this layer is not yet fully formalized

Practical current minimum:

- 1000-step Isaac trace
- 1000-step MuJoCo trace
- multi-checkpoint selection instead of trusting final training state

Layer-7 verdict:

- no final adaptive promotion should rely only on short-window behavior

## Mandatory Gate For New Adaptive Branches

The minimum branch-promotion gate is:

1. Layer 1 passes
2. Layer 2 passes
3. Layer 3 is structurally clean
4. Layer 4 comparison exists
5. Layer 5 A/B against Stage A exists
6. Layer 6 switch-family evidence is positive

If any one of those is missing, the branch is still exploratory.

## Current Quantitative Baselines

Use these current anchors when interpreting adaptive MuJoCo progress.

### Stage A MuJoCo anchor

From:

- `artifacts/debug/adapt_v3_dyn_only_phase2_stage_a_final_runtime_compare.json`

Key values:

- `reward_proxy_mean = 0.411`
- `vel_err_step_mean = 0.155`
- `yaw_err_step_mean = 0.082`
- `base_height_mean = 0.332`
- `base_tilt_projected_gravity_xy_mean = 0.065`
- diagonal support:
  - `FL+RR = 0.267`
  - `FR+RL = 0.262`
- all-4 contact:
  - `0.079`

Meaning:

- current deployment-side stretch target

### Current bounded-latent adaptive MuJoCo baseline

From:

- `artifacts/debug/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_model220_runtime_compare.json`

Key values:

- `reward_proxy_mean = 0.302`
- `vel_err_step_mean = 0.267`
- `yaw_err_step_mean = 0.163`
- `base_height_mean = 0.267`
- `base_tilt_projected_gravity_xy_mean = 0.103`
- diagonal support:
  - `FL+RR = 0.116`
  - `FR+RL = 0.049`
- all-4 contact:
  - `0.558`
- `latent_norm_mean = 25.767`
- `latent_norm_max = 121.183`

Meaning:

- current adaptive Sim2Sim-oriented baseline to beat

## Promotion Criteria

### Exploratory success

A branch counts as exploratory success if:

- adaptation truth is healthy
- Isaac locomotion remains usable
- MuJoCo behavior is at least as good as the current bounded-latent adaptive
  baseline

### Real adaptive improvement

A branch counts as a real adaptive improvement if it beats the current
bounded-latent baseline on most of:

- `reward_proxy_mean`
- `vel_err_step_mean`
- `yaw_err_step_mean`
- `base_height_mean`
- `base_tilt_projected_gravity_xy_mean`
- all-4 contact fraction
- combined diagonal support fraction
- `latent_norm_mean`
- `latent_norm_max`

### Serious promotion candidate

A branch counts as a serious promotion candidate if:

- it materially closes the MuJoCo gap to Stage A
- while preserving real adaptive-branch identity on switch-family tests

This does not require beating Stage A everywhere.
It does require a believable tradeoff in favor of keeping the adaptive branch
alive.

## Recommended Reporting Bundle

For every serious adaptive branch, archive:

1. training smoke snapshot
2. checkpoint shortlist note
3. gait screen outputs
4. blind-suite outputs
5. switch OOD outputs
6. exported bundle path
7. Isaac trace JSON
8. MuJoCo trace JSON
9. runtime comparison JSON / markdown
10. short narrative note:
    - what improved
    - what stayed weak
    - whether the branch should be frozen, refined, or retired

## Operational Checklist

Use this section as the practical runbook for a new adaptive branch.

Before starting, define these shell variables:

```bash
TASK=<adaptive-task-id>
RUN_DIR=<training-run-dir>
OUT_DIR=<eval-output-dir>
OOD_OUT_DIR=<ood-eval-output-dir>
POLICY_NAME=<short-policy-name>
CKPT=<chosen-checkpoint>
BUNDLE_DIR=/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/exported/$POLICY_NAME
```

Recommended examples:

- `TASK=RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg`
- `RUN_DIR=/home/bhuvan/tools/IsaacLab/logs/rsl_rl/<run-name>/<timestamp>`
- `OUT_DIR=/home/bhuvan/projects/rma/rma_go2_lab/artifacts/evaluations/<branch-name>`
- `OOD_OUT_DIR=/home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/<branch-name>`

Historical note:
- `...LatentReg-MaxAbs` was a useful refinement branch, but it is archived and
  should not be used as the default example for current evaluation work

### Checklist A: Training Smoke

Purpose:

- confirm the branch starts cleanly before a longer run

Command:

```bash
CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/tools/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" \
  --max_iterations 1
```

Record:

- `latent_cosine`
- `latent_regression`
- `student_latent_batch_std`
- `student_latent_l2`
- `student_latent_max_abs`
- branch-specific latent stability metrics
- reward
- episode length

### Checklist B: Short Continuation

Purpose:

- make sure the branch remains healthy beyond the first iteration

Command:

```bash
CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/tools/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" \
  --max_iterations 20
```

Decision:

- continue only if latent health and locomotion both remain sane

### Checklist C: Checkpoint Sweep

Purpose:

- avoid trusting the final checkpoint by default

Recommended adaptive shortlist:

- early checkpoint
- early-mid checkpoint
- mid checkpoint
- first degradation-zone checkpoint

Example shell setup:

```bash
mkdir -p "$OUT_DIR" "$OOD_OUT_DIR"

CKPTS=(
  "$RUN_DIR/model_100.pt"
  "$RUN_DIR/model_160.pt"
  "$RUN_DIR/model_220.pt"
  "$RUN_DIR/model_300.pt"
)
```

### Checklist D: Gait Screen

Purpose:

- quickly eliminate obviously weak checkpoints

Command template:

```bash
for CKPT in "${CKPTS[@]}"; do
  NAME=$(basename "$CKPT" .pt)

  env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
    /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/gait.py \
    --task "$TASK" \
    --checkpoint "$CKPT" \
    --num_envs 16 \
    --steps 200 \
    --command-profile standstill \
    --json-out "$OUT_DIR/${NAME}_gait_standstill.json" \
    --headless

  env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
    /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/gait.py \
    --task "$TASK" \
    --checkpoint "$CKPT" \
    --num_envs 16 \
    --steps 200 \
    --command-profile forward \
    --json-out "$OUT_DIR/${NAME}_gait_forward.json" \
    --headless
done
```

Keep:

- the best two or three checkpoints for deeper evaluation

### Checklist E: Blind Suite

Purpose:

- measure broad in-distribution blind locomotion quality

Command template:

```bash
BEST_CKPTS=(
  "$RUN_DIR/model_160.pt"
  "$RUN_DIR/model_220.pt"
)

for CKPT in "${BEST_CKPTS[@]}"; do
  NAME=$(basename "$CKPT" .pt)

  env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
    /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/run_isolated_suite.py \
    --task "$TASK" \
    --checkpoint "$CKPT" \
    --suite blind_baseline_v1 \
    --output-dir "$OUT_DIR/$NAME"
done
```

Archive:

- suite JSONs
- suite CSVs
- shortlist rationale

### Checklist F: Switch OOD

Purpose:

- test hidden-factor stress handling directly

Command template:

```bash
for CKPT in "${BEST_CKPTS[@]}"; do
  NAME=$(basename "$CKPT" .pt)

  env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
    /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py \
    --task "$TASK" \
    --checkpoint "$CKPT" \
    --suite ood_switch_v1 \
    --output-dir "$OOD_OUT_DIR/$NAME"
done
```

Optional:

- `ood_dynamics_v1` if the branch specifically targets hidden-dynamics
  robustness

### Checklist G: Bundle and Export

Purpose:

- validate the deployment surface before simulator comparisons

Command template:

```bash
env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/package_candidate.py \
  --policy-name "$POLICY_NAME" \
  --source-checkpoint "$CKPT" \
  --task "$TASK" \
  --phase <phase-name> \
  --policy-kind blind_adaptive_student \
  --observation-groups policy,policy_history \
  --control-rate-hz 50 \
  --bundle-dir "$BUNDLE_DIR" \
  --freeze-note <freeze-note-path> \
  --latent-update "per-step history update via phi(history) -> z_hat"
```

```bash
env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/export_policy.py \
  --policy-name "$POLICY_NAME" \
  --checkpoint "$CKPT" \
  --task "$TASK" \
  --phase <phase-name> \
  --bundle-dir "$BUNDLE_DIR" \
  --format torchscript
```

### Checklist H: Isaac Full Trace

Purpose:

- capture the source-stack runtime reference

Command template:

```bash
env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/trace_isaac_policy.py \
  --task "$TASK" \
  --checkpoint "$CKPT" \
  --num-envs 1 \
  --max-steps 1000 \
  --trace-steps -1 \
  --command-x 0.5 \
  --command-y 0.0 \
  --command-yaw 0.0 \
  --json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/debug/${POLICY_NAME}_isaac_trace.json
```

### Checklist I: MuJoCo Full Trace

Purpose:

- capture the deploy-side runtime behavior

Run from the `rma-mujoco` environment:

```bash
python /home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/run_sim2sim.py \
  --bundle-dir "$BUNDLE_DIR" \
  --control-rate-hz 50 \
  --history-mode runtime \
  --max-steps 1000 \
  --trace-steps -1 \
  --command-x 0.5 \
  --command-y 0.0 \
  --command-yaw 0.0 \
  --execute-runtime \
  --json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/debug/${POLICY_NAME}_mujoco_trace.json
```

Optional diagnostic:

- rerun with `--latent-clamp-max-abs 5` only if needed to isolate residual
  latent-driven instability

### Checklist J: Runtime Comparison

Purpose:

- compare Isaac vs MuJoCo behavior in one artifact set

Command template:

```bash
python /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/compare_runtime_traces.py \
  --isaac-json /home/bhuvan/projects/rma/rma_go2_lab/artifacts/debug/${POLICY_NAME}_isaac_trace.json \
  --mujoco-json /home/bhuvan/projects/rma/rma_go2_lab/artifacts/debug/${POLICY_NAME}_mujoco_trace.json \
  --json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/debug/${POLICY_NAME}_runtime_compare.json \
  --md-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/debug/${POLICY_NAME}_runtime_compare.md \
  --plot-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/debug/${POLICY_NAME}_runtime_compare_plots
```

### Checklist K: Stage A A/B

Purpose:

- force every adaptive branch to justify itself against the stationary winner

Required comparison target:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`

Minimum requirement:

- compare the new adaptive branch’s runtime comparison outputs against the
  existing Stage A runtime comparison artifact:
  - `artifacts/debug/adapt_v3_dyn_only_phase2_stage_a_final_runtime_compare.json`

### Checklist L: Branch Verdict

Purpose:

- close the loop with a clear repo decision

Every serious branch should end with one short note answering:

1. did adaptation remain real?
2. did Isaac locomotion remain healthy?
3. did MuJoCo improve over the current adaptive baseline?
4. how far is it still from Stage A?
5. should the branch be:
   - frozen
   - refined again
   - or retired

## Current Best Interpretation

The repo is no longer in a casual-eval state.

We already have enough infrastructure to evaluate adaptive branches
substantively.

What this protocol adds is:

- one canonical place to say what “rigorous enough” means
- one repeatable ladder for future branches
- one discipline for separating:
  - adaptive truth
  - Isaac performance
  - export/deploy contract
  - MuJoCo behavior
  - branch-promotion decisions
