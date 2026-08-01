# C2 Adapt-V3 Baseline Candidate Card

This note is the canonical candidate card for the current Candidate 2 line.

## Identity

- candidate line:
  `Candidate 2`
- current task:
  `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg`
- current baseline artifact:
  `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`
- current repo role:
  active adaptive baseline and current C2 reference
- policy kind:
  `blind_adaptive_student`

## What Candidate 2 Is

Candidate 2 is the repo's explicit RMA-style adaptive line.

It uses an explicit:

- `mu`
- `phi`
- `pi`

contract:

```text
privileged dynamics e_t
  -> mu
  -> z_t

policy_history
  -> phi
  -> z_hat_t

policy + z_hat_t
  -> pi
  -> action
```

At deployment:

- no terrain privilege
- no dynamics privilege
- only `policy` and `policy_history`

## Current Repo Truth

The repo now has a canonical working structured C2 pipeline artifact:

- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`

This does not mean all structured C2 optimization is finished.

It does mean the mission-critical pipeline now works end-to-end:

- frozen structured Phase 1 root
- on-policy history collection
- offline supervised `phi` training
- runtime evaluation without the old online PPO collapse

So Candidate 2 should currently be understood as:

- an active adaptive research line
- with a working structured offline baseline now in hand
- plus an older bounded-latent adaptive baseline that still matters as a comparison anchor

## Structured Rebuild

The structured dyn-only root rebuild is now the completed modern C2 restart:

- [C2_STRUCTURED_Z27_REBUILD_PLAN.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_STRUCTURED_Z27_REBUILD_PLAN.md)

It changed the root contract itself instead of trying to rescue the older
free-form latent with more local training tricks.

The Phase 1 root for that rebuild is now frozen as:

- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.pt`

The first viable structured Phase 2 offline candidate is now frozen as:

- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`

Operational runbook:

- [C2_STRUCTURED_OFFLINE_PIPELINE_RUNBOOK.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_STRUCTURED_OFFLINE_PIPELINE_RUNBOOK.md)

Operational hardening card:

- [C2_STRUCTURED_OFFLINE_FINAL_BASELINE_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_STRUCTURED_OFFLINE_FINAL_BASELINE_CARD.md)

Fresh-start blueprint for a future cleaner branch:

- [C2_RMA_RESTART_BLUEPRINT.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_RMA_RESTART_BLUEPRINT.md)

Current plain-language C1 vs C2 standing:

- [C1_C2_STATUS_COMPARISON.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_C2_STATUS_COMPARISON.md)

## Why This Baseline Matters

The bounded-latent recovery artifact still matters because it:

- restored real hidden-switch adaptation pressure
- materially improved unclamped MuJoCo behavior relative to the earlier
  recovery artifact
- remains the pre-structured adaptive baseline that the new offline path was
  trying to replace

The new structured offline `v1` artifact matters because it:

- gives us the first working end-to-end structured C2 pipeline
- preserves healthy locomotion through Phase 2
- avoids the online PPO instability that kept biting the older recovery lines

## Recent Diagnostic Conclusion

Recent runtime diagnostics on the structured offline `v1` candidate support the
following current interpretation:

- `history_length` is not the main bottleneck
- `10`, `20`, and `40` step histories all preserve the same qualitative story
- weak-motor adaptation remains only modestly load-bearing
- low-friction adaptation still matters more than weak-motor adaptation
- `phi ~= mu` remains weak even when behavior is acceptable

The most important runtime result is the new latent-mode ablation:

- `normal`
- `zero`
- `frozen`

where `frozen` means:

- capture `phi(history)` once after reset
- hold that latent fixed until the env resets

Current read:

- weak motor:
  - `normal ~= frozen`
  - both remain reasonably functional
- low friction:
  - `normal ~= frozen >> zero`

Interpretation:

- the latent path is not useless
- but the policy currently uses it more like a coarse episode-level context
  code than a strongly load-bearing online updater
- the proprioceptive actor backbone appears to carry most within-episode
  correction

So the current structured C2 issue should not be described as:

- "the student only needs a longer history"

It should be described more accurately as:

- likely actor masking / weak online latent dependence
- likely target-identifiability and/or encoder-design limitations

## Current Architecture

Implementation:

- [rma_v3_actor_critic.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/adaptation/rma_v3_actor_critic.py)

Key modules:

- `mu`
  - privileged hidden-dynamics encoder
- `phi`
  - deployable history-to-latent path
- `pi`
  - action policy consuming current policy obs plus latent

Current runtime contract:

- `policy`
- `policy_history`
- `phi(history) -> z_hat`
- `pi(policy, z_hat) -> action`

## Training Contract

Task:

- `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg`

Environment chain:

- [rough_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/adaptation/rough_cfg.py)
- [rough_history_stage_a_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/adaptation/rough_history_stage_a_cfg.py)
- [rough_history_switch_recovery_dyn_only_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/adaptation/rough_history_switch_recovery_dyn_only_cfg.py)

Recovery setting:

- switch-enabled episode probability:
  `0.05`
- switch step:
  `500`

Possible within-episode hidden switches:

- `ultra_low_friction`
- `very_heavy`
- `very_weak_motor`

## Active Comparison Anchors

Deployment-side stationary winner:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`

Earlier recovery anchor:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`

Candidate 1 reference:

- [C1_STAGEA_MODEL400_DEPLOY_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_STAGEA_MODEL400_DEPLOY_CARD.md)

Teacher reference:

- [TEACHER_V4_MODEL300_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/TEACHER_V4_MODEL300_CARD.md)

## Recent Failed Follow-Up

The TCN anchored continuation was tried and has now been retired.

Archived note:

- [C2_TCN_ANCHORED_FAILED_BRANCH_2026-05-15.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/archive/adapt_v3/C2_TCN_ANCHORED_FAILED_BRANCH_2026-05-15.md)

Failure pattern:

- latent alignment kept improving
- locomotion degraded later in training
- episodes became too short to sustain meaningful switch exposure
- the branch did not replace the bounded-latent baseline

## Archived Failed Follow-Ups

Several C2 recovery attempts were evaluated and retired without replacing the
bounded-latent dyn-only baseline.

Archived notes:

- [C2_DYNAMICS_BRIDGE_NULL_RUNTIME_RESULT_2026-05-15.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/archive/adapt_v3/C2_DYNAMICS_BRIDGE_NULL_RUNTIME_RESULT_2026-05-15.md)
- [C2_PHI_SUPERVISED_NULL_RUNTIME_RESULT_2026-05-15.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/archive/adapt_v3/C2_PHI_SUPERVISED_NULL_RUNTIME_RESULT_2026-05-15.md)
- [C2_TERRAINLITE_DAGGER_FAILED_BRANCH_2026-05-15.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/archive/adapt_v3/C2_TERRAINLITE_DAGGER_FAILED_BRANCH_2026-05-15.md)
- [C2_TCN_ANCHORED_FAILED_BRANCH_2026-05-15.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/archive/adapt_v3/C2_TCN_ANCHORED_FAILED_BRANCH_2026-05-15.md)

## Current Next Step

The next serious C2 move should be:

- treat the structured offline `v1` artifact as the working pipeline winner-for-now
- archive the `v2`, `v3`, bottleneck, and residual follow-ups as informative
  alternates rather than continuing blind trainer churn
- if C2 work resumes, prefer:
  - target redesign
  - encoder redesign
  - or explicit actor-pressure experiments
  over more history-length sweeps

## What Candidate 2 Still Needs To Prove

Before Candidate 2 can be frozen as a final winner, a future branch still must
show:

1. healthy locomotion over long horizons
2. real hidden-switch adaptation
3. clean export/runtime contract
4. stronger deployment-side behavior than the current bounded-latent baseline
5. reduced low-friction weakness relative to current structured and dyn-only candidates

## Structured Offline Follow-Ups

Archived alternate refresh results:

- [C2_STRUCTURED_OFFLINE_V2_ALT_2026-05-18.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/archive/adapt_v3/C2_STRUCTURED_OFFLINE_V2_ALT_2026-05-18.md)
- [C2_STRUCTURED_OFFLINE_V3_LOWFRIC_ALT_2026-05-18.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/archive/adapt_v3/C2_STRUCTURED_OFFLINE_V3_LOWFRIC_ALT_2026-05-18.md)
- [C2_STRUCTURED_OFFLINE_RESB12_ALT_2026-05-18.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/archive/adapt_v3/C2_STRUCTURED_OFFLINE_RESB12_ALT_2026-05-18.md)
- [C2_STRUCTURED_OFFLINE_RESB12_WEAKMOTOR_SWITCH_ALT_2026-05-18.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/archive/adapt_v3/C2_STRUCTURED_OFFLINE_RESB12_WEAKMOTOR_SWITCH_ALT_2026-05-18.md)
- [C2_FROZEN_LATENT_RUNTIME_DIAGNOSTIC_2026-05-19.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/archive/adapt_v3/C2_FROZEN_LATENT_RUNTIME_DIAGNOSTIC_2026-05-19.md)

## Related Docs

- [ADAPT_V3_ACTIVE_ROADMAP.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/ADAPT_V3_ACTIVE_ROADMAP.md)
- [ADAPT_V3_EXECUTION_SPEC.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/ADAPT_V3_EXECUTION_SPEC.md)
- [ADAPTIVE_POLICY_EVAL_PROTOCOL.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/ADAPTIVE_POLICY_EVAL_PROTOCOL.md)

## Canonical Reading Rule

For Candidate 2 itself, use this file first.

Use the linked docs only for:

- branch history
- architecture detail
- evaluation procedure
- deployment-specific audits
