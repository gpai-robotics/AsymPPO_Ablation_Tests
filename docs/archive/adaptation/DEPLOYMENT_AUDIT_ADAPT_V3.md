# Adapt-V3 Deployment Audit

This note audits the current deployment path for the active `Adapt-V3`
candidates.

Its job is to answer:

- what the repo can already do for deployment packaging
- what is still scaffold-only
- what the eventual winning candidate must satisfy before Sim2Sim and hardware

## Audit Scope

This audit covers:

- `scripts/deploy/package_candidate.py`
- `scripts/deploy/export_policy.py`
- `scripts/deploy/validate_bundle.py`
- `scripts/deploy/play_deploy_policy.py`
- `scripts/deploy/run_sim2sim.py`
- deployment planning docs

It does not yet certify:

- real export correctness
- MuJoCo runtime integration
- hardware runtime integration

## Current Deployment Surface

The repo already has a clean deployment-facing scaffold:

- [package_candidate.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/package_candidate.py)
- [export_policy.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/export_policy.py)
- [validate_bundle.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/validate_bundle.py)
- [play_deploy_policy.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/play_deploy_policy.py)
- [run_sim2sim.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/run_sim2sim.py)

That is good news: the repo surface is now stable enough that we can package a
winner without inventing deployment structure at the last minute.

## What Already Works

### Bundle manifest creation

[package_candidate.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/package_candidate.py)
already records:

- policy name
- source checkpoint
- task
- phase
- policy kind
- deployable observation groups
- control rate
- latent update semantics
- freeze note

This is enough to define a candidate bundle contract.

### Bundle validation

[validate_bundle.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/validate_bundle.py)
already checks:

- required manifest fields
- source checkpoint existence
- exported artifact path existence

This is structurally useful, even though it does not yet validate runtime
semantics.

### Deployment rehearsal entrypoints

The repo already has stable entrypoint filenames for:

- simulation-side deployable-I/O rehearsal
- MuJoCo Sim2Sim rehearsal

That means later implementation can grow in place without changing the repo
surface.

## What Is Still Scaffold-Only

### Export is only partially implemented

[export_policy.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/export_policy.py)
now supports the active dyn-only `Adapt-V3` deployment contract and writes:

- TorchScript export
- ONNX export
- sidecar tensor/runtime metadata
- updated bundle manifest entries

What it does not yet do:

- cover arbitrary earlier policy families
- verify parity between source-stack inference and exported artifacts
- certify runtime correctness beyond structural export success

### Sim-side deployable rehearsal is partially implemented

[play_deploy_policy.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/play_deploy_policy.py)
now supports the active dyn-only `Adapt-V3` bundle and can:

- load the exported TorchScript artifact
- drive IsaacLab through only `policy` + `policy_history`
- record deploy-side tracking, posture, and termination summaries

What it does not yet do:

- produce longer canonical archived rehearsal bundles automatically
- support multiple policy families with different runtime wrappers

### MuJoCo Sim2Sim is partially implemented

[run_sim2sim.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/run_sim2sim.py)
now provides:

- a real preflight gate
- canonical Go2 model selection
- an optional repo-owned MuJoCo runtime bridge path when the backend is
  available

It does not yet certify:

- that the local environment has a working MuJoCo Python runtime
- that the runtime bridge numerically matches IsaacLab behavior
- that the runtime bridge is hardware-ready

Active debugging record:

- [SIM2SIM_DEBUGGING_NOTES.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/SIM2SIM_DEBUGGING_NOTES.md)
  captures the non-obvious bridge bugs already found and fixed, including:
  - body-frame velocity semantics
  - root reset pose alignment
  - training-vs-deploy joint tensor ordering
- [SIM2SIM_STAGEA_VS_ADAPTIVE_COMPARISON.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/SIM2SIM_STAGEA_VS_ADAPTIVE_COMPARISON.md)
  captures the later side-by-side runtime-trace comparison showing that:
  - the Stage A winner walks well in MuJoCo
  - the adaptive bounded-latent branch is the policy that still carries the
    remaining Sim2Sim fragility
- [ADAPTATION_PROBE_NOTES.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/ADAPTATION_PROBE_NOTES.md)
  captures the current deployment-side adaptation probe result:
  - hidden-dynamics switches are real
  - policy history changes strongly
  - the inferred dyn-only latent does not move under the tested probe
  - the actor uses latent, but `phi(history)` currently behaves like a fixed
    code rather than an online-changing estimate

## Candidate Requirements

The eventual winning `Adapt-V3` candidate must package as:

- `policy_kind = blind_adaptive_student`
- deployable observation groups:
  - `policy`
  - `policy_history`
- latent update semantics:
  - per-step history update for `phi(history) -> z_hat`

Not acceptable for deployment:

- privileged terrain group at runtime
- privileged dynamics group at runtime
- direct `mu(e_t)` inference path at runtime

## Active Candidate Audit

### Dyn-only student

Current frozen candidate:

- [adapt_v3_dyn_only_phase2_stage_a_final.pt](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt)
- deployment bundle:
  - [bundle_manifest.json](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/exported/adapt_v3_dyn_only_phase2_stage_a_final/bundle_manifest.json)
  - [export_request.json](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/exported/adapt_v3_dyn_only_phase2_stage_a_final/export_request.json)

Status:

- valid deployment-path candidate
- blind at inference
- first packaged deployment-side baseline now exists
- suitable as the first Sim2Sim/export implementation target
- important caveat:
  current deployment-side probes do not show strong online latent adaptation;
  see [ADAPTATION_PROBE_NOTES.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/ADAPTATION_PROBE_NOTES.md)

### Dyn-only recovery student

Current frozen recovery artifact:

- [adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt)

Status:

- canonical checkpoint from the low-switch recovery branch
- selected from `model_1200.pt` after an intra-run checkpoint sweep
- important meaning:
  this is the first frozen `Adapt-V3` branch artifact in the repo that
  combines real active switch exposure during training with non-collapsed
  student latent behavior
- deployment interpretation:
  do not silently treat this as the default deployment replacement for the
  stationary Stage A winner yet
- current role:
  adaptation-recovery anchor and future deployment-side challenger

### Dyn-only bounded-latent recovery student

Current frozen bounded-latent artifact:

- [adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt)

Status:

- canonical checkpoint from the bounded-latent continuation of the low-switch
  recovery line
- selected from `model_220.pt` after gait screen, blind-suite ranking, and
  `ood_switch_v1` tie-break
- important meaning:
  this is the first frozen training-side fix that materially improved unclamped
  MuJoCo behavior for the recovered adaptive branch
- deployment interpretation:
  do not silently treat this as the default deployment replacement for the
  stationary Stage A winner yet
- current role:
  Sim2Sim-oriented recovery refinement base and deployment-side adaptive
  challenger

### Terrain-lite branch

Current frozen artifact:

- [adapt_v3_terrain_lite_phase1_stage_a_final.pt](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.pt)
- [adapt_v3_terrain_lite_phase2_stage_a_final.pt](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_terrain_lite_phase2_stage_a_final.pt)

Status:

- now a valid terrain-aware blind-student comparison candidate
- still not the automatic deployment winner
- should go through the same final eval battery and deployment packaging checks
  as the dyn-only candidate

## Immediate Gaps Before Sim2Sim

These are the concrete missing pieces.

### Gap 1: export implementation

Need:

- keep the new dyn-only exporter as the reference path
- add parity validation for exported artifacts
- broaden support if later candidates require a different runtime wrapper

### Gap 2: bundle population

Need:

- exported artifacts added into `exported_artifacts`
- bundle manifest updated after export

### Gap 3: deployable-I/O rehearsal

Need:

- longer archived rehearsal runs for the winner bundle
- explicit drift/history-contract checks beyond the current summary metrics
- generalized parity coverage if later candidates need a different export
  wrapper

### Gap 4: MuJoCo bridge

Need:

- backend availability in the active Python environment
- observation mapping
- action scaling mapping
- control-rate timing contract
- longer rehearsal runs and parity-style validation against the source IsaacLab
  behavior

Current model/runtime decision:

- primary Go2 MuJoCo model:
  - `reference_repos/mujoco_menagerie/unitree_go2/scene.xml`
- runtime/integration reference:
  - `reference_repos/unitree_mujoco/`

This is intentional.

The menagerie model gives us a cleaner canonical robot/scene baseline, while
`unitree_mujoco` gives us better guidance for how a Go2-oriented MuJoCo loop is
actually wired.

## Recommended Near-Term Order

Once the final candidate winner is chosen:

1. package the winning checkpoint with `package_candidate.py`
2. implement export in `export_policy.py`
3. validate bundle structure with `validate_bundle.py`
4. implement `play_deploy_policy.py` for deployable-I/O rehearsal
5. implement `run_sim2sim.py` for MuJoCo rehearsal
6. only then move into hardware-specific prep

Current operational interpretation:

- stationary Stage A dyn-only:
  first deployment-side winner and first export/Sim2Sim target
- recovery low-switch dyn-only:
  first strong adaptation-recovery artifact and the next deployment-side
  challenger if later bundle/probe work supports it
- bounded-latent recovery low-switch dyn-only:
  first training-side Sim2Sim robustness repair for the adaptive branch and the
  current best adaptive MuJoCo challenger
- max-abs bounded-latent low-switch dyn-only:
  completed refinement attempt, but not a new adaptive MuJoCo leader
- smooth bounded-latent low-switch dyn-only:
  completed refinement attempt, better than max-abs under clamp, but still not
  a new adaptive MuJoCo leader

Recent branch outcome to keep explicit:

- the exported max-abs `model_300` candidate used the correct dedicated bundle
  and was re-run in MuJoCo
- the result was worse than the earlier bounded-latent `model_220` challenger
  when unclamped
- clamp-5 still improved it, but not enough to justify replacing the earlier
  bounded-latent challenger
- the exported smooth `model_100` candidate also used the correct dedicated
  bundle and was re-run in MuJoCo
- unclamped, it regressed substantially versus the earlier bounded-latent
  `model_220` challenger
- clamp-5 improved it to a usable policy, but still not enough to justify
  replacing the earlier bounded-latent challenger

So the active deployment-side adaptive ordering remains:

1. stationary Stage A winner for the main deployment path
2. bounded-latent recovery challenger as the current best adaptive MuJoCo
   branch artifact
3. max-abs branch as a negative refinement record rather than a promoted
   candidate
4. smoothness branch as a non-winning refinement record rather than a promoted
   candidate

## Current Conclusion

The deployment path is in a good architectural state but not yet a completed
runtime pipeline.

In plain terms:

- the repo is ready to organize and freeze a winner cleanly
- the repo is not yet ready to honestly claim Sim2Sim or hardware readiness
- the current dyn-only deployment artifact is strong, but its adaptation claim
  should be phrased carefully:
  the actor uses latent, while the current `phi(history)` probe result looks
  collapsed rather than visibly online-adaptive

That is fine for the current project stage.

The important thing is that once the winning candidate is selected, the next
deployment work is now well-defined instead of ambiguous.
