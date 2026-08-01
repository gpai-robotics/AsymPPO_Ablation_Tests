# Dyn-Only Deployment Bundle

This directory is the first real deployment-side bundle for the current
`Adapt-V3` winner:

- `adapt_v3_dyn_only_phase2_stage_a_final`

Source checkpoint:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`

Task lineage:

- `RMA-Go2-Adapt-V3-Phase2-StageA`

Bundle contents:

- `bundle_manifest.json`
  - deployment contract metadata
- `export_request.json`
  - completed export record for the frozen candidate
- `adapt_v3_dyn_only_phase2_stage_a_final.torchscript.pt`
  - deterministic deployable TorchScript module
- `adapt_v3_dyn_only_phase2_stage_a_final.onnx`
  - deterministic deployable ONNX module
- `adapt_v3_dyn_only_phase2_stage_a_final.export_metadata.json`
  - deployment-side tensor and runtime contract sidecar

Current status:

- bundle manifest created
- real export completed
- structural validation passed
- deployable-I/O rehearsal implemented and smoke-tested
- source-vs-export parity smoke passed
- MuJoCo Sim2Sim preflight gate implemented
- canonical primary model selected:
  - `reference_repos/mujoco_menagerie/unitree_go2/scene.xml`
- first repo-owned MuJoCo runtime bridge implemented
- local execution is still blocked until the `mujoco` Python backend is
  available in the active environment

Deployable runtime contract for this candidate:

- policy kind:
  - `blind_adaptive_student`
- deployable observation groups:
  - `policy`
  - `policy_history`
- latent update:
  - per-step history update via `phi(history) -> z_hat`

Important claim discipline:

- this bundle exposes a latent-conditioned blind student runtime contract
- it does not by itself prove that the frozen Stage A winner shows strong
  online-changing latent behavior under hidden-dynamics switches
- see:
  - `docs/ADAPTATION_PROBE_NOTES.md`
  - `docs/ADAPT_V3_POISONING_AUDIT.md`

Next bundle-side steps:

1. run the runtime bridge once `mujoco` is available in the active environment
2. add longer deploy-side rehearsal runs and archive their JSON outputs
3. broaden parity coverage if later bundles need different runtime wrappers
