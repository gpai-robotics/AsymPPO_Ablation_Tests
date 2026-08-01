# C1 C2 Transfer Execution Plan

This note defines the next concrete move after baseline selection:

- stop tuning in-place
- move both Candidate 1 and Candidate 2 across the transfer boundary
- learn where the real deployment gap still is

Use this file when the question is:

- what do we run next
- in what order do we take C1 and C2 into Sim2Sim
- what blocks sim2real for either candidate

## Current Truth

### Candidate 1

C1 is the more mature deployment-side candidate.

It already has:

- exported bundle
- source/export parity validation
- Isaac deploy rehearsal
- MuJoCo nominal runtime
- MuJoCo OOD suites
- a repo-native hardware runner path

### Candidate 2

C2 now has:

- a hardened structured offline adaptive baseline
- exported bundle
- bundle validation
- Isaac-side revalidation
- a clean deployable adaptive contract

What it does not yet have to the same degree as C1:

- completed MuJoCo-side adaptive runtime evidence of the same maturity
- a repo-native hardware runner that accepts `blind_adaptive_student`

That means the immediate order should be:

1. C1 transfer confirmation
2. C2 transfer confirmation
3. extend the hardware runner for C2 only after C2 Sim2Sim looks trustworthy

## Candidate Order

### First: Candidate 1

Why first:

- simpler runtime contract
- strongest current deployment story
- best reference for validating the deployment bridge itself

Primary goals:

- confirm the existing deploy surface still runs cleanly
- re-anchor the transfer baseline before comparing adaptive behavior

### Second: Candidate 2

Why second:

- now hardened and exportable
- still the more structurally complex adaptive bundle
- should be tested after the bridge is trusted on C1

Primary goals:

- determine whether the adaptive runtime survives MuJoCo-side execution
- determine whether `phi(history)` remains behaviorally useful across transfer

## Gate Order

For both candidates, the transfer order is:

1. Isaac deploy rehearsal
2. MuJoCo nominal runtime
3. MuJoCo hidden-env suite
4. MuJoCo moderate disturbance suite
5. MuJoCo rough terrain suite

For C1 only, the next immediate real-hardware gate is:

6. hardware dry-run contract check

For C2, hardware is not the next immediate gate yet because the current
hardware runner only supports:

- `blind_history_policy`

and C2 is:

- `blind_adaptive_student`

So C2 hardware bring-up should begin only after:

- C2 Sim2Sim looks acceptable
- `run_go2_hardware.py` is extended to support `phi(history) -> z_hat`

## Script Generator

Generate the execution script:

```bash
python /home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/prepare_c1_c2_transfer_execution.py
```

That now writes two scripts:

- `artifacts/pipeline_runs/c1_c2_transfer_isaac.sh`
- `artifacts/pipeline_runs/c1_c2_transfer_mujoco.sh`

Run the Isaac-side script from the normal Isaac workflow:

```bash
bash /home/bhuvan/projects/rma/rma_go2_lab/artifacts/pipeline_runs/c1_c2_transfer_isaac.sh
```

Then activate the MuJoCo conda env and run:

```bash
bash /home/bhuvan/projects/rma/rma_go2_lab/artifacts/pipeline_runs/c1_c2_transfer_mujoco.sh
```

## What The Generated Scripts Do

### C1

- `c1_c2_transfer_isaac.sh`
  - Isaac deploy rehearsal
  - hardware dry-run contract check
- `c1_c2_transfer_mujoco.sh`
  - MuJoCo nominal runtime
  - MuJoCo hidden-env suite
  - MuJoCo moderate disturbance suite
  - MuJoCo rough terrain suite

### C2

- `c1_c2_transfer_isaac.sh`
  - Isaac deploy rehearsal
- `c1_c2_transfer_mujoco.sh`
  - MuJoCo nominal runtime
  - MuJoCo hidden-env suite
  - MuJoCo moderate disturbance suite
  - MuJoCo rough terrain suite

It intentionally does not launch C2 hardware yet.

## Decision Rule

### If C1 fails

Interpretation:

- the deployment bridge itself is not yet stable enough
- fix C1 transfer-path issues before drawing strong conclusions from C2

### If C1 passes and C2 fails

Interpretation:

- the adaptive runtime contract is the remaining problem
- do not restart architecture immediately
- first localize whether the issue is:
  - exported adaptive contract parity
  - MuJoCo adaptive runtime behavior
  - or adaptive history/latent usefulness after transfer

### If both pass

Next step:

- start disciplined sim2real bring-up
- C1 first
- C2 second after hardware-runner adaptive support is added

## Related Docs

Read these together with this plan:

1. [C1_STAGEA_MODEL400_DEPLOY_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_STAGEA_MODEL400_DEPLOY_CARD.md)
2. [C2_STRUCTURED_OFFLINE_FINAL_BASELINE_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_STRUCTURED_OFFLINE_FINAL_BASELINE_CARD.md)
3. [SIM2REAL_C1_BRINGUP_PLAN.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/SIM2REAL_C1_BRINGUP_PLAN.md)
4. [DEPLOYMENT_AUDIT_ADAPT_V3.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/DEPLOYMENT_AUDIT_ADAPT_V3.md)
