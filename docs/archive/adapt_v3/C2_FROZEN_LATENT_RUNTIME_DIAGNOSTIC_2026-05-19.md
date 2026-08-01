# C2 Frozen-Latent Runtime Diagnostic 2026-05-19

This note records the runtime diagnostic that compared:

- `latent_mode=normal`
- `latent_mode=zero`
- `latent_mode=frozen`

for the current structured offline C2 candidate:

- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`

Test setting:

- task:
  `RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch`
- terrain:
  `random_rough`
- terrain level:
  `5`
- history length:
  `10`

Stress cases:

- weak motor:
  - `motor_stiffness_scale=0.6`
  - `motor_damping_scale=0.6`
- low friction:
  - `static_friction=0.25`
  - `dynamic_friction=0.25`

## Why this diagnostic mattered

The immediate question was whether the current C2 line was failing mainly
because:

- the deployable history window was too short
- the student needed stronger online latent updating
- or the actor was already doing most of the work

Earlier history sweeps had already shown:

- `history_length=10`
- `history_length=20`
- `history_length=40`

all preserve the same qualitative behavior.

So the next more informative check was:

- hold the latent fixed after the first post-reset estimate
- see whether behavior collapses or remains close to `normal`

## Runtime meaning of `frozen`

For this diagnostic:

- `normal`
  - `phi(history)` updates every step
- `zero`
  - actor receives no latent information
- `frozen`
  - capture `phi(history)` once after reset
  - hold that actor latent fixed until reset

This is not the same as disabling the adaptation encoder.

It specifically tests whether online latent *updating* is load-bearing.

## Main result

Current result:

- weak motor:
  - `normal ~= frozen`
  - both remain reasonably functional
- low friction:
  - `normal ~= frozen >> zero`

## Interpretation

This strongly suggests:

- the latent path is not useless
- but the time-varying latent update is not currently the main source of
  deployment success

The current C2 line appears to use the latent more like:

- a coarse episode-level context code

than like:

- a strongly load-bearing online hidden-dynamics tracker

This fits the other recent findings:

- `phi ~= mu` remains weak
- weak-motor adaptation is only modestly load-bearing
- low-friction adaptation matters more than weak-motor adaptation
- longer history does not materially change the story

## Practical conclusion

The structured offline `v1` issue should not currently be described as:

- "the student only needs more history"

It is better described as:

- actor masking / weak online latent dependence
- target-identifiability limitations
- and/or encoder-design limitations

## Recommended next branch priorities

If C2 work resumes, prioritize:

- target redesign
- encoder redesign
- explicit actor-pressure experiments

Prefer those over:

- additional history-length sweeps
- or more blind trainer churn on the same deploy contract
