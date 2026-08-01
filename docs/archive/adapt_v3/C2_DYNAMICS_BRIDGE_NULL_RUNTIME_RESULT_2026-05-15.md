# C2 Dynamics-Bridge Null Runtime Result

Date:

- `2026-05-15`

Branch identity:

- offline dyn-only dynamics-bridge restart
- deployable path:
  - `history -> predicted dynamics -> learned bridge -> latent -> actor`

Purpose:

- test the sharper post-audit hypothesis that raw teacher latent was the wrong
  supervision geometry for C2
- replace `phi(history) -> teacher_latent` with:
  - direct `history -> dynamics_privileged` prediction
  - a learned bridge into the frozen controller latent space

Why this branch existed:

- history observability probes showed `policy_history` contained real hidden
  dynamics information
- latent-sensitivity audit showed the controller used latent, but offline
  `phi -> teacher_latent` improvement mostly changed low-control-value latent
  structure
- so the next question became:
  - can direct dynamics supervision plus a learned bridge improve runtime
    behavior better than direct teacher-latent imitation?

Implemented live branch at the time:

- task:
  - `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-DynamicsBridge`
- offline trainer:
  - `scripts/adaptation/train_adapt_v3_dynamics_bridge_supervised.py`

Observed offline result:

- offline optimization behaved sensibly
- validation dynamics loss decreased
- validation action loss decreased strongly
- latent loss did not improve, which was acceptable for this hypothesis

Runtime outcome:

- did **not** beat the canonical dyn-only bounded-latent C2 baseline

Blind suite:

- dynamics-bridge:
  - average score `11.77`
- current baseline:
  - average score `12.36`

OOD dynamics:

- dynamics-bridge:
  - average score `8.92`
- current baseline:
  - average score `9.44`

OOD switch:

- dynamics-bridge:
  - average score `8.12`
- current baseline:
  - average score `8.50`

Gait read:

- remained in the same broad family:
  - `high_duty_diagonal_gait_staggered_touchdown`
- did not produce a cleaner or more convincing locomotion style than the
  current baseline

Conclusion:

- this was a useful hypothesis test
- direct dynamics supervision plus a latent bridge was more principled than
  direct teacher-latent imitation
- but it still did **not** convert into a runtime winner
- therefore it should be archived as another informative non-winning C2 branch

Meaning for next work:

- the remaining blocker is likely deeper than:
  - `phi` optimizer choice
  - latent target geometry alone
- future C2 work should now focus on:
  - a deeper controller/root rebuild
  - a richer deployable observability contract
  - or a more serious rethink of explicit C2 adaptation as the main path
