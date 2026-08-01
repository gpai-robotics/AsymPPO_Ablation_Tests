# C1 vs C2 Status Comparison

This note gives the current plain-language comparison between the repo's two
main deployment-facing lines:

- `C1`
- `C2`

It is meant to answer four questions clearly:

1. what each line is trying to be
2. where they differ structurally
3. who is currently better where
4. what each line still needs to prove

## One-line identity

- `C1`
  - blind history-conditioned deployable controller
  - robustness-first
  - no explicit deploy-time adaptive latent contract

- `C2`
  - explicit RMA-style adaptive deployable controller
  - deploy-time history-to-latent path
  - actor consumes current policy obs plus adaptive latent

## Architectural difference

## Teacher lineage difference

This is one of the most important high-level distinctions between the two
lines.

### C1

Current C1 should be described as:

- a blind deployable history-conditioned student
- derived from a teacher path that includes both:
  - terrain privilege
  - dynamics privilege

Careful wording:

- C1 is best described as **terrain+dynamics teacher-derived**
- not as proof that every historical C1 teacher used both sources equally well
  in every older branch

For the current active C1 line, this statement is grounded in the current
teacher-of-record story documented in:

- [C1_STAGEA_MODEL400_DEPLOY_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_STAGEA_MODEL400_DEPLOY_CARD.md)

### C2

Current C2 should be described as:

- an explicit adaptive student
- derived from a **dynamics-only** teacher/root contract

That means C2 is best described as:

- **dynamics-only teacher-derived**

This is not just an implementation detail.

It changes what each line is trying to infer:

- C1 inherits a broader upstream teacher story and then solves deployment as a
  blind history-conditioned policy
- C2 inherits a narrower but more explicit hidden-dynamics adaptation story and
  then tries to preserve that structure through `phi(history) -> z_hat`

### C1

Deploy contract:

```text
policy_obs + policy_history
  -> blind history policy
  -> action
```

Meaning:

- adaptation, if present, is implicit in the history-conditioned policy itself
- there is no separate runtime `mu / phi / z_hat` contract

### C2

Deploy contract:

```text
policy_history
  -> phi
  -> z_hat

policy_obs + z_hat
  -> pi
  -> action
```

Meaning:

- adaptation is explicit
- the branch is supposed to prove that deployable history can infer hidden
  dynamics through a dedicated latent path

## Current canonical artifacts

### C1

- [C1_STAGEA_MODEL400_DEPLOY_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_STAGEA_MODEL400_DEPLOY_CARD.md)
- exported bundle:
  `rma_go2_lab/policies/exported/c1_ethlike_v3_model_400_candidate/`

### C2

- [C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md)
- current structured offline candidate:
  `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`

## Where C1 is currently stronger

- deployment simplicity
  - simpler runtime contract
  - fewer moving parts
  - easier to explain and trust

- export/source parity maturity
  - C1 export parity is already audited and essentially exact

- MuJoCo deployment evidence depth
  - nominal
  - hidden-env mismatch
  - moderate disturbance
  - continuous corridor
  - history ablations

- proof of real time-varying history use
  - C1 moderate-disturbance MuJoCo ablations show:
    - `normal > frozen > zero`
  - this is the clearest current evidence in the repo that the time-varying
    temporal path is behaviorally load-bearing after deployment

- blind-reactive robustness framing
  - C1 is currently the cleaner answer if the question is:
    - "what is the best current blind deployable history-conditioned policy?"

## Where C2 is currently stronger

- explicit adaptation structure
  - C2 is the line that directly tests the RMA-style adaptation hypothesis

- hidden-factor specialization
  - low-friction hidden dynamics shifts are where C2 currently shows its most
    meaningful advantage over removing the latent entirely

- structured training pipeline progress
  - C2 now has a working frozen structured Phase 1 root
  - on-policy history collection works
  - offline supervised `phi` training works
  - runtime evaluation works

- adaptive research value
  - C2 remains the right line if the question is:
    - "can we build a deployable explicit online-adaptive controller rather
      than only a strong blind reactive one?"

## Current honest difference in behavior

### C1

Current best interpretation:

- strong blind reactive policy
- history is genuinely useful
- especially validated on:
  - Isaac switch/push suites
  - MuJoCo moderate disturbances
- main remaining weakness:
  - lateral push recovery under hard corridor geometry

### C2

Current best interpretation:

- adaptive latent is not useless
- but online latent updating is not yet strongly load-bearing
- recent C2 diagnostic supports:
  - weak motor:
    - `normal ~= frozen`
  - low friction:
    - `normal ~= frozen >> zero`

So C2 currently behaves more like:

- a policy using the latent as a coarse context code

than like:

- a policy whose success depends heavily on step-by-step online latent updates

## Who is better where

### C1 is better when the priority is

- a clean deployment-facing policy
- strong blind reactive robustness
- clearer evidence that the time-varying history path matters
- better audited source/export/runtime trust
- lower conceptual complexity

### C2 is better when the priority is

- preserving the explicit adaptive-RMA research question
- testing deployable hidden-dynamics inference directly
- studying friction/contact-specific adaptive behavior
- pushing toward a future controller that explains *why* adaptation helps, not
  only *that* a blind policy is robust

## Current winner by objective

If the objective is:

- "best current deployment-facing blind history policy"
  - `C1` is ahead

If the objective is:

- "best current explicit adaptive research line"
  - `C2` is still the right line

If the objective is:

- "best proven online-adaptive deployed controller in the strong RMA sense"
  - neither line is fully there yet
  - C1 is stronger as a deployable controller
  - C2 is stronger as the explicit adaptation research vehicle

## What each line still needs

### C1 still needs

- stronger corridor lateral-push recovery
- broader proof that its history-conditioned robustness stays strong under the
  hardest contact-rich deployment regimes

### C2 still needs

- stronger evidence that online latent updating is truly load-bearing
- better `phi ~= mu` style student-teacher agreement or a better replacement
  target
- stronger deployment-side behavior if it wants to beat the simpler C1-style
  story on practical grounds

## Bottom line

Current repo truth should be stated plainly:

- `C1` is the stronger current deployment candidate
- `C2` is the stronger current explicit adaptation research line

They are not failing or succeeding on the same question.

That distinction should be kept explicit whenever the two lines are compared.
