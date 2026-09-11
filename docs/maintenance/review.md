# Maintenance review — round 1

Baseline: `a5a813106fb6501d32a7d7c9ed2752e36901673e`
Branch: `chore/maintenance-round1`
Upstream reference: <https://github.com/ZinYY/Online_RLHF> (paper:
<https://arxiv.org/html/2502.07193v2>); read-only reference project:
<https://github.com/OpenRLHF/OpenRLHF>.
Provenance note: the historical fork-base of the paper repo
(ZinYY/Online_RLHF) relative to the OpenRLHF reference project is
**unknown** from this checkout — no local command evidence establishes their
common-history point, so none is claimed. This says nothing about the
Gary-CF fork's own relationship to ZinYY/Online_RLHF; that fork lineage is
as recorded by the git remotes.

## Scope delivered this round

- README corrections (usage instructions only).
- CPU test suite for the legacy reward-model head-HVP code:
  HVP, conjugate gradient, damping schedule, pair-selection helpers.
- Minimal CI (lint of new code + CPU pytest) and maintenance docs.
- No production code under `openrlhf/`, `pipeline/`, `merge_peft.py`, or the
  root `requirements.txt` was modified (verified via `git diff <baseline>`).

## Confirmed defects (observed behavior -> impact -> boundary -> next step)

### CG-1: conjugate-gradient `beta` equals 1.0 whenever its denominator is nonzero

- Where: `openrlhf/trainer/rm_trainer_head_hvp.py` and
  `openrlhf/trainer/rm_active_trainer_head_hvp.py`, method
  `conjugate_gradient_solver` — line computing `beta = r_norm_sq / r_norm_sq`
  (both operands are the freshly updated squared residual).
- Observed behavior: the Fletcher-Reeves beta — the ratio of the NEW to the
  OLD squared residual — is computed as new/new. Whenever that denominator
  is nonzero, `beta` is exactly 1.0 (an exactly-zero residual would give
  0/0), so the search-direction update becomes `p = r + p` instead of the
  standard form `p = r + (r_new.r_new / r_old.r_old) * p`.
- Impact: in exact arithmetic, standard linear CG on an SPD system converges
  within at most `dim` iterations; this implementation does not (float64
  tolerances apply to the fixture below). Reproduced locally on the audit
  fixture (seed=7, Z `(16,4)` float64 with column scales `[0.5,1,2,3]`,
  theta=`[0.1,-0.2,0.05,0.15]`, rejected=`0`, `cg_damping=0.2`,
  direction-mixing weight `args.damping=0`): relative residual of `(H+0.2I)x = g`
  after 4 steps is `0.231158` for BOTH trainers; after 8 steps `0.191879`,
  after 16 steps `0.097288`. The prior audit's "~0.231 in 4 steps" matches.
- Verification boundary: unit-level, CPU float64, isolated imports. No GPU
  training was run; the practical impact on full training is unquantified.
- Next step: author decision — fix `beta` to the standard ratio, or document
  the current recurrence as the intended algorithm.

### CG-2: `residual_tol` compares against the squared residual

- Where: same methods, loop break condition `if r_norm_sq < residual_tol`.
- Observed behavior: the threshold is compared with the SQUARED residual
  norm, so `residual_tol=1e-10` actually means residual norm < 1e-5.
- Impact: callers intending a residual-norm tolerance get a looser stop by
  construction. Documented; pinned by
  `test_cg_residual_tol_compares_squared_residual`.
- Next step: author decision — rename the knob or compare `sqrt(r_norm_sq)`.

## Behaviors pinned, not classified as defects

- Damping schedule (`get_current_damping`): with
  `damping_growth_rate > 1.0`, normalized time is `total_steps /
  damping_growth_rate` and `total_T` is ignored by the schedule.
  `constant` jumps to exactly 1.0 at `t >= 1.0`; `cosine` is
  `min(base * (2 - cos(pi t)), 1.0)` and falls back below 1.0 once `t`
  exceeds 1.0 (e.g. base=0.8, growth=100: steps 0/50/100/150/200 give
  0.8/1/1/1/0.8, re-verified locally). These tests pin current behavior; they
  do NOT prove the scheduled value is dynamically wired into the HVP
  (the HVP uses the constant `cg_damping`; schedule gating happens outside).
- `hessian_vector_product` adds `cg_damping * v` to the raw HVP and clears
  `param.grad` for all params as a side effect (pinned by tests; do not
  remove the clearing in production).
- `get_best_quartile_indices` returns the best and the 4th-best entries of
  the descending sort (fixed index 3), not a general quartile computation.
- `get_random_indices` uses the Python `random` module; identical seed and
  call sequence give identical pairs.
- `conjugate_gradient_solver` mixes the CG direction with the raw gradient:
  `x = args.damping * flat_grad + (1 - args.damping) * x` whenever
  `max_iter > 1` (no mixing for a single step). In the isolated tests
  `args.damping` is set to 0.0 to exclude the mixing; this is NOT the default
  CLI training configuration.

## User review checklist

1. README diff — wording-only fixes to existing instructions.
2. Tests really call the production implementations (file-path loading from
   this checkout; see `tests/support/legacy_imports.py`) and the 2 known
   failures are transparent strict xfails.
3. Production files are untouched: `git diff a5a8131 -- openrlhf pipeline
   merge_peft.py requirements.txt` is empty and no new untracked files were
   added under those paths.
