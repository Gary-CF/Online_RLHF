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
- CPU test suite for the legacy reward-model HVP code:
  HVP, conjugate gradient, damping schedule, pair-selection helpers.
- Round 2 additions: full-model HVP trainer coverage (`rm_trainer_hvp`),
  conversion-pipeline and score-selection tests, isolation-adapter and
  RNG-guard regression tests, and the damping-wiring audit below.
- Round 3 additions (branch `fix/cg-fletcher-reeves-beta`): CG-1 fixed in the
  three HVP trainers (one-line beta correction each), `--cg_mixing_weight`
  CLI flag, "apo" alias and `one_step_embedding_l2_score` naming in
  `rm_score_selection.py`, and solver docstrings documenting the
  squared-residual `residual_tol` semantics.
- Minimal CI (lint of new code + CPU pytest) and maintenance docs.
- Production modifications are limited to the round-3 authorization list;
  `pipeline/`, `merge_peft.py` and the root `requirements.txt` remain
  untouched (verified via `git diff`).

## Confirmed defects (observed behavior -> impact -> boundary -> next step)

### CG-1: conjugate-gradient `beta` equals 1.0 whenever its denominator is nonzero — FIXED

- Where (pre-fix): `rm_trainer_head_hvp.py`, `rm_active_trainer_head_hvp.py`
  and `rm_trainer_hvp.py`, method `conjugate_gradient_solver` — the line
  `beta = r_norm_sq / r_norm_sq` (both operands the freshly updated squared
  residual).
- Observed behavior (pre-fix): the Fletcher-Reeves beta — the ratio of the
  NEW to the OLD squared residual — was computed as new/new, i.e. exactly 1.0
  whenever its denominator was nonzero, so the search-direction update became
  `p = r + p` instead of the standard `p = r + (r_new.r_new/r_old.r_old) * p`.
  Measured on the audit fixture (seed=7, Z `(16,4)` float64, column scales
  `[0.5,1,2,3]`, theta=`[0.1,-0.2,0.05,0.15]`, rejected=`0`, `cg_damping=0.2`,
  mixing weight 0): 4-step relative residual `0.231158` on all three
  trainers; 8 steps `0.191879`; 16 steps `0.097288`.
- Fix (branch `fix/cg-fletcher-reeves-beta`): save the old squared residual
  before the update and compute `beta = r_norm_sq_new / r_norm_sq_old`, in
  all three trainers, matching the paper's Algorithm 5
  (beta_{k+1} = r_{k+1}ᵀr_{k+1} / r_kᵀr_k). Nothing else in the solver
  changed.
- Post-fix measurement (same fixture, 4 steps): relative residual
  `4.493e-06`, relative solution error `7.672e-06` on all three trainers;
  the previously-xfailed test now passes normally and a dedicated
  anti-revert test (`test_cg_converges_on_conjugate_direction_system`) pins
  the corrected recurrence.
- Mitigating context (kept on record): under the DEFAULT CLI configuration
  the mixing weight is 0.8, so every multi-step CG update direction is 80%
  raw gradient — the practical impact of CG-1 on already-run experiments was
  likely diluted. It remains a genuine implementation deviation from
  Algorithm 5; the new `--cg_mixing_weight` flag (D-3b) lets users set the
  pure CG direction explicitly.
- Verification boundary: unit-level, CPU float64, isolated imports. No GPU
  training was run; impact on full training runs is unquantified here.

### CG-2: `residual_tol` compares against the squared residual

- Where: same methods, loop break condition `if r_norm_sq < residual_tol`.
- Observed behavior: the threshold is compared with the SQUARED residual
  norm, so `residual_tol=1e-10` actually means residual norm < 1e-5.
- Impact: callers intending a residual-norm tolerance get a looser stop by
  construction. Documented; pinned by
  `test_cg_residual_tol_compares_squared_residual`.
- Round-3 update: the three solvers now carry a docstring stating the
  squared-residual semantics explicitly (documentation only; the comparison
  itself is unchanged). Whether to change the semantics (compare
  `sqrt(r_norm_sq)`) remains an open author decision.

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

## Round 2: damping-wiring audit (read-only, with file:line evidence)

- The HVP regularization term is ALWAYS the constructor-time constant
  `cg_damping` (`rm_trainer_head_hvp.py:154`, `rm_active_trainer_head_hvp.py:128`,
  `rm_trainer_hvp.py:116`). `get_current_damping()`'s scheduled value never
  enters the HVP/CG mathematics in any trainer.
- Head trainers: the schedule gates the update path only —
  `if self.args.use_hvp and current_damping < 1.0`
  (`rm_trainer_head_hvp.py:300`, `rm_active_trainer_head_hvp.py:345`). Once the
  schedule reaches 1.0, training falls back to the raw gradient for the rest
  of the run.
- Full-model trainer: `current_damping` is computed at `rm_trainer_hvp.py:256`
  but used ONLY for printing/logging (`:273`, `:313`) — the schedule has no
  control-flow effect there (behavioral drift vs. the head trainers).
- `--damping` has double duty in the CLI (`train_rm_head_hvp.py:289`,
  `train_rm_hvp.py`): it is both (a) the schedule base value / HVP
  regularization `cg_damping`, and (b) the mixing weight that blends the CG
  output with the raw gradient. With the CLI default `0.8`, every multi-step
  CG update direction is 80% raw gradient by construction. Round-3 update
  (D-3b implemented): the new `--cg_mixing_weight` flag decouples (b) from
  (a); when omitted it defaults to `--damping`, so default behavior is
  unchanged.
- CLI defaults (`train_rm_head_hvp.py:287-294`): `--use_hvp` False,
  `--damping` 0.8, `--damping_strategy` linear, `--damping_growth_rate` 100,
  `--num_cg_steps` 3.
- Only the full-model trainer's HVP actually calls
  `deepspeed.zero.GatheredParameters` (`rm_trainer_hvp.py:101`); in the head
  trainers that call is commented out. The CPU tests pass this gate via a
  scoped no-op context manager that mocks NO numerics (see
  `tests/support/legacy_imports.py`).

## Round 2: additional observations (pinned, not classified as defects)

- `rm_trainer_head.py`, `rm_active_trainer_head.py` and
  `rm_trainer_head_NewtonStep.py` do not implement hessian_vector_product /
  conjugate_gradient_solver / get_current_damping at all; no symbol drift
  exists among the three HVP trainers (HVP implementations differ only by the
  deepspeed gate).
- `get_score_fn` docstring listed "apo" as an option but the dispatcher key
  was "uncertainty_score"; requesting "apo" raised ValueError. Also,
  `one_step_fisher_score` is named "Fisher" but computes the squared L2
  distance of chosen/rejected embeddings. Round-3 update (D-5 implemented):
  "apo" is now a working alias of "uncertainty_score", and
  `one_step_embedding_l2_score` is the canonical name with
  `one_step_fisher_score` kept as a deprecated alias.
- `convert_to_preference_dataset` iterates a `set` intersection, so output row
  order is non-deterministic; tests assert sets/counts only.

## Pending author decisions

Implemented on `fix/cg-fletcher-reeves-beta` (round 3):

- D-1 (CG-1): FIXED — standard new/old squared-residual beta in all three
  trainers; xfails converted to passing tests plus an anti-revert test.
- D-3b: IMPLEMENTED — `--cg_mixing_weight` decouples the mixing weight from
  `--damping`; omitted flag reproduces previous behavior exactly.
- D-5: IMPLEMENTED — "apo" alias for "uncertainty_score";
  `one_step_embedding_l2_score` canonical name with deprecated
  `one_step_fisher_score` alias.

Still open (no production change made; evidence preserved above):

- D-2 (CG-2 semantics): keep squared-residual comparison (now documented in
  the solver docstrings) or compare `sqrt(r_norm_sq)` against `residual_tol`.
- D-3a: should the full-model trainer honor the schedule gate like the head
  trainers?
- D-3c: should the scheduled damping value feed the HVP instead of the
  constant `cg_damping`?
- D-4 (selection semantics): `best_quartile` fixed 4th-best vs. a true
  quartile; `get_random_indices` uses the global `random` module.

## User review checklist

1. README diff — wording-only fixes to existing instructions.
2. Tests really call the production implementations (file-path loading from
   this checkout; see `tests/support/legacy_imports.py`). On the fix branch
   the suite is fully green (0 xfail); the pre-fix CG-1 behavior stays
   documented above with measured residuals, and
   `test_cg_converges_on_conjugate_direction_system` guards against silently
   reverting the beta fix.
3. Production changes on the fix branch are exactly the authorized set —
   review them hunk by hunk:
   `git diff chore/maintenance-round1 -- openrlhf/`. `pipeline/`,
   `merge_peft.py` and the root `requirements.txt` have no diff.
