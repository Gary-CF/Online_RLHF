# Maintenance review — rounds 1–4

Paper-repository maintenance baseline: `a5a813106fb6501d32a7d7c9ed2752e36901673e`
Historical infrastructure branch: `chore/maintenance-round1` at `024546c`.
Round-3/fix snapshot and round-4 starting HEAD: `14bbff0de58b765a7e4303a85af0fd1fc032b89d`.
Earlier round summaries below retain their historical scope; round-4 evidence follows.
Upstream reference: <https://github.com/ZinYY/Online_RLHF> (paper:
<https://arxiv.org/html/2502.07193v2>); read-only reference project:
<https://github.com/OpenRLHF/OpenRLHF>.
Provenance note: the historical fork-base of the paper repo
(ZinYY/Online_RLHF) relative to the OpenRLHF reference project is
**unknown** from this checkout — no local command evidence establishes their
common-history point, so none is claimed. This says nothing about the
Gary-CF fork's own relationship to ZinYY/Online_RLHF; that fork lineage is
as recorded by the git remotes.

## Historical scope delivered in rounds 1–3

- README corrections (usage instructions only).
- CPU test suite for the legacy reward-model HVP code:
  HVP, conjugate gradient, damping schedule, pair-selection helpers.
- Round 2 additions: full-model HVP trainer coverage (`rm_trainer_hvp`),
  conversion-pipeline and score-selection tests, isolation-adapter and
  RNG-guard regression tests, and the damping-wiring audit below.
- Round 3 additions (branch `fix/cg-fletcher-reeves-beta`): CG-1 fixed in the
  three HVP trainers (one-line beta correction each), `--cg_mixing_weight`
  CLI flag in two entries (remaining entries completed in round 4), "apo" alias and `one_step_embedding_l2_score` naming in
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
- Mixing context: default w=0.8 is the coefficient in
  `w * grad + (1 - w) * x_cg`, not a guarantee that 80% of the final
  vector norm comes from the raw gradient. Vector magnitude and alignment
  matter. The impact on historical experiments has not been quantified.
  The new flag allows an explicit pure CG direction when CG is called.
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
  `x = w * flat_grad + (1 - w) * x_cg`, with w falling back to
  args.damping when cg_mixing_weight is missing/None, whenever
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
  schedule is at/above 1.0, training takes the raw-gradient path. Permanent
  fallback requires a schedule that stays there (e.g. the current clamped
  monotone schedules under suitable settings); cosine can descend below 1.0
  later and re-enter HVP. These line references describe the earlier audit.
- Full-model trainer: `current_damping` is computed at `rm_trainer_hvp.py:256`
  but used ONLY for printing/logging (`:273`, `:313`) — the schedule has no
  control-flow effect there (behavioral drift vs. the head trainers).
- `--damping` has double duty in the CLI (`train_rm_head_hvp.py:289`,
  `train_rm_hvp.py`): it is both (a) the schedule base value / HVP
  regularization `cg_damping`, and (b) the mixing weight that blends the CG
  output with the raw gradient. With the CLI default `0.8`, the raw-gradient term has coefficient 0.8;
  this is not a vector-norm share. Round-3 update
  (D-3b implemented): the new `--cg_mixing_weight` flag decouples (b) from
  (a); when omitted it defaults to `--damping`, preserving that mixing
  rule. The beta correction still changes multi-step solver output.
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
  `one_step_fisher_score` kept as a documentation-deprecated compatibility
  alias, without a runtime warning. Accepting apo is an API expansion, not
  strictly zero behavior change; pipeline choices still do not accept apo.
- `convert_to_preference_dataset` iterates a `set` intersection, so output row
  order is non-deterministic; tests assert sets/counts only.

## Pending author decisions

Implemented on `fix/cg-fletcher-reeves-beta` (round 3):

- D-1 (CG-1): FIXED — standard new/old squared-residual beta in all three
  trainers; xfails converted to passing tests plus an anti-revert test.
- D-3b: IMPLEMENTED — `--cg_mixing_weight` decouples the mixing weight from
  `--damping`; the omitted flag preserves the mixing rule, not the
  pre-beta-fix multi-step numerical output.
- D-5: IMPLEMENTED — "apo" alias for "uncertainty_score";
  `one_step_embedding_l2_score` canonical name with documentation-deprecated
  `one_step_fisher_score` alias (no runtime warning).

Still open (no production change made; evidence preserved above):

- D-2 (CG-2 semantics): keep squared-residual comparison (now documented in
  the solver docstrings) or compare `sqrt(r_norm_sq)` against `residual_tol`.
- D-3a: should the full-model trainer honor the schedule gate like the head
  trainers?
- D-3c (three distinct decisions): (1) the schedule formulas, (2) time
  normalization (growth rate vs total_T), and (3) whether/how to feed the
  scheduled value into HVP regularization. Merely wiring a value into HVP
  would not establish agreement with the paper; all three need author review.
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

## Round 4 — local maintenance closeout (2026-09-18)

Starting repository: /home/gary/projects/Online_RLHF; branch
`fix/cg-fletcher-reeves-beta`; HEAD
`14bbff0de58b765a7e4303a85af0fd1fc032b89d`.
Starting index/worktree/untracked status was clean. Ignored personal task files
were present and retained. No Git writes, hooks installation, commits, pushes,
branch movement or PR actions were performed. This round is uncommitted.

### Implemented scope and evidence level

- Added the existing cg_mixing_weight definition/help and explicit None fallback
  to active_train_rm_head_hvp.py, online_train_rm_head_hvp.py and
  online_train_rm_hvp.py. train_rm_head_hvp.py and train_rm_hvp.py already had
  correct implementations and are unchanged.
- All five direct entry points now pass four real parameter-block cases with
  damping=0.37: omitted -> 0.37, explicit 0.0 -> 0.0, 0.3 -> 0.3, 1.0 -> 1.0.
  tests/support/cli_args.py executes the source AST for the contiguous parser
  constructor, every add_argument, parse_args and the immediate real fallback.
  Structural mismatches fail immediately. It supplies real argparse/datetime
  and required builtins (datetime.strftime needs the standard importer);
  it does not import the training module, load a model or call train().
- These are **parameter-block tests**, not full CLI startup or training integration.
  The historical 90 tests exercised isolated trainer/helper numerics and adapter
  behavior, not CLI parsing. The current suite has 113 parameterized instances:
  all original 90 retained, 20 CLI matrix instances and 3 new single-step mixing
  instances. The existing fallback test now also checks explicit None on each
  trainer. Counts are not counts of independent behaviors or project coverage.
- Three real trainers compare missing, None and explicit damping for equal
  current-solver output; explicit zero and intermediate linear blending remain
  covered; two weights (0 and 1) give identical one-step output distinct from
  the gradient. torch.equal compares those current-solver paths, not an old
  pre-beta-fix implementation. The beta fix changes multi-step CG results.
- The existing file-path adapter, controlled shells, DeepSpeed fail-fast sentinel
  and full-model-only GatheredParameters empty context were retained unchanged.
  Numerical gradients/HVP/CG outputs were not mocked or copied.
- Solver docstrings now state the SPD assumption, squared-residual stopping
  threshold and possible mixed return. Executable AST for all three trainers
  matches the starting snapshot exactly (docstrings excluded); beta is untouched.
- Ruff formatting is limited to tests/. CI now checks lint AND format, adds
  pip check/version summaries, accepts all push branches and cancels stale runs
  for the same workflow/ref. Python 3.12, CPU torch installation, permissions,
  timeouts and offline variables are retained; the CPU job does not install Ruff.
  Optional system hooks filter Python files and require the lint venv on PATH.
- README changes only the maintenance entry/scope. Previously corrected PyTorch,
  Citation, line breaks and stop_t are complete, not new pending work.

### Source-level argument route (not an integration run)

Each CLI calls get_strategy(args) at line 20. In utils/utils.py:26–38 the
same namespace enters DeepspeedStrategy(args=args); utils/deepspeed/deepspeed.py:53
stores self.args = args. The trainer is constructed with strategy=strategy:

| CLI | strategy=strategy wiring | cg_damping remains args.damping | Trainer self.args |
| --- | --- | --- | --- |
| train_rm_head_hvp.py | line 159 | line 168 | rm_trainer_head_hvp.py:52 |
| train_rm_hvp.py | line 131 | line 140 | rm_trainer_hvp.py:44 |
| active_train_rm_head_hvp.py | line 159 | line 168 | rm_active_trainer_head_hvp.py:52 |
| online_train_rm_head_hvp.py | line 156 | line 165 | rm_trainer_head_hvp.py:52 |
| online_train_rm_hvp.py | line 133 | line 142 | rm_trainer_head_hvp.py:52 |

Thus the parsed mixing value reaches trainer self.args through strategy.args;
solver getattr/None fallback reads it. This conclusion is static source evidence
plus separate parameter-block and numerical tests, not end-to-end training proof.
Mixing is applied only for max_iter > 1. Head trainers can still bypass CG through
their schedule gate; w=0 does not guarantee CG at every training step. Cosine can
fall below 1 and re-enter HVP. Pipeline wrappers are outside scope and do not
directly forward the new flag.

### Targeted reverse validation

Only external copies under /tmp/online-rlhf-round4-jhlg7_n5 were mutated.
The active CLI was selected once per key risk; the other 16 CLI instances were
deselected, not unexpectedly skipped.

- missing-flag copy: remove the real add_argument statement -> 4 failed,
  16 deselected, exit 1 (missing attribute / unrecognized option).
- or-fallback copy: replace the explicit None expression with
  args.cg_mixing_weight or args.damping -> 1 failed, 3 passed, 16 deselected,
  exit 1; explicit zero incorrectly became 0.37.
- Command in each copy:
  /home/gary/venvs/online_rlhf_cpu/bin/python -m pytest tests/cpu/test_cli_args.py -q -ra --strict-markers --tb=short -k active_train_rm_head_hvp
- The working repository's production code was never broken for this experiment.
  No persistent mutation framework was added.

### CG-3: existing scale sensitivity / numerical contract to discuss

The fixed alpha denominator addition of 1e-8 can dominate a small-curvature
product even for an SPD system. This is an observation, not a newly imposed
accuracy contract and not proof of actual training harm. No epsilon, tolerance,
stopping, beta, iteration order or return calculation was changed. No failing CI
test, xfail or vacuous passing test was added for CG-3; the author must first
decide the intended numerical contract.

Reproduced with real autograd through the existing isolation adapter: float64,
one Parameter theta=0, loss=0.5*theta.square().sum()-1e-6*theta.sum(),
cg_damping=0, mixing=0, residual_tol=1e-30. H=1, grad=-1e-6, hence exact
linear-system solution x=-1e-6. All three current trainers returned:

| max_iter | x | relative error |
| --- | --- | --- |
| 1 | -9.999000099990e-11 | 0.9999000100 |
| 3 | -5.992511809761e-10 | 0.9994007488 |
| 10 | -5.423376506746e-09 | 0.9945766235 |

The 202d4cd pre-beta-fix source already contains the same +1e-8.
A git archive snapshot outside the repo also reproduces the sensitivity in all
three trainers: at 1/3/10 steps, x is respectively -9.999000099990e-11,
-5.993707321216e-10, -5.439184320996e-09 (relative errors 0.9999000100,
0.9994006293, 0.9945608157). Thus the scale issue predates beta; multi-step
values need not be bit-identical across that fix.

Reproduction from the repository root with the dedicated CPU interpreter
(save this snippet outside the repo, or supply it on stdin):

```python
import types
import torch
from tests.support.legacy_imports import (
    isolated_legacy_imports, load_trainer, make_trainer_shell, trainer_hvp_gate,
)

for kind in ("rm", "rm_active", "rm_hvp"):
    with isolated_legacy_imports():
        module = load_trainer(kind)
        for steps in (1, 3, 10):
            theta = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
            loss = 0.5 * theta.square().sum() - 1e-6 * theta.sum()
            grad = torch.autograd.grad(loss, [theta], create_graph=True)[0].reshape(-1)
            trainer = make_trainer_shell(
                module, cg_damping=0.0,
                args=types.SimpleNamespace(damping=0.0, cg_mixing_weight=0.0),
            )
            with trainer_hvp_gate(kind):
                x = trainer.conjugate_gradient_solver(
                    [theta], loss, grad, max_iter=steps, residual_tol=1e-30,
                )
            print(kind, steps, x.item(), abs((x.item() + 1e-6) / 1e-6))
```

Actual executed command:
`/home/gary/venvs/online_rlhf_cpu/bin/python /tmp/online-rlhf-round4-jhlg7_n5/cg3.py`
from the working repo and separately the pre-beta archive; both exit 0.
The saved script adds the working directory to sys.path for external-script use.

### Branch integrity and remaining user decisions

Read-only commit-content checks confirm main=a5a813106fb6501d32a7d7c9ed2752e36901673e,
chore/maintenance-round1=024546c95b1a945365baa04091e923f589418cbb and
202d4cde0b85c9c189f8bdb593a19976b044da47 is an ancestor of the current HEAD.
At 024546c conftest imports load_rm_score_selection but the adapter lacks it;
202d4cd defines the loader and full-model gate. The earlier independent review
reported collection exit 4 at 024546c and 76 passed/3 xfailed at 202d4cd; those
historical full-suite runs were not repeated here. Source content confirms the
incomplete old branch reference. Fixing current files does not repair that ref.

A future infrastructure PR must include at least the complete content through
202d4cd. The user must decide how to organize it later; no branch was moved,
no beta changes were merged into the infrastructure branch and no history was
rewritten. First review/save this round on the current fix branch; do not switch
while these changes are uncommitted. Future boundaries: infrastructure, pure CG
correctness, optional interfaces. 5ac11bf mixed beta and flag changes, so future
separation must be based on final diffs, not blindly reusing that commit.

D-2, D-3a, all three parts of D-3c, D-4 and CG-3 remain for author decision.
The apo API expansion/compatibility aliases from round 3 are unchanged; aliases
have no runtime warning and pipeline choices still do not accept apo.
This round's authorization and Git division of labor are personal fork-stage
agreements, not upstream-approved policy.

### Personal filenames

Actual local attachments use ASCII ":" before Zone.Identifier; no U+F03A
attachments were found. git check-ignore -v --stdin confirms all four existing
personal task files/attachments match the existing narrow rules. No deletion,
rename or .gitignore edit was needed, and no U+F03A cleanup is claimed.

### Validation status

See environment.md for authoritative commands, versions, exit codes and the
distinction between local results, verified historical remote CI and this
uncommitted round's pending remote validation. Final scope checks retain all
original test IDs/assertions, leave protected files unchanged and restrict
production executable edits to the three authorized CLI parameter blocks.
