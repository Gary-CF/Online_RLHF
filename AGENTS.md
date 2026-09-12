# AGENTS.md — collaboration boundaries

## Write scope (this maintenance round)

- Allowed: `README.md` (wording fixes), `tests/cpu/`, `tests/support/`,
  `pytest.ini`, `requirements/cpu-test.txt`, `requirements/lint.txt`,
  `ruff.toml`, `.pre-commit-config.yaml`,
  `.github/workflows/cpu-checks.yml`, `CONTRIBUTING.md`,
  `docs/maintenance/`, `.gitignore`, this file.
- Read-only: `openrlhf/**`, `pipeline/**`, `merge_peft.py`, root
  `requirements.txt` (including `transformers==4.46.3`).
- If a CPU test cannot run without touching production code, use the
  test-side isolation adapter (`tests/support/legacy_imports.py`) instead.

## Git division of labor

- The user creates branches/worktrees, stages, commits, merges, pushes and
  opens PRs. Agent work stays uncommitted unless the user asks otherwise.
- Read-only git inspection is fine; never stash/reset/rebase/amend/force-push
  or change remotes/global config.

## Check commands

```bash
python -m pytest tests/cpu -q -ra --strict-markers   # expect 90 passed, 0 xfailed (on fix/cg-fletcher-reeves-beta)
ruff check tests/                                     # pinned via requirements/lint.txt
pre-commit run --all-files                            # optional; same ruff, tests/ only
git diff chore/maintenance-round1 -- openrlhf/        # on the fix branch: ONLY the round-3 authorized changes
git diff a5a813106fb6501d32a7d7c9ed2752e36901673e -- pipeline merge_peft.py requirements.txt  # must be empty
```

## Known-failure policy

- CG-1 is FIXED on `fix/cg-fletcher-reeves-beta`: the suite is fully green
  (0 xfail). `test_cg_solves_spd_system_within_dimension_steps` and
  `test_cg_converges_on_conjugate_direction_system` pin the corrected
  Fletcher-Reeves beta; if they fail, the fix was reverted — restore it,
  never loosen tolerances.
- CG-2 (squared-residual `residual_tol` semantics) is pinned by a normally
  PASSING test and documented in the solver docstrings; changing the
  semantics is an author decision (D-2), not a maintenance action.
- Do not mock gradients, HVP or CG outputs in these tests; the real
  implementations are the objects under test.
