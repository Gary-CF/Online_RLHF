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
python -m pytest tests/cpu -q -ra --strict-markers   # expect 76 passed, 3 xfailed
ruff check tests/                                     # pinned via requirements/lint.txt
pre-commit run --all-files                            # optional; same ruff, tests/ only
git diff a5a813106fb6501d32a7d7c9ed2752e36901673e -- openrlhf pipeline merge_peft.py requirements.txt  # must be empty
```

## Known-failure policy

- The 3 strict xfails are ONE test (`test_cg_solves_spd_system_within_dimension_steps`)
  instantiated on ALL THREE HVP trainers (two head + full-model); all document
  the same confirmed legacy defect CG-1 (`docs/maintenance/review.md`). They
  must fail for the documented reason; an XPASS means the defect was fixed —
  un-xfail the test, never loosen tolerances or broaden the xfail scope.
- CG-2 (squared-residual `residual_tol` semantics) is pinned by a normally
  PASSING test, not by an xfail.
- Do not mock gradients, HVP or CG outputs in these tests; the real
  implementations are the objects under test.
