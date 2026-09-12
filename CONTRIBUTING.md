# Contributing

## Running the CPU checks

The CPU tests in `tests/cpu/` characterize the numerical behavior of the
legacy reward-model code and are isolated from the training stack (no
transformers/DeepSpeed/GPU needed). Use a dedicated virtualenv, e.g.:

```bash
python3.12 -m venv /path/to/venv
/path/to/venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
/path/to/venv/bin/pip install -r requirements/cpu-test.txt
/path/to/venv/bin/python -m pytest tests/cpu -q -ra --strict-markers
```

Lint (new maintenance code under `tests/` only, pinned version). Use the
same dedicated venv as above, or a separate lint venv — always call the
interpreter by path (never a bare `pip`/`ruff`, which could install into or
run from your base environment):

```bash
python3.12 -m venv /path/to/lint-venv
/path/to/lint-venv/bin/pip install -r requirements/lint.txt
/path/to/lint-venv/bin/ruff check tests/
```

Optional git hooks (offline local hooks, same pinned ruff, `tests/` only;
installing writes `.git/hooks`, so it stays a user-run step):

```bash
/path/to/lint-venv/bin/pip install pre-commit
/path/to/lint-venv/bin/pre-commit install
/path/to/lint-venv/bin/pre-commit run --all-files
```

Expected result on `fix/cg-fletcher-reeves-beta`: all tests pass with **0
xfail** (90 passed as of round 3). The pre-fix CG-1 behavior is documented in
`docs/maintenance/review.md` and guarded by
`test_cg_converges_on_conjugate_direction_system`; if that test ever fails,
the beta fix was silently reverted — restore it instead of weakening the
test.

## Preparing small changes

- Keep production code under `openrlhf/`, `pipeline/` and `merge_peft.py`
  read-only unless the change has been discussed; the CPU tests must keep
  passing without installing the training stack.
- Do not run repo-wide formatters or bulk import cleanups.
- See `docs/maintenance/environment.md` for the verified environment and
  `docs/maintenance/review.md` for the review baseline and known defects.

## What needs discussion before changing

- Anything that alters algorithm behavior: the conjugate-gradient `beta`
  computation, `residual_tol` semantics, the damping schedules and their
  wiring into the HVP, and the pair-selection rules in
  `openrlhf/utils/convert_to_dataset.py`.
- Training defaults (learning rates, batch sizes, sampling budgets, output
  paths) and the pinned `transformers==4.46.3` in the root requirements.
