# Contributing

## Running the CPU checks

The CPU suite tests real legacy reward-model numerics through isolated file-path
imports and separately executes the five real CLI argparse blocks. Parameter-block
tests do not start complete CLI modules, load models or integrate training.
See [environment.md](docs/maintenance/environment.md) for actual versions/results
and [review.md](docs/maintenance/review.md) for findings and unresolved decisions.

Use a dedicated CPU virtualenv; never conda base or the full training requirements:

```bash
python3.12 -m venv /path/to/cpu-venv
/path/to/cpu-venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
/path/to/cpu-venv/bin/pip install -r requirements/cpu-test.txt
/path/to/cpu-venv/bin/python -m pip check
/path/to/cpu-venv/bin/python -m pytest tests/cpu -q -ra --strict-markers
```

Use a separate lint virtualenv with the pinned Ruff version:

```bash
python3.12 -m venv /path/to/lint-venv
/path/to/lint-venv/bin/pip install -r requirements/lint.txt
/path/to/lint-venv/bin/python -m pip check
/path/to/lint-venv/bin/ruff check tests/
/path/to/lint-venv/bin/ruff format --check tests/
```

The pins cover maintenance tools and direct CPU test dependencies. They do not
lock the full training stack or all transitive dependencies, and do not establish
complete training reproducibility. Formatting is limited to tests/.

## Optional local hooks

Hooks are optional system hooks, use the same pinned Ruff and select only Python
files under tests/. Install them only if the user chooses to write .git/hooks.
Activate the lint venv (or put its bin directory first on PATH): invoking
pre-commit by absolute path alone does NOT ensure its system hooks find that Ruff.

```bash
source /path/to/lint-venv/bin/activate
python -m pip install pre-commit
ruff --version
pre-commit install
pre-commit run --all-files
```

Expected: Ruff matches requirements/lint.txt; hook installation succeeds; both
Ruff checks pass. The check hook can apply lint fixes; inspect the diff afterwards.
None of these optional installation steps was run by the agent in round 4.

On the fix branch expect no failures, unexpected skips or xfails. If a CG
regression fails, inspect the environment, actual loaded implementation, numerical
path and source changes before attributing it to a beta revert. Do not relax
tolerances without evidence or mark unexpected failures xfail to obtain green CI.
Strict xfails remain reserved for agreed, reproducible defects; an XPASS requires
review. CG-3 is a discussion item, not an agreed accuracy contract.

## Preparing small changes

These are local fork practices, not upstream-approved policy. Keep algorithm
changes, maintenance infrastructure and optional interfaces logically separate.
The current round's narrow authorization is in AGENTS.md; it is not a standing
permission to change production behavior.

- Preserve user changes; do not run repository-wide formatters or import cleanups.
- The user performs all Git writes, branch/worktree operations, hooks installation
  and PR actions. Agents leave their work uncommitted.
- README PyTorch/Citation/line-break/stop_t corrections are already complete.
- Local tests do not prove that the revised GitHub Actions workflow has run.
  Remote validation follows a user-controlled commit/push.

## What needs discussion before changing

CG beta, alpha epsilon, residual tolerance semantics, damping formulas/time
normalization/HVP wiring and head/full gates, mixing rules, selection/randomness,
training defaults and dependency changes require discussion. D-2, D-3a, the three
parts of D-3c and D-4 remain open in review.md.

All five direct HVP CLIs accept --cg_mixing_weight. Its None fallback preserves
the damping-based mixing rule; the earlier beta fix still changes multi-step CG
outputs. A weight of zero controls mixing only when the solver is called; the
head schedule gate can bypass it. Pipeline wrappers do not forward this new flag.
The earlier apo alias adds an accepted API value; pipeline choices still reject
apo and compatibility score aliases have no runtime deprecation warning.
