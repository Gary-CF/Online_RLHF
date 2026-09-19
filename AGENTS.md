# AGENTS.md — collaboration boundaries

## Long-term contribution guidance

These are local fork maintenance practices, not policies accepted by the paper
authors or upstream. Keep changes small, preserve existing user work, and discuss
algorithm/interface decisions before expanding scope. CPU tests must use real
production numerics through file-path isolation, controlled package shells and
the DeepSpeed fail-fast sentinel. Only the full-model GatheredParameters gate
gets a scoped empty context. Never mock gradients, HVP or CG outputs.

## Temporary authorization: round 4

This round starts from `14bbff0de58b765a7e4303a85af0fd1fc032b89d` on
`fix/cg-fletcher-reeves-beta`. The user's explicit round-4 authorization
supersedes older read-only restrictions only for the following scope:

- Tests under `tests/cpu/`, `tests/support/`, `pytest.ini`; maintenance CI,
  hooks and Ruff configuration; `docs/maintenance/`, CONTRIBUTING and this file.
- README maintenance entry/scope only; .gitignore only narrow personal-task
  filename fixes, without deleting or renaming files.
- The five HVP CLIs: only cg_mixing_weight argument definition/help and explicit
  None fallback after parsing. No parser refactor or new validation/warnings.
- Three HVP trainers: docstrings/comments only; executable AST must be unchanged.

Do not change dependencies, pipeline/**, merge_peft.py, root requirements.txt,
LICENSE, score/selection code or any other production behavior. In particular,
beta, alpha's fixed epsilon, stopping, schedules/normalization/HVP wiring/gates,
mixing rules, autograd, optimizer and distributed behavior stay unchanged.
This temporary permission is not standing authorization for future rounds.

## Git division of labor

The user alone performs ALL Git writes: staging, commits, pushes, branch/worktree
creation or switching, reset, merge, cherry-pick, rebase, stash, clean, remotes
and config changes, hooks installation and PR creation. Agents may inspect Git
read-only and use git archive for snapshots outside the repository.
This round leaves changes uncommitted and never switches a dirty worktree.

## Check commands

Run from the repository in dedicated environments; never conda base:

```bash
/home/gary/venvs/online_rlhf_cpu/bin/python -m pytest tests/cpu -q -ra --strict-markers
/home/gary/venvs/online_rlhf_lint/bin/ruff check tests/
/home/gary/venvs/online_rlhf_lint/bin/ruff format --check tests/
/home/gary/venvs/online_rlhf_cpu/bin/python -m pip check
git diff --check
```

Expect no failed tests, unexpected skips or xfails on the fix branch. Actual
counts and versions belong in docs/maintenance/environment.md, not a fixed
acceptance count here. Optional hooks require the lint venv bin on PATH;
agents must not install them.

## Known-failure policy

- CG-1 is fixed on this branch. A regression requires diagnosis of environment,
  import/numerical paths and source changes; failure alone does not prove beta
  was reverted. Never relax tolerances without evidence or hide it with xfail.
- CG-2's squared-residual semantics are characterized by a passing test.
- CG-3 is an observed pre-existing scale sensitivity, not a newly agreed accuracy
  contract or failing CI test. Author decisions are recorded in review.md.
- A strict xfail is appropriate only for an agreed, reproducible known defect,
  with a specific reason and narrow numerical assertion. Unexpected passes must
  trigger review/removal of the marker, not be silently accepted.
