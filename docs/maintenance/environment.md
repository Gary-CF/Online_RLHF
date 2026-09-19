# CPU test environment and validation record

This document records the environment actually used to verify the CPU tests in
`tests/cpu/`. It is not the original authors' training environment.

## Historical environment (rounds 1–3)

- OS: Linux x86_64
- Python: CPython 3.12 (`/usr/bin/python3.12`)
- Virtualenv: `/home/gary/venvs/online_rlhf_cpu` (outside the repository)
- Key packages (from `pip list`):
  - `torch 2.5.1+cpu` (installed from <https://download.pytorch.org/whl/cpu>)
  - `numpy 2.5.3`
  - `pytest 9.1.1`
  - `tqdm 4.70.0`
- Lint: separate venv with `ruff 0.16.7` (pinned in `requirements/lint.txt`)

## Installation steps

```bash
python3.12 -m venv /path/to/venv
/path/to/venv/bin/pip install --upgrade pip
/path/to/venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
/path/to/venv/bin/pip install -r requirements/cpu-test.txt
```

The repository root `requirements.txt` (training stack, e.g.
`transformers==4.46.3`) is intentionally NOT installed. transformers,
DeepSpeed, flash-attn, Ray and vLLM are not needed for these tests: the
adapter in `tests/support/legacy_imports.py` loads the production modules by
file path behind controlled package shells, and a fail-fast `deepspeed`
sentinel guarantees no test can silently route around the missing stack.

## Historical local verification (round 3)

Final full verification — round 3 (snapshot `14bbff0de58b765a7e4303a85af0fd1fc032b89d`, branch `fix/cg-fletcher-reeves-beta`,
based on the round-2 HEAD `202d4cd`; raw log with the real pytest exit code
captured via `tee`):

```bash
python -m pip check          # No broken requirements found.
python -m pytest tests/cpu -q -ra --strict-markers
# 90 passed, 0 xfailed in 0.43s (pytest); process real 2.394s — well under the 60s budget
ruff check tests/            # All checks passed!
```

Raw log of the final run (verbatim stdout + real pytest exit code 0):
`/home/gary/pytest-round3-final-20260912-165355.log`.

Round-3 production-diff scope check:
`git diff chore/maintenance-round1 -- openrlhf/` contains ONLY the
authorized changes (CG beta one-line fix + solver docstrings in three
trainers, `--cg_mixing_weight` in two CLIs, aliases in rm_score_selection.py);
`pipeline/`, `merge_peft.py`, root `requirements.txt`: no diff.

Historical note: "76 passed, 3 xfailed" was the round-2 suite (3 strict
xfails documenting CG-1); "48 passed, 2 xfailed" was round-1b; "~1.5s /
40 passed, 2 xfailed" was round 1. All are kept here only as history; the
round-4 suite is recorded below.

The CG-1 defect is fixed on this branch; the previously-xfailed tests now
pass and `test_cg_converges_on_conjugate_direction_system` guards the fix
against silent reverts. Open decision items (CG-2 semantics, D-3a, D-3c,
D-4) are unchanged and listed in `review.md`.

## Not verified in this environment

- GPU training, multi-GPU runs, DeepSpeed ZeRO stages, FlashAttention.
- The README training commands (they require the full training stack).
- The revised round-4 GitHub Actions workflow: local checks do not prove
  its remote execution. A verified historical run is recorded below.

## Round-4 local evidence (2026-09-18)

Repository /home/gary/projects/Online_RLHF; branch fix/cg-fletcher-reeves-beta;
starting and ending HEAD 14bbff0de58b765a7e4303a85af0fd1fc032b89d. Starting
staged/unstaged/untracked status was empty. Results below include the current
uncommitted changes, not a new commit.

Actual interpreters/tools:

- CPU interpreter: /home/gary/venvs/online_rlhf_cpu/bin/python
- CPython 3.12.3, GCC 13.3.0; torch 2.5.1+cpu; numpy 2.5.3;
  pytest 9.1.1; tqdm 4.70.0.
- Ruff: /home/gary/venvs/online_rlhf_lint/bin/ruff, version 0.16.7.
- No environment was created, packages installed or hooks installed this round.
  No checks ran in conda base.
- requirements/cpu-test.txt and requirements/lint.txt pin direct CPU dependencies
  and maintenance tools only. The complete training stack and transitive
  dependencies are not locked; full training reproducibility is not claimed.

From the repository root, actual baseline/final commands and outputs:

| Command | Baseline (exit) | Final (exit) |
| --- | --- | --- |
| /home/gary/venvs/online_rlhf_cpu/bin/python -m pytest tests/cpu -q -ra --strict-markers | 90 passed in 0.58s (0) | 113 passed in 0.71s (0) |
| /home/gary/venvs/online_rlhf_lint/bin/ruff check tests/ | All checks passed! (0) | All checks passed! (0) |
| /home/gary/venvs/online_rlhf_lint/bin/ruff format --check tests/ | test_cg.py needs formatting; 1 would change, 10 already formatted (1) | 13 files already formatted (0) |
| /home/gary/venvs/online_rlhf_cpu/bin/python -m pip check | No broken requirements found. (0) | No broken requirements found. (0) |
| /home/gary/venvs/online_rlhf_lint/bin/python -m pip check | not run at baseline | No broken requirements found. (0) |
| git diff --check | not run at baseline | no output (0) |

No failed tests, skips or xfails in the full final CPU run. Original 90 node IDs
were compared against a git archive HEAD snapshot: all retained, 23 added,
113 total. Existing test_cg.py assertion ASTs were retained. The counts refer to
parameterized test instances, not independent behaviors or project-wide coverage.
No coverage tool or full training integration was run.

Version commands (exit 0):

```bash
/home/gary/venvs/online_rlhf_cpu/bin/python -c 'import sys, torch, numpy, pytest, tqdm; print(sys.executable); print(sys.version); print({m.__name__: m.__version__ for m in (torch, numpy, pytest, tqdm)})'
/home/gary/venvs/online_rlhf_lint/bin/ruff --version
```

Targeted parameter/numerical command (exit 0, 50 passed in 0.39s):

```bash
/home/gary/venvs/online_rlhf_cpu/bin/python -m pytest tests/cpu/test_cli_args.py tests/cpu/test_cg.py -q -ra --strict-markers --tb=short
```

This consists of 20 real CLI parameter-block instances and 30 real trainer CG
instances. The first development run of the new CLI adapter failed because
datetime.strftime required the real builtin importer; the raw output was saved.
Supplying that standard importer fixed the test harness without altering
production behavior or reducing assertions.

Reverse verification on external copies: removing the active entry's argument
caused 4 failures; replacing None fallback with or caused the zero case to fail
(1 failed, 3 passed). Both pytest exits were 1 as intended; detailed scope and
commands are in review.md. They are not failures in the final working tree.

Local raw evidence is retained outside the repository at
/tmp/online-rlhf-round4-jhlg7_n5 (temporary, not an archival dependency):

- baseline-0 through baseline-5.log: commands, outputs, real exit codes.
- final-cpu/lint/format/pip/lint-pip/collection/scope/diff.log.
- missing-flag.log, or-fallback.log and external mutated copies.
- cg3.py, cg3.log, pre-beta-snapshot.log; actual numbers and portable reproduction
  are retained in review.md.
- original production sources, start-hashes.json, start-status.txt,
  start-head.txt, audit_scope.py and collection-comparison.txt.
- remote-runs.json and remote-jobs.json from read-only GitHub public API queries.

Scope audit command (exit 0):

```bash
/home/gary/venvs/online_rlhf_cpu/bin/python /tmp/online-rlhf-round4-jhlg7_n5/audit_scope.py
```

It compares trainer ASTs excluding docstrings against the starting sources;
compares CLI ASTs excluding only the mixing definition/fallback; checks protected
file hashes and retained CG assertions. All three trainer executable ASTs are
unchanged. Existing two CLI files are byte-unchanged; only three missing entries
gain executable code. No pipeline, merge_peft.py, root requirements.txt, LICENSE
or dependency-list changes occurred.

## Verified historical remote CI, separate from round 4

Read-only GitHub API queries found
[cpu-checks run 34686855913](https://github.com/Gary-CF/Online_RLHF/actions/runs/34686855913)
for SHA 14bbff0de58b765a7e4303a85af0fd1fc032b89d, completed/success.
UTC created/run_started_at: 2026-09-12 09:49:44; updated_at: 09:50:28
(a 44-second span, consistent with the user report).
Lint job: 09:49:47–09:49:52, success.
CPU pytest job: 09:49:48–09:50:28, success.

That historical workflow did not include the new format check. This is evidence
for the old commit only, not the round-4 working-tree changes. Round-4 remote CI
is **pending the user's commit/push and a matching new-SHA run**; no push is
required or performed by this maintenance session. CPU checks do not verify
GPU/distributed training, training performance or full-stack reproducibility.
