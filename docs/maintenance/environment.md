# CPU test environment (round 1)

This document records the environment actually used to verify the CPU tests in
`tests/cpu/`. It is not the original authors' training environment.

## Verified environment

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

## Commands and results (local run)

Final full verification (round 2: full-model trainer coverage, conversion
pipeline, score selection, pre-commit config; raw log with the real pytest
exit code captured via `tee`):

```bash
python -m pip check          # No broken requirements found.
python -m pytest tests/cpu -q -ra --strict-markers
# 76 passed, 3 xfailed in 0.27s (pytest); process real 1.366s — well under the 60s budget
ruff check tests/            # All checks passed!
pre-commit run --all-files   # ruff + ruff-format on tests/: Passed (offline local hooks)
```

Raw log of the final run (verbatim stdout + real pytest exit code 0):
`/home/gary/pytest-round2-final-20260912-124030.log`.

Historical note: "48 passed, 2 xfailed" was the round-1b suite; "~1.5s /
40 passed, 2 xfailed" was the original round-1 suite. Both are kept here
only as history; the current suite is the one above.

The 3 xfails are strict and all document the same confirmed legacy defect
(CG-1 in `review.md`), one instance per HVP trainer; they must fail for the
expected reason. An XPASS means the defect was fixed and the test should be
updated.

## Not verified in this environment

- GPU training, multi-GPU runs, DeepSpeed ZeRO stages, FlashAttention.
- The README training commands (they require the full training stack).
- GitHub Actions: the workflow `.github/workflows/cpu-checks.yml` was
  generated and mirrors the local commands, but no remote run exists yet.
