"""Shared fixtures for the legacy CPU tests.

Import boundary: ``tests/support`` is imported as a namespace package from the
repository root, so these tests never execute the production package
aggregators.
"""

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.support.legacy_imports import (  # noqa: E402
    isolated_legacy_imports,
    load_convert_to_dataset,
    load_legacy_loss,
    load_trainer,
)
from tests.support.random_state import preserve_python_random_state  # noqa: E402


@pytest.fixture(autouse=True)
def _preserve_python_random_state():
    """Every test in tests/cpu must leave the global random state untouched."""
    with preserve_python_random_state():
        yield


@pytest.fixture(
    params=["rm", "rm_active"],
    ids=["rm_trainer_head_hvp", "rm_active_trainer_head_hvp"],
)
def legacy_trainer(request):
    """Both legacy head-HVP trainer modules, loaded from this checkout."""
    with isolated_legacy_imports():
        loss_mod = load_legacy_loss()
        trainer_mod = load_trainer(request.param)
        yield types.SimpleNamespace(
            loss=loss_mod, trainer=trainer_mod, kind=request.param
        )


@pytest.fixture()
def legacy_convert_to_dataset():
    """The real openrlhf/utils/convert_to_dataset.py module."""
    with isolated_legacy_imports():
        yield load_convert_to_dataset()
