"""Conjugate gradient characterization tests for the legacy HVP reward trainers.

Isolation notes:
- ``args.damping`` (the direction-mixing weight) is set to 0.0 so the solver
  output is the pure CG result; ``cg_damping > 0`` supplies the Tikhonov-style
  regularization. This is NOT the default CLI training configuration.
- ``residual_tol`` is compared against the SQUARED residual norm inside the
  production loop (``r_norm_sq < residual_tol``), not the residual norm.
  This is interface semantics (the name can mislead), pinned by
  ``test_cg_residual_tol_compares_squared_residual``.

Known defect under test (CG-1, see docs/maintenance/review.md): the
Fletcher-Reeves beta — the ratio of the NEW to the OLD squared residual —
is computed as ``r_norm_sq / r_norm_sq`` where both operands are the freshly
updated value. Whenever that denominator is nonzero this yields beta = 1.0
(exact zero residual would be 0/0); the search direction update becomes
``p = r + p`` instead of the standard CG recurrence. In exact arithmetic,
standard linear CG on an SPD system converges to the solution within at most
``dim`` iterations; with this beta it does not, which the strict xfail below
exposes on the small float64 fixture (numerical tolerances apply).
"""

import types

import pytest
import torch

from tests.support import logistic
from tests.support.legacy_imports import make_trainer_shell, trainer_hvp_gate


def _solve(
    legacy_trainer, max_iter, damping_mix=0.0, residual_tol=1e-10, damping=0.2, seed=7
):
    Z, theta, rejected = logistic.make_logistic_problem(seed=seed)
    trainer = make_trainer_shell(
        legacy_trainer.trainer,
        cg_damping=damping,
        args=types.SimpleNamespace(damping=damping_mix),
    )
    loss, flat_grad = logistic.pairwise_loss_and_flat_grad(
        legacy_trainer.loss.PairWiseLoss, Z, theta, rejected
    )
    with trainer_hvp_gate(legacy_trainer.kind):
        result = trainer.conjugate_gradient_solver(
            [theta], loss, flat_grad, max_iter=max_iter, residual_tol=residual_tol
        )
    return (
        result,
        flat_grad,
        Z,
        theta,
        rejected,
        damping,
    )


def _relative_residual(operator, x, rhs):
    return (torch.linalg.norm(operator @ x - rhs) / torch.linalg.norm(rhs)).item()


def test_cg_output_contract_at_dimension_budget(legacy_trainer):
    """Shape/dtype/device/finiteness for the same max_iter=4 budget the strict
    xfail uses, so the xfail only ever carries numerical-solver precision."""
    x, flat_grad, *_ = _solve(legacy_trainer, max_iter=4)
    assert x.shape == flat_grad.shape
    assert x.dtype == torch.float64
    assert x.device.type == "cpu"
    assert torch.isfinite(x).all()


# Known defect CG-1 (see docs/maintenance/review.md): beta == 1.0 whenever the
# residual-squared denominator is nonzero prevents convergence on SPD systems.
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="known defect CG-1 still present: beta == r_norm_sq/r_norm_sq == 1.0 "
    "prevents convergence on SPD systems; see docs/maintenance/review.md",
)
def test_cg_solves_spd_system_within_dimension_steps(legacy_trainer):
    """Standard linear CG on this SPD system would reach rel. residual <= 1e-5
    within dim=4 steps in exact arithmetic (float64 tolerances apply)."""
    x, flat_grad, Z, theta, rejected, damping = _solve(legacy_trainer, max_iter=4)
    operator = logistic.regularized_hessian(Z, theta, rejected, damping)
    reference = torch.linalg.solve(operator, flat_grad)

    assert torch.allclose(x, reference, rtol=1e-5, atol=1e-7)
    assert _relative_residual(operator, x, flat_grad) <= 1e-5


def test_cg_full_mix_returns_gradient_direction(legacy_trainer):
    """With the mixing weight args.damping = 1.0 the output is flat_grad itself."""
    x, flat_grad, *_ = _solve(legacy_trainer, max_iter=2, damping_mix=1.0)
    assert torch.equal(x, flat_grad)


def test_cg_residual_tol_compares_squared_residual(legacy_trainer):
    """Discriminating check: pick tol strictly between ||r1||^2 and ||r1|| for
    the independently derived one-step residual r1. Stopping after the first
    update (returning exactly the one-step iterate) is only possible if the
    loop compares the SQUARED residual against tol."""
    Z, theta, rejected = logistic.make_logistic_problem()
    damping = 0.2
    loss, flat_grad = logistic.pairwise_loss_and_flat_grad(
        legacy_trainer.loss.PairWiseLoss, Z, theta, rejected
    )
    operator = logistic.regularized_hessian(Z, theta, rejected, damping)
    g = flat_grad.detach()
    # Independent one-step iterate, keeping the production denominator's 1e-8.
    alpha = (g @ g) / (g @ operator @ g + 1e-8)
    r1 = g - alpha * operator @ g
    squared = (r1 @ r1).item()
    norm = torch.linalg.norm(r1).item()
    # Reference values for this fixture (independent computation):
    #   ||r1|| ~= 0.09713945660306085, ||r1||^2 ~= 0.009436074029137943
    assert squared == pytest.approx(0.009436074029137943, rel=1e-12)
    assert norm == pytest.approx(0.09713945660306085, rel=1e-12)
    tol = (squared + norm) / 2
    assert squared < tol < norm

    trainer = make_trainer_shell(
        legacy_trainer.trainer,
        cg_damping=damping,
        args=types.SimpleNamespace(damping=0.0),
    )
    x = None
    with trainer_hvp_gate(legacy_trainer.kind):
        x = trainer.conjugate_gradient_solver(
            [theta], loss, flat_grad, max_iter=10, residual_tol=tol
        )
    assert torch.allclose(x, alpha * g, rtol=1e-7, atol=1e-9)


def test_cg_truncated_default_steps_stay_finite(legacy_trainer):
    """The production default is num_cg_steps=3; a truncated run must still
    return a finite, correctly shaped direction (accuracy is NOT asserted)."""
    x, flat_grad, *_ = _solve(legacy_trainer, max_iter=3)
    assert x.shape == flat_grad.shape
    assert x.dtype == torch.float64
    assert torch.isfinite(x).all()


def test_cg_is_deterministic(legacy_trainer):
    x1, _, *_ = _solve(legacy_trainer, max_iter=4)
    x2, _, *_ = _solve(legacy_trainer, max_iter=4)
    assert torch.equal(x1, x2)
