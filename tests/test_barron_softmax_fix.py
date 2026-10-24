"""Tests for the softmax, log-softmax and cross-entropy fixes of arXiv:2610.31642.

The ``*_where_torch_*`` tests also assert the failure of the torch built-in on the same input. If a
torch release fixes that failure, the assertion breaks, and the port may no longer be needed.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from barron_softmax_fix import (
    barron_cross_entropy,
    barron_log_softmax,
    barron_soft_target_cross_entropy,
    barron_softmax,
)


def _torch_soft_target_cross_entropy(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return -(torch.softmax(teacher_logits, dim=-1) * torch.log_softmax(student_logits, dim=-1)).sum(dim=-1)


def _value_and_grad(function, logits: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    leaf = logits.clone().requires_grad_(True)
    value = function(leaf, *args)
    value.sum().backward()
    return value.detach(), leaf.grad


@pytest.mark.parametrize("shape", [(7,), (4, 7), (2, 3, 11)])
def test_matches_torch_in_float64(shape):
    torch.manual_seed(0)
    logits = torch.randn(shape, dtype=torch.float64) * 3
    teacher = torch.randn(shape, dtype=torch.float64) * 3
    upstream = torch.randn(shape, dtype=torch.float64)
    target = torch.randint(0, shape[-1], shape[:-1])
    cases = [
        (
            lambda z: barron_log_softmax(z) * upstream,
            lambda z: torch.log_softmax(z, dim=-1) * upstream,
        ),
        (
            lambda z: barron_softmax(z) * upstream,
            lambda z: torch.softmax(z, dim=-1) * upstream,
        ),
        (
            lambda z: barron_cross_entropy(z, target),
            lambda z: F.cross_entropy(z.movedim(-1, 1) if z.dim() > 1 else z, target, reduction="none"),
        ),
        (
            lambda z: barron_soft_target_cross_entropy(z, teacher),
            lambda z: _torch_soft_target_cross_entropy(z, teacher),
        ),
    ]

    for barron, reference in cases:
        value, grad = _value_and_grad(barron, logits)
        expected_value, expected_grad = _value_and_grad(reference, logits)
        torch.testing.assert_close(value, expected_value, rtol=0, atol=1e-12)
        torch.testing.assert_close(grad, expected_grad, rtol=0, atol=1e-12)


def test_gradients_pass_gradcheck():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    target = torch.randint(0, 5, (3,))

    assert torch.autograd.gradcheck(barron_log_softmax, (logits,))
    assert torch.autograd.gradcheck(barron_softmax, (logits,))
    assert torch.autograd.gradcheck(lambda z: barron_cross_entropy(z, target), (logits,))
    assert torch.autograd.gradcheck(barron_soft_target_cross_entropy, (logits, teacher))


@pytest.mark.parametrize(
    "logits, target",
    [
        ([3.0, 3.0, 0.0], 0),
        ([1.5], 0),
        ([2.0, float("-inf"), 0.0], 0),
        ([2.0, float("-inf"), float("-inf")], 0),
        ([float("-inf"), 1.0, 0.0], 0),
        ([2.0, float("-inf"), float("-inf")], 1),
    ],
    ids=[
        "tie-at-max",
        "single-class",
        "one-masked",
        "all-but-target-masked",
        "masked-target",
        "masked-target-one-left",
    ],
)
def test_ties_and_masked_logits_match_torch(logits, target):
    logits = torch.tensor([logits])
    target = torch.tensor([target])

    value, grad = _value_and_grad(barron_cross_entropy, logits, target)
    expected_value, expected_grad = _value_and_grad(
        lambda z, t: F.cross_entropy(z, t, reduction="none"), logits, target
    )
    torch.testing.assert_close(value, expected_value, rtol=0, atol=1e-6)
    torch.testing.assert_close(grad, expected_grad, rtol=0, atol=1e-6)

    value, grad = _value_and_grad(lambda z: barron_log_softmax(z)[..., target], logits)
    expected_value, expected_grad = _value_and_grad(lambda z: torch.log_softmax(z, dim=-1)[..., target], logits)
    torch.testing.assert_close(value, expected_value, rtol=0, atol=1e-6)
    torch.testing.assert_close(grad, expected_grad, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "function, inputs",
    [
        (barron_cross_entropy, (torch.randn(3, 4), torch.tensor([2]))),
        (barron_soft_target_cross_entropy, (torch.randn(3, 4), torch.randn(4))),
    ],
    ids=["cross_entropy", "soft_target_cross_entropy"],
)
def test_mismatched_shapes_raise_instead_of_broadcasting(function, inputs):
    with pytest.raises(ValueError, match="shape"):
        function(*inputs)


@pytest.mark.parametrize(
    "name, function, inputs",
    [
        (
            "log_softmax",
            lambda f, z: f(z)[:, 0],
            (torch.tensor([[9.0, -9.0, -9.0], [30.0, 1.0, -2.0]]),),
        ),
        (
            "softmax",
            lambda f, z: f(z)[:, 0],
            (torch.tensor([[9.0, -9.0, -9.0], [30.0, 1.0, -2.0]]),),
        ),
        (
            "cross_entropy",
            lambda f, z, t: f(z, t),
            (
                torch.tensor([[9.0, -9.0, -9.0], [30.0, 1.0, -2.0]]),
                torch.tensor([0, 0]),
            ),
        ),
        (
            "soft_target_cross_entropy",
            lambda f, z, t: f(z, t),
            (
                torch.tensor([[2.0 - 1e-7, 1.0 + 1e-7, 0.0]]),
                torch.tensor([[2.0, 1.0, 0.0]]),
            ),
        ),
    ],
)
def test_torchscript_matches_eager_after_warm_up(name, function, inputs):
    """The profiling executor differentiates scripted graphs with its own formulas after warm-up runs."""
    eager = {
        "log_softmax": barron_log_softmax,
        "softmax": barron_softmax,
        "cross_entropy": barron_cross_entropy,
        "soft_target_cross_entropy": barron_soft_target_cross_entropy,
    }[name]
    scripted = torch.jit.script(eager)

    expected_value, expected_grad = _value_and_grad(lambda *a: function(eager, *a), *inputs)
    for _ in range(4):
        value, grad = _value_and_grad(lambda *a: function(scripted, *a), *inputs)

    assert torch.equal(value, expected_value)
    assert torch.equal(grad, expected_grad)
    assert bool(grad.abs().max() > 0)


def test_paper_example_where_torch_collapses():
    """Section 2 of the paper: ``z = [9, -9, -9]``, ``t = 0`` in float32."""
    logits = torch.tensor([[9.0, -9.0, -9.0]])
    target = torch.tensor([0])
    tail = 2 * math.exp(-18.0)
    expected_loss = math.log1p(tail)
    expected_ce_grad = torch.tensor([[-tail, tail / 2, tail / 2]]) / (1 + tail)

    loss, grad = _value_and_grad(barron_cross_entropy, logits, target)
    torch.testing.assert_close(loss, torch.tensor([expected_loss]), rtol=1e-6, atol=0)
    torch.testing.assert_close(grad, expected_ce_grad, rtol=1e-6, atol=0)
    torch.testing.assert_close(
        barron_log_softmax(logits)[0, 0],
        torch.tensor(-expected_loss),
        rtol=1e-6,
        atol=0,
    )
    _, softmax_grad = _value_and_grad(lambda z: barron_softmax(z)[:, 0], logits)
    torch.testing.assert_close(softmax_grad, -expected_ce_grad, rtol=1e-6, atol=0)

    torch_loss, torch_grad = _value_and_grad(lambda z, t: F.cross_entropy(z, t, reduction="none"), logits, target)
    assert torch_loss.item() == 0.0
    assert torch_grad[0, 0].item() == 0.0
    assert torch.log_softmax(logits, dim=-1)[0, 0].item() == 0.0


@pytest.mark.parametrize(
    "dtype, margin",
    [
        (torch.float16, 9.0),
        (torch.bfloat16, 8.0),
        (torch.float32, 20.0),
        (torch.float64, 40.0),
    ],
    ids=["float16", "bfloat16", "float32", "float64"],
)
def test_cross_entropy_gradient_past_the_precision_cliff_where_torch_is_zero(dtype, margin):
    """Past ``m = log(2^p)`` for a ``p``-bit significand, torch rounds the target gradient to 0."""
    logits = torch.tensor([[margin, 0.0, 0.0]], dtype=dtype)
    target = torch.tensor([0])
    tail = 2 * math.exp(-margin)
    exact_target_grad = -tail / (1 + tail)
    rtol = {
        torch.float16: 2e-3,
        torch.bfloat16: 1e-2,
        torch.float32: 1e-6,
        torch.float64: 1e-14,
    }[dtype]

    _, grad = _value_and_grad(barron_cross_entropy, logits, target)
    torch.testing.assert_close(
        grad[0, 0].double(),
        torch.tensor(exact_target_grad, dtype=torch.float64),
        rtol=rtol,
        atol=0,
    )
    assert grad.double().sum().abs().item() <= rtol * grad.double().abs().max().item()

    _, torch_grad = _value_and_grad(lambda z, t: F.cross_entropy(z, t, reduction="none"), logits, target)
    assert torch_grad[0, 0].item() == 0.0
    assert torch_grad.double().sum().item() > 0.0


def test_paper_soft_target_example_where_torch_is_inaccurate():
    """Section 3 of the paper: a float32 perturbation of ``z_T = [2, 1, 0]`` by ``[-1e-7, 1e-7, 0]``."""
    teacher = torch.tensor([[2.0, 1.0, 0.0]])
    student = teacher + torch.tensor([[-1e-7, 1e-7, 0.0]])
    _, exact = _value_and_grad(_torch_soft_target_cross_entropy, student.double(), teacher.double())

    _, grad = _value_and_grad(barron_soft_target_cross_entropy, student, teacher)
    torch.testing.assert_close(grad.double(), exact, rtol=1e-5, atol=0)

    _, torch_grad = _value_and_grad(_torch_soft_target_cross_entropy, student, teacher)
    assert ((torch_grad.double() - exact).abs() / exact.abs()).max().item() > 0.1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_soft_target_gradient_is_zero_at_the_teacher_where_torch_is_not(dtype):
    torch.manual_seed(0)
    teacher = (torch.randn(16, 100) * 3).to(dtype)

    value, grad = _value_and_grad(barron_soft_target_cross_entropy, teacher, teacher)
    assert torch.equal(grad, torch.zeros_like(grad))
    torch.testing.assert_close(
        value.double(),
        _torch_soft_target_cross_entropy(teacher.double(), teacher.double()),
        rtol=1e-2,
        atol=0,
    )

    if dtype != torch.float64:
        _, torch_grad = _value_and_grad(_torch_soft_target_cross_entropy, teacher, teacher)
        assert bool(torch_grad.abs().max() > 0)


def test_soft_target_cross_entropy_is_finite_when_expm1_overflows():
    """A teacher probability of 0 times an overflowed ``expm1`` is ``0 * inf``; the paper falls back to ``q - p``."""
    teacher = torch.tensor([[0.0, -200.0, 0.0]])
    student = torch.tensor([[0.0, 100.0, 0.0]])
    expected_value, expected_grad = _value_and_grad(
        _torch_soft_target_cross_entropy, student.double(), teacher.double()
    )

    value, grad = _value_and_grad(barron_soft_target_cross_entropy, student, teacher)
    torch.testing.assert_close(value.double(), expected_value, rtol=1e-6, atol=0)
    torch.testing.assert_close(grad.double(), expected_grad, rtol=0, atol=1e-6)


def test_adamw_margin_settles_near_log_inverse_epsilon_where_torch_stops_short():
    """AdamW balances weight decay against ``g / (|g| + eps)``, which puts the margin near ``log(1 / eps)``.

    torch rounds the target gradient to 0 past the float32 cliff, so only the other logits move and the
    margin settles lower.
    """
    eps = 1e-8

    def final_margin(loss_function) -> float:
        logits = torch.nn.Parameter(torch.zeros(1, 31))
        target = torch.zeros(1, dtype=torch.long)
        optimizer = torch.optim.AdamW([logits], lr=1e-2, betas=(0.9, 0.98), eps=eps, weight_decay=0.01)
        for _ in range(5000):
            optimizer.zero_grad()
            loss_function(logits, target).sum().backward()
            optimizer.step()
        logits = logits.detach().double()[0]
        return (logits[0] - torch.logsumexp(logits[1:], dim=-1)).item()

    barron_margin = final_margin(barron_cross_entropy)
    torch_margin = final_margin(lambda z, t: F.cross_entropy(z, t, reduction="none"))

    assert abs(barron_margin - math.log(1 / eps)) < 2.0
    assert torch_margin < barron_margin - 2.0


def test_soft_target_cross_entropy_ignores_classes_both_logits_mask():
    torch.manual_seed(0)
    teacher = torch.randn(4, 7) * 3
    student = teacher + 1e-6 * torch.randn(4, 7)
    keep = torch.tensor([True, False, True, True, False, True, True])
    masked_teacher = teacher.masked_fill(~keep, float("-inf"))
    masked_student = student.masked_fill(~keep, float("-inf"))
    expected_value, expected_grad = _value_and_grad(
        barron_soft_target_cross_entropy, student[:, keep], teacher[:, keep]
    )

    value, grad = _value_and_grad(barron_soft_target_cross_entropy, masked_student, masked_teacher)
    torch.testing.assert_close(value, expected_value, rtol=1e-6, atol=0)
    torch.testing.assert_close(grad[:, keep], expected_grad, rtol=1e-5, atol=1e-12)
    assert torch.equal(grad[:, ~keep], torch.zeros_like(grad[:, ~keep]))

    _, grad_at_teacher = _value_and_grad(barron_soft_target_cross_entropy, masked_teacher, masked_teacher)
    assert torch.equal(grad_at_teacher, torch.zeros_like(grad_at_teacher))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_adam_distillation_stays_at_the_teacher_where_torch_drifts(dtype):
    """Adam rescales a rounding-noise gradient to a full step, so torch moves a student that is already exact."""
    torch.manual_seed(0)
    teacher = (torch.randn(32, 53) * 4).to(dtype)

    def distil(loss_function) -> torch.Tensor:
        student = torch.nn.Parameter(teacher.clone())
        optimizer = torch.optim.Adam([student], lr=2e-3, eps=1e-8)
        for _ in range(10):
            optimizer.zero_grad()
            loss_function(student, teacher).mean().backward()
            optimizer.step()
        return student.detach()

    assert torch.equal(distil(barron_soft_target_cross_entropy), teacher)
    assert bool((distil(_torch_soft_target_cross_entropy) - teacher).abs().max() > 1e-3)
