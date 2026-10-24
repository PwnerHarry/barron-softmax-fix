"""Softmax, log-softmax and cross-entropy from "A Fix for Softmax and Cross-Entropy".

Port of the JAX listings in Barron, Gnanapragasam and Tschannen, arXiv:2610.31642, as published on
https://jonbarron.info/softmaxfix/. This is not Barron's robust loss.

The standard implementations have two numerical bugs:

* Softmax Collapse. Once the margin of the largest logit over the others passes a precision cliff
  (about 5.5 in bfloat16, 16.6 in float32 and 36.7 in float64), ``1 - p_t`` rounds to 0. The target
  gradient is then exactly 0 while the other gradients are not, so the gradient no longer sums to 0.
* Softmax Cancellation. The soft-target gradient ``q - p`` is a difference of two nearly equal
  vectors, so it does not reach 0 when the student logits equal the teacher logits.

These functions compute every quantity from the margin ``m = z_pivot - logsumexp_{j != pivot} z_j``
with the pivot entry held at the constant 0, so ``log p_pivot = logsigmoid(m)`` is never formed by a
subtraction from 1. All functions reduce over the last dimension and return unreduced values.

Differences from the JAX listings, which do not change any finite result:

* ``softplus(-m)`` is written as ``-logsigmoid(m)``, whose backward stays accurate under TorchScript,
  ``torch.compile`` and float16.
* Masked (-inf) logits give finite gradients where the listings give NaN. A row with one unmasked class
  and a masked cross-entropy target give the results of the torch built-ins, and the soft-target loss
  ignores classes that both logits mask.

Limits:

* Pass float32 logits. The functions do not cast, so in bfloat16 and float16 every intermediate is
  rounded to the input dtype. Away from the cliff that is noisier than the torch kernels, which
  accumulate in float32. In float16 the gradient underflows at a margin of about 17.
* They are several times slower than the fused torch kernels and keep more intermediates for backward.
* The log-softmax gradient at the argmax is minus the sum over the other classes. With an arbitrary
  upstream gradient over many classes it loses relative precision. The cross-entropy upstreams do not.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

__version__ = "0.1.0"

__all__ = [
    "barron_cross_entropy",
    "barron_log_softmax",
    "barron_soft_target_cross_entropy",
    "barron_softmax",
]


def _logsumexp(values: Tensor, keepdim: bool) -> Tensor:
    """``torch.logsumexp`` over the last dimension, with a zero gradient for rows that are all -inf."""
    all_masked = torch.isneginf(values).all(dim=-1, keepdim=True)
    result = torch.logsumexp(values.masked_fill(all_masked, 0.0), dim=-1, keepdim=True)
    result = result.masked_fill(all_masked, float("-inf"))
    return result if keepdim else result.squeeze(-1)


def barron_log_softmax(logits: Tensor) -> Tensor:
    """Log-softmax over the last dimension, pivoted on the largest logit.

    Args:
        logits: ``[..., C]`` logits.

    Returns:
        ``[..., C]`` log-probabilities.
    """
    max_index = logits.argmax(dim=-1, keepdim=True)
    max_logit = logits.gather(-1, max_index)
    shifted = (logits - max_logit).scatter(-1, max_index, 0.0)
    others = shifted.scatter(-1, max_index, float("-inf"))
    margin = -_logsumexp(others, keepdim=True)
    return shifted + F.logsigmoid(margin)


def barron_softmax(logits: Tensor) -> Tensor:
    """Softmax over the last dimension, as the exponential of :func:`barron_log_softmax`.

    Args:
        logits: ``[..., C]`` logits.

    Returns:
        ``[..., C]`` probabilities.
    """
    return barron_log_softmax(logits).exp()


def barron_cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy against integer class labels, pivoted on the target logit.

    Equal to ``F.cross_entropy(logits, target, reduction="none")`` in exact arithmetic.

    Args:
        logits: ``[..., C]`` logits.
        target: ``[...]`` int64 class indices.

    Returns:
        ``[...]`` per-element loss in nats.

    Raises:
        ValueError: If ``target`` does not have the shape of ``logits`` without the class dimension.
    """
    if target.shape != logits.shape[:-1]:
        raise ValueError("target must have the shape of logits without the class dimension")
    target_index = target.unsqueeze(-1)
    target_logit = logits.gather(-1, target_index)
    target_is_masked = torch.isneginf(target_logit)
    others = (logits - target_logit.masked_fill(target_is_masked, 0.0)).scatter(-1, target_index, float("-inf"))
    margin = target_logit.masked_fill(~target_is_masked, 0.0).squeeze(-1) - _logsumexp(others, keepdim=False)
    return -F.logsigmoid(margin)


def barron_soft_target_cross_entropy(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    """Cross-entropy of the student distribution against the teacher distribution.

    The forward value is ``-sum(p * log q)`` with ``p = softmax(teacher_logits)`` and
    ``q = softmax(student_logits)``. The gradient with respect to the student logits is replaced by
    ``q - p`` computed from the logit difference, so it is exactly 0 when the two logits are equal.
    The teacher logits receive the gradient of the forward value; detach them for a frozen teacher.
    Apply any distillation temperature to both logits before the call. Classes that both logits mask
    with -inf do not contribute.

    The replaced gradient has no second derivative with respect to the student logits. Under
    ``torch.compile`` on CPU with torch 2.11, inductor computes ``expm1(x)`` as ``exp(x) - 1``, which
    brings the cancellation back; torch 2.14 computes it accurately, and eager mode and TorchScript keep
    the fix in both.

    Args:
        student_logits: ``[..., C]`` student logits.
        teacher_logits: ``[..., C]`` teacher logits.

    Returns:
        ``[...]`` per-element loss in nats.

    Raises:
        ValueError: If the two logits do not have the same shape.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape")
    p = barron_softmax(teacher_logits)
    student_is_masked = torch.isneginf(student_logits)
    # The paper's stop-gradient terms.
    with torch.no_grad():
        log_q = barron_log_softmax(student_logits)
        q = log_q.exp()
        delta = (student_logits - teacher_logits).masked_fill(student_is_masked & torch.isneginf(teacher_logits), 0.0)
        # q - p from the logit difference, which avoids subtracting two close probabilities. Where
        # |log(q / p)| >= 1 there is no such cancellation, and q - p avoids 0 * inf when expm1 overflows.
        centered = delta - (p * delta).sum(dim=-1, keepdim=True)
        log_ratio = centered - torch.log1p((p * torch.expm1(centered)).sum(dim=-1, keepdim=True))
        difference = torch.where(log_ratio.abs() < 1, p * torch.expm1(log_ratio), q - p)
        # Removes the rounding residue so that the gradient sums to 0.
        gradient = difference - q * difference.sum(dim=-1, keepdim=True)
        log_q = log_q.masked_fill(p == 0, 0.0)
    loss = -(p * log_q).sum(dim=-1)
    finite_student = student_logits.masked_fill(student_is_masked, 0.0)
    # Zero in value. Its gradient with respect to the student logits is ``gradient``.
    return loss + (gradient * (finite_student - finite_student.detach())).sum(dim=-1)
