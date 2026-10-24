"""Small-scale reproduction of the paper's training experiments.

    python validation/training.py grokking       # a + b mod 31: Slingshot spikes and the margin equilibrium
    python validation/training.py distillation   # a + b mod 53 teacher, student initialised at the teacher

Defaults follow the paper's settings where it states them. The model is a one-layer width-64 ReLU
transformer; the paper does not give every architectural detail, so this is not an exact replica.
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

from barron_softmax_fix import barron_cross_entropy, barron_soft_target_cross_entropy

FP32_CLIFF = math.log(2**24)


class Transformer(nn.Module):
    """One-layer ReLU transformer reading ``[a, b, =]`` and predicting from the last position."""

    def __init__(self, modulus: int, width: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(modulus + 1, width)
        self.position = nn.Parameter(torch.randn(3, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, bias=False)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.ReLU(), nn.Linear(4 * width, width))
        self.unembed = nn.Linear(width, modulus, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(tokens) + self.position
        hidden = hidden + self.attention(hidden, hidden, hidden, need_weights=False)[0]
        hidden = hidden + self.mlp(hidden)
        return self.unembed(hidden[:, -1])


def dataset(modulus: int, seed: int, device: str):
    a, b = torch.meshgrid(torch.arange(modulus), torch.arange(modulus), indexing="ij")
    tokens = torch.stack([a.flatten(), b.flatten(), torch.full((modulus * modulus,), modulus)], dim=-1)
    labels = (a.flatten() + b.flatten()) % modulus
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
    split = len(labels) // 2
    train, test = order[:split], order[split:]
    return tokens[train].to(device), labels[train].to(device), tokens[test].to(device), labels[test].to(device)


def margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.double()
    target = logits.gather(-1, labels[:, None]).squeeze(-1)
    others = logits.scatter(-1, labels[:, None], float("-inf"))
    return target - torch.logsumexp(others, dim=-1)


def grokking(args) -> None:
    print(f"== E11a grokking a+b mod {args.modulus}, {args.steps} full-batch AdamW steps ==")
    for eps in args.eps:
        for variant in args.variants:
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                dtype = torch.float64 if variant == "torch_fp64" else torch.float32
                model = Transformer(args.modulus).to(args.device, dtype)
                train_x, train_y, test_x, test_y = dataset(args.modulus, seed, args.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0, eps=eps)
                loss_fn = barron_cross_entropy if variant == "barron_fp32" else (lambda z, y: F.cross_entropy(z, y, reduction="none"))
                losses = []
                first_full_train = None
                spikes = 0
                start = time.perf_counter()
                for step in range(args.steps):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(train_x)
                    loss = loss_fn(logits, train_y).mean()
                    loss.backward()
                    optimizer.step()
                    value = loss.item()
                    losses.append(value)
                    if first_full_train is None and step % 10 == 0:
                        if (logits.argmax(-1) == train_y).all():
                            first_full_train = step
                    if step > 200 and value > 1e-3 and value > 100 * min(losses[-200:]):
                        spikes += 1
                with torch.no_grad():
                    train_logits = model(train_x)
                    test_logits = model(test_x)
                    test_acc = (test_logits.argmax(-1) == test_y).float().mean().item()
                    train_margin = margins(train_logits, train_y)
                tail = torch.tensor(losses[len(losses) // 2 :])
                print(
                    f"eps={eps:.0e} {variant:11s} seed={seed}: test acc {test_acc:.4f}; train 100% at step {first_full_train}; "
                    f"loss spikes (>100x recent min) {spikes}; second-half loss min {tail.min().item():.2e} max {tail.max().item():.2e}; "
                    f"final train margin median {train_margin.median().item():.2f} min {train_margin.min().item():.2f} "
                    f"frac past fp32 cliff {(train_margin > FP32_CLIFF).float().mean().item():.2f}; log(1/eps)={math.log(1 / eps):.1f}; "
                    f"{time.perf_counter() - start:.0f}s"
                )
                if args.trace:
                    trace = ",".join(f"{v:.3e}" for v in losses[:: max(1, args.steps // 400)])
                    print(f"TRACE eps={eps:.0e} {variant} seed={seed} loss={trace}")


def bf16_device(device: str) -> str:
    if device != "cuda":
        return "cpu"
    try:
        a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        (a @ a).sum().backward()
        torch.cuda.synchronize()
        return "cuda"
    except RuntimeError as error:
        print("bf16 not usable on cuda, falling back to cpu:", str(error)[:200])
        return "cpu"


def distillation(args) -> None:
    print(f"== E11b distillation a+b mod {args.distill_modulus} ==")
    torch.manual_seed(0)
    train_x, train_y, test_x, test_y = dataset(args.distill_modulus, 0, args.device)
    teacher = Transformer(args.distill_modulus).to(args.device)
    if args.teacher_path and os.path.exists(args.teacher_path):
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=args.device))
        print(f"teacher loaded from {args.teacher_path}")
    else:
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)
        for step in range(args.teacher_steps):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(teacher(train_x), train_y).backward()
            optimizer.step()
        if args.teacher_path:
            torch.save(teacher.state_dict(), args.teacher_path)
    with torch.no_grad():
        teacher_test = (teacher(test_x).argmax(-1) == test_y).float().mean().item()
    print(f"teacher after {args.teacher_steps} steps: test acc {teacher_test:.4f}")
    inputs = torch.cat([train_x, test_x])
    labels = torch.cat([train_y, test_y])
    for dtype_name, device in (("fp32", args.device), ("bf16", bf16_device(args.device))):
        dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
        frozen = Transformer(args.distill_modulus)
        frozen.load_state_dict(teacher.state_dict())
        frozen = frozen.to(device, dtype).eval()
        x = inputs.to(device)
        with torch.no_grad():
            teacher_logits = frozen(x) / args.temperature
        for variant, loss_fn in (
            ("torch", lambda s, t: F.cross_entropy(s, F.softmax(t, dim=-1), reduction="none")),
            ("barron", barron_soft_target_cross_entropy),
        ):
            student = Transformer(args.distill_modulus)
            student.load_state_dict(teacher.state_dict())
            student = student.to(device, dtype)
            reference = [p.detach().clone() for p in student.parameters()]
            optimizer = torch.optim.Adam(student.parameters(), lr=2e-3, eps=1e-8)
            start = time.perf_counter()
            for step in range(args.distill_steps + 1):
                if step in (0, 100, 500, 1000, args.distill_steps):
                    with torch.no_grad():
                        student_logits = student(x)
                        p = torch.log_softmax(teacher_logits.double(), -1)
                        q = torch.log_softmax(student_logits.double() / args.temperature, -1)
                        kl = (p.exp() * (p - q)).sum(-1).mean().item()
                        accuracy = (student_logits.argmax(-1) == labels.to(device)).float().mean().item()
                        agree = (student_logits.argmax(-1) == teacher_logits.argmax(-1)).float().mean().item()
                        drift = max((a.double() - b.double()).abs().max().item() for a, b in zip(student.parameters(), reference))
                    print(
                        f"{dtype_name} {variant:6s} step {step:5d}: KL(teacher||student) {kl:.2e} acc {accuracy:.4f} "
                        f"agree {agree:.4f} max |param drift| {drift:.2e} ({time.perf_counter() - start:.0f}s)"
                    )
                if step == args.distill_steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss_fn(student(x) / args.temperature, teacher_logits).mean().backward()
                optimizer.step()
                if step == 9:
                    per_step = (time.perf_counter() - start) / 10
                    print(f"{dtype_name} {variant} on {device}: {per_step:.3f} s/step, ETA {per_step * args.distill_steps / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["grokking", "distillation"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modulus", type=int, default=31)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-8, 1e-12])
    parser.add_argument("--variants", nargs="+", default=["torch_fp32", "barron_fp32", "torch_fp64"])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--distill-modulus", type=int, default=53)
    parser.add_argument("--teacher-steps", type=int, default=20000)
    parser.add_argument("--distill-steps", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--teacher-path", default="")
    arguments = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    print("torch", torch.__version__, "device", arguments.device)
    {"grokking": grokking, "distillation": distillation}[arguments.experiment](arguments)
