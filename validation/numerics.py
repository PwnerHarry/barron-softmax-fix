"""Numerical checks of the paper's claims, comparing the port with the torch built-ins.

Run ``python validation/numerics.py [experiment ...]``; with no arguments every experiment runs. Each
experiment runs on CPU and, when available, CUDA.

    e1     Precision-cliff sweep: margin at which the cross-entropy target gradient becomes exactly 0.
    e2     Zero-sum residual of the cross-entropy gradient against the margin.
    e3     Ordinary-regime parity of values and gradients against a float64 reference.
    e4     Soft-target gradient at and near the teacher, including the paper's Section 3 example.
    e5     Optimizer-level toys: Adam distillation drift and the AdamW margin equilibrium.
    e6     Edge cases: ties, one class, masked (-inf) logits, +inf, NaN, empty and non-contiguous input.
    e7     float16 autocast on CUDA.
    e8     TorchScript and torch.compile against eager on the cases where the fix matters.
    e9     Forward and backward time and peak memory on CUDA.
    e10    gradcheck, gradgradcheck and the soft-target Hessian.
    probe  inductor CPU precision of expm1 and log1p, and TorchScript gradients after warm-up.
"""

import math
import sys
import time

import torch
import torch.nn.functional as F

from barron_softmax_fix import (
    barron_cross_entropy,
    barron_log_softmax,
    barron_soft_target_cross_entropy,
    barron_softmax,
)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}
CUDA = torch.cuda.is_available()


def torch_soft_ce(student, teacher):
    return F.cross_entropy(student, F.softmax(teacher, dim=-1), reduction="none")


def ce_value_and_grad(fn, logits, target):
    z = logits.clone().requires_grad_(True)
    loss = fn(z, target)
    loss.sum().backward()
    return loss.detach(), z.grad.detach()


def torch_ce(z, t):
    return F.cross_entropy(z.reshape(-1, z.shape[-1]), t.reshape(-1), reduction="none").reshape(t.shape)


def e1(device):
    """Two classes z=[m, 0]: where does the target gradient become exactly 0?"""
    print(f"== E1 precision cliff sweep ({device}) ==")
    margins = torch.arange(0.0, 110.0, 0.01, dtype=torch.float64)
    for name, dtype in DTYPES.items():
        if dtype == torch.bfloat16 and device == "cuda":
            continue
        z = torch.stack([margins, torch.zeros_like(margins)], dim=-1).to(dtype).to(device)
        m = (z[:, 0].double() - z[:, 1].double()).cpu()
        t = torch.zeros(len(margins), dtype=torch.long, device=device)
        exact_loss = torch.log1p(torch.exp(-m))
        exact_target_grad = -torch.exp(-m) / (1 + torch.exp(-m))
        for label, fn in (("torch", torch_ce), ("barron", barron_cross_entropy)):
            loss, grad = ce_value_and_grad(fn, z, t)
            loss, grad = loss.double().cpu(), grad.double().cpu()
            zero_target = (grad[:, 0] == 0).nonzero()
            zero_loss = (loss == 0).nonzero()
            first_zero_grad = m[zero_target[0, 0]].item() if len(zero_target) else float("nan")
            first_zero_loss = m[zero_loss[0, 0]].item() if len(zero_loss) else float("nan")
            nonzero = grad[:, 0] != 0
            rel = ((grad[:, 0] - exact_target_grad).abs() / exact_target_grad.abs())[nonzero & (m < 40)]
            rel_loss = ((loss - exact_loss).abs() / exact_loss)[(loss != 0) & (m < 40)]
            zero_sum = (grad.sum(-1).abs() / grad.abs().amax(-1).clamp_min(1e-300))
            print(
                f"{name:5s} {label:6s} first m with target grad==0: {first_zero_grad:7.2f}  "
                f"first m with loss==0: {first_zero_loss:7.2f}  "
                f"max rel err target grad (where !=0, m<40): {rel.max().item():.2e}  "
                f"max rel err loss: {rel_loss.max().item():.2e}  "
                f"max |sum g|/max|g|: {zero_sum.max().item():.2e}"
            )
    print("predicted cliffs log(2^p): fp16", math.log(2**11), "bf16", math.log(2**8), "fp32", math.log(2**24), "fp64", math.log(2**53))
    # The paper's example.
    z = torch.tensor([[9.0, -9.0, -9.0]])
    t = torch.tensor([0])
    for label, fn in (("torch", torch_ce), ("barron", barron_cross_entropy)):
        loss, grad = ce_value_and_grad(fn, z, t)
        print(f"paper example {label}: loss {loss.tolist()} grad {grad.tolist()}")
    zl = z.clone().requires_grad_(True)
    print("torch log_softmax", torch.log_softmax(z, -1).tolist(), "barron", barron_log_softmax(z).tolist())
    for label, fn in (("torch", lambda x: torch.softmax(x, -1)), ("barron", barron_softmax)):
        zl = z.clone().requires_grad_(True)
        fn(zl)[0, 0].backward()
        print(f"grad softmax[0] {label}: {zl.grad.tolist()}")


def e2(device):
    print(f"== E2 zero-sum residual ({device}) ==")
    torch.manual_seed(0)
    for name in ("bf16", "fp32"):
        dtype = DTYPES[name]
        if dtype == torch.bfloat16 and device == "cuda":
            continue
        for classes in (3, 1000, 32000):
            for margin in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
                others = torch.randn(16, classes, dtype=torch.float64)
                others[:, 0] = 0.0
                others[:, 0] = torch.logsumexp(others[:, 1:], -1) + margin
                z = others.to(dtype).to(device)
                t = torch.zeros(16, dtype=torch.long, device=device)
                row = []
                for label, fn in (("torch", torch_ce), ("barron", barron_cross_entropy)):
                    _, grad = ce_value_and_grad(fn, z, t)
                    grad = grad.double()
                    residual = (grad.sum(-1).abs() / grad.abs().amax(-1)).max().item()
                    target_zero = (grad[:, 0] == 0).float().mean().item()
                    row.append(f"{label}: |sum|/max={residual:.2e} target==0 frac={target_zero:.2f}")
                print(f"{name} C={classes:6d} margin={margin:5.1f}  " + "  ".join(row))


def max_errors(value, reference):
    diff = (value.double() - reference).abs()
    scale = reference.abs().amax().clamp_min(1e-300)
    elementwise_rel = diff / reference.abs().clamp_min(1e-30)
    return diff.max().item() / scale.item(), elementwise_rel.max().item()


def e3(device):
    print(f"== E3 ordinary-regime parity ({device}); errors vs fp64 torch: (max abs/max|ref|, max elementwise rel) ==")
    torch.manual_seed(0)
    for name in ("bf16", "fp32", "fp64"):
        dtype = DTYPES[name]
        if dtype == torch.bfloat16 and device == "cuda":
            continue
        for sigma in (1.0, 5.0, 20.0):
            for classes in (10, 3721, 32000):
                z64 = torch.randn(64, classes, dtype=torch.float64) * sigma
                zt64 = torch.randn(64, classes, dtype=torch.float64) * sigma
                upstream = torch.randn(64, classes, dtype=torch.float64)
                t = torch.randint(0, classes, (64,))
                z = z64.to(dtype)
                zt = zt64.to(dtype)
                z_ref = z.double()
                zt_ref = zt.double()
                results = []
                cases = (
                    ("log_softmax", lambda x: torch.log_softmax(x, -1), barron_log_softmax, True),
                    ("softmax", lambda x: torch.softmax(x, -1), barron_softmax, True),
                    ("ce", lambda x: torch_ce(x, t.to(x.device)), lambda x: barron_cross_entropy(x, t.to(x.device)), False),
                    ("soft_ce", lambda x: torch_soft_ce(x, zt.to(x.device)), lambda x: barron_soft_target_cross_entropy(x, zt.to(x.device)), False),
                )
                for case, torch_fn, barron_fn, use_upstream in cases:
                    ref_in = z_ref.clone().requires_grad_(True)
                    if case == "soft_ce":
                        ref_out = torch_soft_ce(ref_in, zt_ref)
                    elif case == "ce":
                        ref_out = torch_ce(ref_in, t)
                    else:
                        ref_out = torch_fn(ref_in)
                    (ref_out * upstream).sum().backward() if use_upstream else ref_out.sum().backward()
                    entries = []
                    for fn in (torch_fn, barron_fn):
                        x = z.to(device).clone().requires_grad_(True)
                        out = fn(x)
                        up = upstream.to(dtype).to(device)
                        (out * up).sum().backward() if use_upstream else out.sum().backward()
                        v = max_errors(out.detach().cpu(), ref_out.detach())
                        g = max_errors(x.grad.cpu(), ref_in.grad)
                        entries.append(f"val {v[0]:.1e}/{v[1]:.1e} grad {g[0]:.1e}/{g[1]:.1e}")
                    results.append(f"{case}: torch[{entries[0]}] barron[{entries[1]}]")
                print(f"{name} sigma={sigma:4.1f} C={classes:5d}\n    " + "\n    ".join(results))


def e4(device):
    print(f"== E4 soft target ({device}) ==")
    torch.manual_seed(0)
    for name in ("bf16", "fp32", "fp64"):
        dtype = DTYPES[name]
        if dtype == torch.bfloat16 and device == "cuda":
            continue
        for classes in (3, 100, 3721):
            teacher = (torch.randn(64, classes, dtype=torch.float64) * 3).to(dtype).to(device)
            row = []
            for label, fn in (("torch", torch_soft_ce), ("barron", barron_soft_target_cross_entropy)):
                x = teacher.clone().requires_grad_(True)
                fn(x, teacher).sum().backward()
                row.append(f"{label} max|grad| at z_S=z_T: {x.grad.abs().max().item():.2e}")
            print(f"{name} C={classes:5d}  " + "  ".join(row))
    z_t = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_s = z_t + torch.tensor([-1e-7, 1e-7, 0.0], dtype=torch.float32, device=device)
    for label, fn in (("torch", torch_soft_ce), ("barron", barron_soft_target_cross_entropy)):
        x = z_s.clone().requires_grad_(True)
        fn(x, z_t).sum().backward()
        print(f"paper sec3 {label}: {x.grad.tolist()}")
    x = z_s.double().clone().requires_grad_(True)
    torch_soft_ce(x, z_t.double()).sum().backward()
    print(f"paper sec3 fp64 reference: {x.grad.tolist()}")
    print("-- perturbation sweep: median / max relative error of grad vs fp64 barron reference (on dtype-rounded inputs, rows with nonzero reference)")
    for name in ("bf16", "fp32"):
        dtype = DTYPES[name]
        if dtype == torch.bfloat16 and device == "cuda":
            continue
        for eps in (1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0):
            teacher = (torch.randn(256, 100, dtype=torch.float64) * 3).to(dtype)
            student = (teacher.double() + eps * torch.randn(256, 100, dtype=torch.float64)).to(dtype)
            xr = student.double().clone().requires_grad_(True)
            barron_soft_target_cross_entropy(xr, teacher.double()).sum().backward()
            ref = xr.grad
            xt = student.double().clone().requires_grad_(True)
            torch_soft_ce(xt, teacher.double()).sum().backward()
            identical = (student == teacher).all(-1).float().mean().item()
            print(f"    {name} eps={eps:.0e}: rows with student==teacher after rounding {identical:.2f}; fp64 torch vs fp64 barron max abs {(xt.grad - ref).abs().max().item():.1e}")
            row = []
            for label, fn in (("torch", torch_soft_ce), ("barron", barron_soft_target_cross_entropy)):
                x = student.to(device).clone().requires_grad_(True)
                fn(x, teacher.to(device)).sum().backward()
                err = (x.grad.double().cpu() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-300)
                err = err[ref.norm(dim=-1) > 0]
                row.append(f"{label} {err.median().item():.2e}/{err.max().item():.2e}" if len(err) else f"{label} ref=0")
            print(f"{name} eps={eps:.0e}  " + "  ".join(row))


def e5(device):
    print(f"== E5a distillation under Adam ({device}) ==")
    for name in ("bf16", "fp32"):
        dtype = DTYPES[name]
        if dtype == torch.bfloat16 and device == "cuda":
            continue
        torch.manual_seed(0)
        teacher = (torch.randn(128, 53, dtype=torch.float64) * 4).to(dtype).to(device)
        for label, fn in (("torch", torch_soft_ce), ("barron", barron_soft_target_cross_entropy)):
            student = torch.nn.Parameter(teacher.clone())
            optimizer = torch.optim.Adam([student], lr=2e-3, eps=1e-8)
            for step in range(2001):
                optimizer.zero_grad()
                fn(student, teacher).mean().backward()
                optimizer.step()
                if step in (0, 10, 100, 500, 2000):
                    p = torch.softmax(teacher.double(), -1)
                    kl = (p * (torch.log_softmax(teacher.double(), -1) - torch.log_softmax(student.detach().double(), -1))).sum(-1)
                    drift = (student.detach().double() - teacher.double()).abs().max().item()
                    agree = (student.detach().argmax(-1) == teacher.argmax(-1)).float().mean().item()
                    print(f"{name} {label:6s} step {step:5d}: mean KL {kl.mean().item():.2e} max|z_S-z_T| {drift:.2e} argmax agree {agree:.3f}")
    print(f"== E5a' linear student ({device}) ==")
    for name in ("bf16", "fp32"):
        dtype = DTYPES[name]
        if dtype == torch.bfloat16 and device == "cuda":
            continue
        torch.manual_seed(0)
        features = torch.randn(256, 32, dtype=torch.float64).to(dtype).to(device)
        weight_t = (torch.randn(32, 53, dtype=torch.float64)).to(dtype).to(device)
        teacher = features @ weight_t
        for label, fn in (("torch", torch_soft_ce), ("barron", barron_soft_target_cross_entropy)):
            weight = torch.nn.Parameter(weight_t.clone())
            optimizer = torch.optim.Adam([weight], lr=2e-3, eps=1e-8)
            for step in range(501):
                optimizer.zero_grad()
                fn(features @ weight, teacher).mean().backward()
                optimizer.step()
            drift = (weight.detach().double() - weight_t.double()).abs().max().item()
            print(f"{name} {label:6s} after 500 steps: max|W_S-W_T| {drift:.2e}")
    print(f"== E5b hard label AdamW margin ({device}) ==")
    for name in ("fp32",):
        dtype = DTYPES[name]
        for eps in (1e-8, 1e-12):
            for weight_decay in (0.01, 0.1):
                for label, fn in (("torch", torch_ce), ("barron", barron_cross_entropy)):
                    logits = torch.nn.Parameter(torch.zeros(1, 31, dtype=dtype, device=device))
                    target = torch.zeros(1, dtype=torch.long, device=device)
                    optimizer = torch.optim.AdamW([logits], lr=1e-2, betas=(0.9, 0.98), eps=eps, weight_decay=weight_decay)
                    margins = []
                    for step in range(20000):
                        optimizer.zero_grad()
                        fn(logits, target).sum().backward()
                        optimizer.step()
                        z = logits.detach().double()
                        margins.append((z[0, 0] - torch.logsumexp(z[0, 1:], -1)).item())
                    tail = torch.tensor(margins[-5000:])
                    print(
                        f"{name} eps={eps:.0e} wd={weight_decay} {label:6s}: margin@1k {margins[999]:.2f} @5k {margins[4999]:.2f} "
                        f"@20k {margins[-1]:.2f}; last5k mean {tail.mean().item():.2f} std {tail.std().item():.3f} "
                        f"min {tail.min().item():.2f} max {tail.max().item():.2f}; log(1/eps)={math.log(1/eps):.2f}"
                    )


def e6(device):
    print(f"== E6 edge cases ({device}) ==")
    inf = float("inf")
    cases = {
        "tie at max": ([[3.0, 3.0, 0.0]], 0),
        "single class": ([[1.5]], 0),
        "some -inf": ([[2.0, -inf, 0.0]], 0),
        "all but target -inf": ([[2.0, -inf, -inf]], 0),
        "target -inf": ([[-inf, 1.0, 0.0]], 0),
        "all -inf": ([[-inf, -inf, -inf]], 0),
        "+inf": ([[inf, 1.0, 0.0]], 0),
        "nan": ([[float("nan"), 1.0, 0.0]], 0),
        "all but max -inf, target other": ([[2.0, -inf, -inf]], 1),
    }
    for case, (values, target) in cases.items():
        z = torch.tensor(values, device=device)
        t = torch.tensor([target], device=device)
        out = [case]
        for label, fn in (("torch", torch_ce), ("barron", barron_cross_entropy)):
            loss, grad = ce_value_and_grad(fn, z, t)
            out.append(f"{label} ce {loss.tolist()} grad {grad.tolist()}")
        for label, fn in (("torch", lambda x: torch.log_softmax(x, -1)), ("barron", barron_log_softmax)):
            x = z.clone().requires_grad_(True)
            y = fn(x)
            y[..., target].sum().backward()
            out.append(f"{label} logsm {y.tolist()} grad {x.grad.tolist()}")
        for label, fn in (("torch", torch_soft_ce), ("barron", barron_soft_target_cross_entropy)):
            x = z.clone().requires_grad_(True)
            y = fn(x, z)
            y.sum().backward()
            out.append(f"{label} softce(z,z) {y.tolist()} grad {x.grad.tolist()}")
        print("\n    ".join(out))
    empty = torch.zeros(0, 5, device=device)
    print("empty:", barron_log_softmax(empty).shape, barron_cross_entropy(empty, torch.zeros(0, dtype=torch.long, device=device)).shape,
          barron_soft_target_cross_entropy(empty, empty).shape)
    z = torch.randn(5, 7, device=device).t()
    print("non-contiguous max diff:", (barron_log_softmax(z) - torch.log_softmax(z, -1)).abs().max().item())
    z = torch.randn(2, 3, 4, 5, device=device)
    t = torch.randint(0, 5, (2, 3, 4), device=device)
    print("leading dims CE max diff:", (barron_cross_entropy(z, t) - torch_ce(z, t)).abs().max().item())


def e7():
    print("== E7 autocast fp16 on CUDA ==")
    margins = torch.arange(0.0, 40.0, 0.01, dtype=torch.float64)
    z16 = torch.stack([margins, torch.zeros_like(margins)], dim=-1).half().cuda()
    t = torch.zeros(len(margins), dtype=torch.long, device="cuda")
    m = z16[:, 0].double().cpu()
    for autocast in (False, True):
        for label, fn in (("torch", torch_ce), ("barron", barron_cross_entropy)):
            z = z16.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
                loss = fn(z, t)
            loss.sum().backward()
            zero = (z.grad[:, 0] == 0).nonzero()
            first = m[zero[0, 0]].item() if len(zero) else float("nan")
            zero_l = (loss == 0).nonzero()
            first_l = m[zero_l[0, 0]].item() if len(zero_l) else float("nan")
            print(f"autocast={autocast} {label:6s}: loss dtype {loss.dtype}, grad dtype {z.grad.dtype}, first m target grad==0: {first:.2f}, loss==0: {first_l:.2f}")
    # Upcast inside autocast? Check intermediate dtypes for the port.
    with torch.autocast("cuda", dtype=torch.float16):
        x = torch.randn(4, 8, device="cuda").half()
        print("autocast barron_log_softmax out dtype", barron_log_softmax(x).dtype, "softmax", barron_softmax(x).dtype,
              "logsumexp", torch.logsumexp(x, -1).dtype, "softplus", F.softplus(x).dtype)
    # fp32 logits produced by a fp16 matmul under autocast, the common case.
    weight = torch.randn(16, 2, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = torch.randn(4, 16, device="cuda") @ weight
        print("matmul logits dtype under autocast:", logits.dtype)


def e8(device):
    print(f"== E8 TorchScript and torch.compile ({device}) ==")
    scripted = {fn.__name__: torch.jit.script(fn) for fn in (barron_log_softmax, barron_softmax, barron_cross_entropy, barron_soft_target_cross_entropy)}
    z = torch.tensor([[9.0, -9.0, -9.0], [20.0, 0.0, 0.0]], device=device)
    t = torch.tensor([0, 0], device=device)
    z_t = torch.tensor([[2.0, 1.0, 0.0]], device=device)
    z_s = z_t + torch.tensor([[-1e-7, 1e-7, 0.0]], device=device)

    def run(fns):
        out = {}
        x = z.clone().requires_grad_(True)
        y = fns["barron_cross_entropy"](x, t)
        y.sum().backward()
        out["ce"] = (y.detach(), x.grad)
        x = z.clone().requires_grad_(True)
        y = fns["barron_log_softmax"](x)
        y[:, 0].sum().backward()
        out["logsm"] = (y.detach(), x.grad)
        x = z.clone().requires_grad_(True)
        y = fns["barron_softmax"](x)
        y[:, 0].sum().backward()
        out["sm"] = (y.detach(), x.grad)
        x = z_s.clone().requires_grad_(True)
        y = fns["barron_soft_target_cross_entropy"](x, z_t)
        y.sum().backward()
        out["softce"] = (y.detach(), x.grad)
        x = z_t.clone().requires_grad_(True)
        y = fns["barron_soft_target_cross_entropy"](x, z_t)
        y.sum().backward()
        out["softce_equal"] = (y.detach(), x.grad)
        return out

    eager = {fn.__name__: fn for fn in (barron_log_softmax, barron_softmax, barron_cross_entropy, barron_soft_target_cross_entropy)}
    reference = run(eager)
    for key, (value, grad) in reference.items():
        print(f"eager {key}: value {value.tolist()} grad {grad.tolist()}")
    for name, fns in (("script", scripted),):
        result = run(fns)
        for key in reference:
            same = torch.equal(result[key][0], reference[key][0]) and torch.equal(result[key][1], reference[key][1])
            print(f"{name} {key}: bitwise equal to eager: {same}")
    try:
        compiled = {k: torch.compile(v, dynamic=False) for k, v in eager.items()}
        result = run(compiled)
        for key in reference:
            same = torch.equal(result[key][0], reference[key][0]) and torch.equal(result[key][1], reference[key][1])
            print(f"compile {key}: bitwise equal {same}; value {result[key][0].tolist()} grad {result[key][1].tolist()}")
    except Exception as error:  # noqa: BLE001 - debug script
        print("torch.compile failed:", type(error).__name__, str(error)[:500])


def bench(fn, *args, steps=20):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for _ in range(3):
        x = args[0].clone().requires_grad_(True)
        fn(x, *args[1:]).sum().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        x = args[0].clone().requires_grad_(True)
        fn(x, *args[1:]).sum().backward()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / steps * 1e3
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    return elapsed, peak


def e9():
    print("== E9 cost on CUDA (fwd+bwd ms, peak extra MiB) ==")
    for shape in ((256, 3721), (4096, 32000)):
        z = torch.randn(*shape, device="cuda")
        zt = torch.randn(*shape, device="cuda")
        t = torch.randint(0, shape[1], (shape[0],), device="cuda")
        rows = {
            "log_softmax": (lambda x: torch.log_softmax(x, -1), barron_log_softmax),
            "softmax": (lambda x: torch.softmax(x, -1), barron_softmax),
            "ce": (lambda x: F.cross_entropy(x, t, reduction="none"), lambda x: barron_cross_entropy(x, t)),
            "soft_ce": (lambda x: torch_soft_ce(x, zt), lambda x: barron_soft_target_cross_entropy(x, zt)),
        }
        for name, (torch_fn, barron_fn) in rows.items():
            torch_time, torch_mem = bench(torch_fn, z)
            barron_time, barron_mem = bench(barron_fn, z)
            print(
                f"{shape} {name:11s} torch {torch_time:7.2f} ms {torch_mem:8.1f} MiB | barron {barron_time:7.2f} ms {barron_mem:8.1f} MiB "
                f"| ratio time {barron_time / torch_time:5.2f}x mem {barron_mem / max(torch_mem, 1e-9):5.2f}x"
            )
        del z, zt
        torch.cuda.empty_cache()


def e10():
    print("== E10 second order ==")
    torch.manual_seed(0)
    z = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    t = torch.randint(0, 5, (3,))
    zt = torch.randn(3, 5, dtype=torch.float64)
    for name, fn in (
        ("log_softmax", barron_log_softmax),
        ("softmax", barron_softmax),
        ("ce", lambda x: barron_cross_entropy(x, t)),
    ):
        print(name, "gradcheck", torch.autograd.gradcheck(fn, (z,)), "gradgradcheck", torch.autograd.gradgradcheck(fn, (z,)))
    zt_in = zt.clone().requires_grad_(True)
    print("soft_ce gradcheck (student, teacher):", torch.autograd.gradcheck(barron_soft_target_cross_entropy, (z, zt_in)))
    try:
        print("soft_ce gradgradcheck:", torch.autograd.gradgradcheck(barron_soft_target_cross_entropy, (z, zt_in), raise_exception=False))
    except Exception as error:  # noqa: BLE001 - debug script
        print("soft_ce gradgradcheck raised:", str(error)[:300])
    hessian_barron = torch.autograd.functional.hessian(lambda x: barron_soft_target_cross_entropy(x, zt).sum(), z.detach())
    hessian_torch = torch.autograd.functional.hessian(lambda x: torch_soft_ce(x, zt).sum(), z.detach())
    print("soft_ce Hessian max |barron - torch|:", (hessian_barron - hessian_torch).abs().max().item(), "max |torch|", hessian_torch.abs().max().item())


def probe():
    print("== probe: inductor CPU precision of expm1/log1p and soft-ce pieces ==")
    x = torch.tensor([1e-7, -1e-7, 3e-8, 1e-5, 0.0], dtype=torch.float32)
    try:
        torch.compile(torch.expm1)(x)
    except Exception as error:  # noqa: BLE001
        print("torch.compile unavailable on CPU:", type(error).__name__, str(error)[:200])
        return
    for name, fn in (("expm1", torch.expm1), ("log1p", torch.log1p)):
        eager = fn(x)
        compiled = torch.compile(fn)(x)
        exact = fn(x.double())
        print(name, "eager rel err", ((eager.double() - exact).abs() / exact.abs().clamp_min(1e-300)).tolist())
        print(name, "compiled rel err", ((compiled.double() - exact).abs() / exact.abs().clamp_min(1e-300)).tolist())
    x8 = torch.full((8,), 1e-7)  # vectorized path
    for name, fn in (("expm1", torch.expm1), ("log1p", torch.log1p)):
        compiled = torch.compile(fn)(x8)
        exact = fn(x8.double())
        print(name, "compiled (vec) rel err", ((compiled.double() - exact).abs() / exact.abs()).max().item())
    z_t = torch.tensor([[2.0, 1.0, 0.0]])
    z_s = z_t + torch.tensor([[-1e-7, 1e-7, 0.0]])

    def pieces(student, teacher):
        log_q = barron_log_softmax(student)
        q = log_q.exp()
        p = barron_softmax(teacher)
        delta = student - teacher
        centered = delta - (p * delta).sum(dim=-1, keepdim=True)
        log_ratio = centered - torch.log1p((p * torch.expm1(centered)).sum(dim=-1, keepdim=True))
        difference = p * torch.expm1(log_ratio)
        gradient = difference - q * difference.sum(dim=-1, keepdim=True)
        return delta, centered, log_ratio, difference, gradient

    eager = pieces(z_s, z_t)
    compiled = torch.compile(pieces)(z_s, z_t)
    reference = pieces(z_s.double(), z_t.double())
    for name, e, c, r in zip(("delta", "centered", "log_ratio", "difference", "gradient"), eager, compiled, reference):
        print(f"{name:10s} eager {e.tolist()}\n           compiled {c.tolist()}\n           fp64 {r.tolist()}")
    for device in ["cpu"] + (["cuda"] if CUDA else []):
        print(f"== probe: TorchScript after warm-up vs eager ({device}) ==")
        z = torch.tensor([[9.0, -9.0, -9.0], [20.0, 0.0, 0.0], [30.0, 1.0, -2.0]], device=device)
        t = torch.tensor([0, 0, 0], device=device)
        z_t = torch.tensor([[2.0, 1.0, 0.0]], device=device)
        z_s = z_t + torch.tensor([[-1e-7, 1e-7, 0.0]], device=device)
        calls = (
            ("log_softmax", barron_log_softmax, lambda f, x: f(x)[:, 0].sum(), z),
            ("softmax", barron_softmax, lambda f, x: f(x)[:, 0].sum(), z),
            ("ce", barron_cross_entropy, lambda f, x: f(x, t).sum(), z),
            ("soft_ce", barron_soft_target_cross_entropy, lambda f, x: f(x, z_t).sum(), z_s),
        )
        for name, fn, loss_of, inputs in calls:
            scripted = torch.jit.script(fn)
            for _ in range(4):
                x = inputs.clone().requires_grad_(True)
                loss_of(scripted, x).backward()
            xe = inputs.clone().requires_grad_(True)
            loss_of(fn, xe).backward()
            print(f"{name:11s} scripted grad == eager grad: {torch.equal(x.grad, xe.grad)}; max |diff| {(x.grad - xe.grad).abs().max().item():.2e}; eager max|grad| {xe.grad.abs().max().item():.2e}")
    if False:
        z = torch.tensor([[9.0, -9.0, -9.0], [20.0, 0.0, 0.0], [30.0, 1.0, -2.0]], device="cuda")
        t = torch.tensor([0, 0, 0], device="cuda")
        for fn in (barron_log_softmax, barron_softmax):
            scripted = torch.jit.script(fn)
            for _ in range(3):  # profiling executor specializes after warm-up runs
                x = z.clone().requires_grad_(True)
                y = scripted(x)
                y[:, 0].sum().backward()
            xe = z.clone().requires_grad_(True)
            ye = fn(xe)
            ye[:, 0].sum().backward()
            print(fn.__name__, "value diff", (y - ye).abs().max().item(), "grad diff", (x.grad - xe.grad).abs().max().item())
            print("   scripted", y.tolist(), x.grad.tolist())
            print("   eager   ", ye.tolist(), xe.grad.tolist())
        scripted = torch.jit.script(barron_cross_entropy)
        for _ in range(3):
            x = z.clone().requires_grad_(True)
            y = scripted(x, t)
            y.sum().backward()
        xe = z.clone().requires_grad_(True)
        ye = barron_cross_entropy(xe, t)
        ye.sum().backward()
        print("ce scripted", y.tolist(), x.grad.tolist())
        print("ce eager   ", ye.tolist(), xe.grad.tolist())


if __name__ == "__main__":
    all_experiments = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "probe"]
    experiments = sys.argv[1:] or all_experiments
    unknown = sorted(set(experiments) - set(all_experiments))
    if unknown:
        sys.exit(f"unknown experiments {unknown}; choose from {all_experiments}")
    torch.set_printoptions(precision=10)
    print("torch", torch.__version__, "cuda", CUDA, torch.cuda.get_device_name() if CUDA else "")
    for experiment in experiments:
        if experiment == "probe":
            probe()
        elif experiment in ("e7", "e9"):
            if CUDA:
                globals()[experiment]()
            else:
                print(f"{experiment} needs CUDA; skipped")
        elif experiment == "e10":
            e10()
        else:
            for device in ["cpu"] + (["cuda"] if CUDA else []):
                globals()[experiment](device)
