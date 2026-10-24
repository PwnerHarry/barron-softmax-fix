# barron-softmax-fix

A PyTorch port of [**A Fix for Softmax and Cross-Entropy**](https://jonbarron.info/softmaxfix/) by Jonathan T. Barron, Dan Gnanapragasam and Michael Tschannen (arXiv:2610.31642). It includes tests that show where the torch built-ins go wrong, and scripts that check the paper's claims.

This is an independent port. It is not affiliated with the authors.

## What it fixes

The standard softmax, log-softmax and cross-entropy have two numerical bugs:

- **Softmax Collapse.** Once the target logit's margin over the other logits passes a precision cliff, `1 - p_t` rounds to 0. The cliff is about 5.5 in bf16, 16.6 in fp32 and 36.7 in fp64. The target gradient then becomes exactly 0 while the other gradients do not, so the gradient no longer sums to 0. AdamW turns that into Slingshot loss spikes. With the default `eps=1e-8` it also settles the margin near `log(1/eps) ≈ 18.4`, which is past the fp32 cliff.
- **Softmax Cancellation.** The distillation gradient `q - p` is a difference of nearly equal probabilities. It does not reach 0 when the student matches the teacher, so Adam pushes a perfect student away.

The fix computes everything from the logit margin `m = z_pivot - logsumexp(other logits)`. The pivot entry is held at the constant 0, so `1 - p` is never formed by subtracting from 1. The soft-target gradient is computed from the logit difference with `expm1` and `log1p`.

## Install

```bash
pip install git+https://github.com/PwnerHarry/barron-softmax-fix
```

The only dependency is `torch>=2.0`.

## Usage

| torch | this package |
|---|---|
| `torch.log_softmax(z, dim=-1)` | `barron_log_softmax(z)` |
| `torch.softmax(z, dim=-1)` | `barron_softmax(z)` |
| `F.cross_entropy(z, t, reduction="none")` | `barron_cross_entropy(z, t)` |
| `F.cross_entropy(z_s, F.softmax(z_t, -1), reduction="none")` | `barron_soft_target_cross_entropy(z_s, z_t)` |

```python
from barron_softmax_fix import barron_cross_entropy, barron_soft_target_cross_entropy

loss = barron_cross_entropy(logits, labels).mean()  # logits [..., C], labels [...] int64
kd = barron_soft_target_cross_entropy(student_logits / T, teacher_logits.detach() / T).mean()
```

- All functions reduce over the last dimension and return unreduced values.
- The soft-target loss takes the teacher **logits**, not probabilities.
- Apply any temperature to both sets of logits, and detach the teacher if it is frozen.

## What was verified

The `torch` results are for torch 2.11 (Tesla T4 and CPU) unless noted. The cliffs and worked examples were rechecked on torch 2.14.1 (CPU).

| | torch | this package |
|---|---|---|
| Margin where the CE target gradient becomes exactly 0 (bf16 / fp32 / fp64) | 5.56 / 16.64 / 36.74, matching the paper | 93 / 104 / none (exponent underflow) |
| Paper §2 example, `z=[9,-9,-9]`, `t=0`, fp32: CE value and target gradient | `0.0`, `0.0` | `3.046e-8`, `-3.046e-8`, matching the paper |
| `\|sum(grad)\| / max\|grad\|`, fp32, 32k classes, margin 20 | about 1e3 | about 1e-7 |
| Soft-target `max\|grad\|` when student equals teacher (bf16 / fp32) | 2e-3 / 3e-7 | exactly 0 |
| Paper §3 example: fp32 gradient against the fp64 reference | 8–65% error per entry | matches to 7 digits |
| Ordinary logits, fp32: value and CE-gradient error against fp64 | 3e-8 to 6e-6 | 4e-8 to 4e-6 |

### Small-scale reproduction of the paper's training experiments

The **Slingshot / grokking** run uses `a + b mod 31`, a 1-layer width-64 ReLU transformer, a 50% split and 10k full-batch AdamW steps (lr 1e-3, betas (0.9, 0.98), weight decay 1), with 3 seeds:

| eps | | torch fp32 | this package, fp32 | torch fp64 |
|---|---|---|---|---|
| 1e-8 | loss spikes >100x in the second half | 477–651 per run | 0 | 0 |
| 1e-8 | final median train margin | 6.1–9.0 | 16.4–16.5 | 16.4–16.6 |
| 1e-12 | final median train margin | 10.5–14.5 | 25.8–26.0 | 25.8–26.0 |

The fixed loss in fp32 behaves like fp64. Torch in fp32 shows Slingshot spikes and never reaches the equilibrium margin. With this model no variant grokked within 10k steps, so the paper's grokking speed-up was **not** tested here.

The **distillation** run trains a teacher on `a + b mod 53` to 100% test accuracy. The student starts at the teacher and takes 2,000 Adam steps (lr 2e-3, eps 1e-8, T = 4):

| | KL(teacher‖student) | max parameter drift |
|---|---|---|
| torch fp32 / bf16 | 1.2e-6 / 2.5e-5 | 5e-2 / 1.1e-1 |
| this package fp32 / bf16 | exactly 0 / exactly 0 | exactly 0 / exactly 0 |

## Differences from the paper's JAX listings

None of these changes a finite result:

- **softplus.** `softplus(-m)` is written as `-logsigmoid(m)`. TorchScript's derivative formula for `softplus` is `(exp(y) - 1) / exp(y)`. Once a scripted function has warmed up, that gives a **zero** gradient exactly where the fix matters. `logsigmoid` stays accurate under TorchScript, `torch.compile` and float16.
- **Masked logits.** Logits of `-inf` give finite gradients where the listings give NaN:
  - a row with one unmasked class gets a zero gradient;
  - a masked CE target gets an infinite loss with torch's gradient;
  - the soft-target loss ignores classes that both logits mask.
- **Shapes.** Mismatched shapes raise `ValueError` instead of broadcasting.

## Limits

- **Speed.** It is 3.4–6x slower than the fused torch kernels, and uses 1.25–1.75x the memory (about 3x for the soft-target loss). That was measured on a T4 at `[256, 3721]` and `[4096, 32000]`.
- **Precision.** Pass fp32 logits. The functions do not cast, so in bf16 and fp16 every intermediate is rounded. Away from the cliff that is noisier than torch's kernels, which accumulate in fp32. In fp16 the gradient underflows at a margin of about 17.
- **Arbitrary upstream gradients.** The `barron_log_softmax` gradient at the argmax is minus the sum over the other classes. With an arbitrary upstream gradient over many classes it loses relative precision. Cross-entropy upstreams are unaffected.
- **Soft-target second derivatives.** The soft-target loss has no second derivative with respect to the student logits. This follows from the paper's stop-gradient construction.
- **`torch.compile` on CPU with torch 2.11.** Inductor computes `expm1(x)` as `exp(x) - 1`, which undoes the soft-target fix. Torch 2.14 does not.

## Tests and validation

```bash
pip install -e ".[test]"
pytest
```

- **Correctness tests:** parity with torch in fp64, `gradcheck`, edge cases, and TorchScript after warm-up.
- **"Where torch is wrong" tests:** these also assert that the torch built-in fails on the same input. If a torch release fixes one of these bugs, that test fails, and the corresponding part of this package may no longer be needed.

`validation/` holds the scripts behind the numbers above:

```bash
python validation/numerics.py            # precision cliffs, parity, edge cases, TorchScript/compile, cost, ...
python validation/numerics.py e1 e4      # selected experiments; see the module docstring
python validation/training.py grokking
python validation/training.py distillation
```

The training scripts use CUDA when it is available. bf16 on CPU is slow.

## Citation

Please cite the paper:

```bibtex
@article{barron2026softmaxfix,
  title   = {A Fix for Softmax and Cross-Entropy},
  author  = {Barron, Jonathan T. and Gnanapragasam, Dan and Tschannen, Michael},
  journal = {arXiv preprint arXiv:2610.31642},
  year    = {2026}
}
```

## License

This port is released under the MIT license; see `LICENSE`. The algorithm and the reference JAX code are from the paper above.
