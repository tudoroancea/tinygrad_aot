import os
import subprocess
from functools import wraps
from typing import Callable

import pytest
from tinygrad.uop.ops import UOp

from tinygrad import Tensor
from tinygrad_aot import aot

##################################################################
# Some functions to test
##################################################################


def grad(fn: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
  @wraps(fn)
  def g(x: Tensor) -> Tensor:
    old_requires_grad = x.requires_grad
    x.requires_grad = True
    x.grad = None
    fn(x).backward()
    x.requires_grad = old_requires_grad
    return x.grad

  return g


def linear_sum(x: Tensor) -> Tensor:
  return (0.5 * x).sum()


linear_sum_grad = grad(linear_sum)


def do_square(x: Tensor) -> Tensor:
  return 0.5 * x.square()


do_square_grad = grad(do_square)


def _bicycle_ocp(x: Tensor, u: Tensor) -> Tensor:
  theta = x[2]
  v = x[3]
  a = u[0]
  delta = u[1]
  dx = v * Tensor.cos(theta)
  dy = v * Tensor.sin(theta)
  dtheta = v * Tensor.tan(delta)
  dv = a
  return Tensor.stack([dx, dy, dtheta, dv], dim=-1)


def _rk4(fn: Callable[[Tensor], Tensor], x: Tensor, u: Tensor, dt: float) -> Tensor:
  k1 = fn(x, u)
  k2 = fn(x + dt * k1 / 2, u)
  k3 = fn(x + dt * k2 / 2, u)
  k4 = fn(x + dt * k3, u)
  return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def bicycle_cont(z: Tensor) -> Tensor:
  return _bicycle_ocp(z[:4], z[4:])


def bicycle_cont2(z: Tensor) -> Tensor:
  theta = z[:, 2]
  v = z[:, 3]
  a = z[:, 4]
  delta = z[:, 5]
  dx = v * Tensor.cos(theta)
  dy = v * Tensor.sin(theta)
  dtheta = v * Tensor.tan(delta)
  dv = a
  return Tensor.stack([dx, dy, dtheta, dv], dim=0)


def bicycle_disc(z: Tensor) -> Tensor:
  return _rk4(_bicycle_ocp, z[:4], z[4:], 0.01)


def f(x: Tensor) -> Tensor:
  return x[0] ** 2 + x[1] ** 3


def f2(x: Tensor) -> Tensor:
  return x[:, 0] ** 2 + x[:, 1] ** 3


def f3(x: Tensor) -> Tensor:
  a = UOp.range(x.shape[0], 1)
  return f(x[a]).contiguous(a)


##################################################################
# Test AOT
##################################################################

args = [
  ("do_square_8", do_square, (8,)),
  ("do_square_1024", do_square, (1024,)),
  ("linear_sum", linear_sum, (1024,)),
  pytest.param("linear_sum_grad", linear_sum_grad, (1024,), marks=pytest.mark.xfail),
  ("bicycle_cont_6", bicycle_cont, (6,)),
  ("bicycle_cont_6_10", bicycle_cont, (6, 10)),
  ("bicycle_cont2_10_6", bicycle_cont2, (10, 6)),
  ("bicycle_disc_6", bicycle_disc, (6,)),
  pytest.param("bicycle_disc_6_10", bicycle_disc, (6, 10), marks=pytest.mark.xfail),
]


@pytest.mark.parametrize("name, fn, inshape", args)
def test_run(name: str, fn: Callable[[Tensor], Tensor], inshape: tuple[int, ...]):
  fn(Tensor.ones(inshape)).realize()


@pytest.mark.parametrize("name, fn, inshape", args)
def test_aot(name: str, fn: Callable[[Tensor], Tensor], inshape: tuple[int, ...]):
  prefix = os.path.join(os.path.dirname(__file__), "aot_out")
  os.makedirs(prefix, exist_ok=True)
  aot_src = aot(name, fn, inshape)
  with open(f"{prefix}/{name}.c", "w") as file:
    file.write(aot_src)
  try:
    subprocess.run(
      ["clang", f"{prefix}/{name}.c", "-c", "-o", f"{prefix}/{name}.o"],
      check=True,
      text=True,
      capture_output=True,
    )
  except subprocess.CalledProcessError as e:
    raise pytest.fail(f"Compilation failed: {e.stderr}")
