import os
import subprocess
from functools import wraps
from typing import Callable

import pytest
from tinygrad import Tensor
from tinygrad.helpers import prod
from tinygrad.uop.ops import UOp

from tinygrad_aot import Codegenable, OneOrMore, Shape, aot

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


def euler(fn: Callable[[Tensor, Tensor], Tensor], x: Tensor, u: Tensor, dt: float) -> Tensor:
  return x + dt * fn(x, u)


def rk2(fn: Callable[[Tensor, Tensor], Tensor], x: Tensor, u: Tensor, dt: float) -> Tensor:
  k1 = fn(x, u)
  k2 = fn(x + dt * k1 / 2, u)
  return x + dt * k2


def rk4(fn: Callable[[Tensor, Tensor], Tensor], x: Tensor, u: Tensor, dt: float) -> Tensor:
  k1 = fn(x, u)
  k2 = fn(x + dt * k1 / 2, u)
  k3 = fn(x + dt * k2 / 2, u)
  k4 = fn(x + dt * k3, u)
  return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def bicycle_cont(x: Tensor, u: Tensor) -> Tensor:
  theta = x[2]
  v = x[3]
  a = u[0]
  delta = u[1]
  dx = v * Tensor.cos(theta)
  dy = v * Tensor.sin(theta)
  dtheta = v * Tensor.tan(delta).contiguous()
  dv = a
  return Tensor.stack(dx, dy, dtheta, dv)


def lti_cont(x: Tensor, u: Tensor):
  # NOTE: the following will incur a copy from a buffer with PYTHON device to a buffer with CPU device (no inlined constant)
  # return Tensor(np.array([[1.0, 2.0], [0.0, 3.0]])) @ x + Tensor(np.array([[0.0], [1.0]])) @ u
  return Tensor.stack(x[0] + 2 * x[1], 3 * x[1] + u[0])


def pol(x: Tensor) -> Tensor:
  return x[0] ** 2 + x[1] ** 3


def pol2(x: Tensor) -> Tensor:
  return x[:, 0] ** 2 + x[:, 1] ** 3


def pol3(x: Tensor) -> Tensor:
  a = UOp.range(x.shape[0], 1)
  return pol(x[a]).contiguous(a)


def eq_constraints(Z: Tensor) -> Tensor:
  """
  Z = [x0, u0, ..., x_{N-1}, u_{N-1}, x_N]
  eq_constraints = [x0, x1 - f(x0, u0), ..., xN - f(xN-1, uN-1)]
  """
  nx, nu = 4, 2
  N = Z.shape[0] // (nx + nu)
  # NOTE: contiguous below makes the same state be contiguous in memory and should help with vectorization
  staging = Z.pad_to((nx + nu) * (N + 1)).reshape((N + 1, nx + nu)).transpose().contiguous()  # shape (nx + nu, N + 1)
  interstage = staging[:nx, 1:] - rk4(bicycle_cont, staging[:nx, :N], staging[nx:, :N], dt=1.0)  # shape (nx, N)
  return Tensor.cat(Z[:nx], interstage.transpose().flatten().contiguous())  # is this contiguous necessary?


##################################################################
# Test AOT
##################################################################


args = [
  ("do_square_8", do_square, (8,)),
  ("do_square_1024", do_square, (1024,)),
  ("linear_sum", linear_sum, (1024,)),
  ("bicycle_cont", bicycle_cont, ((4,), (2,))),
  ("bicycle_cont_10", bicycle_cont, ((4, 10), (2, 10))),
  ("bicycle_rk2", lambda x, u: rk2(bicycle_cont, x, u, 0.01), ((4,), (2,))),
  ("bicycle_rk4", lambda x, u: rk4(bicycle_cont, x, u, 0.01), ((4,), (2,))),
  ("pol_2", pol, (2,)),
  ("pol_2_1024", pol, (2, 1024)),
  ("pol_1024_2", pol2, (1024, 2)),
  ("eq_constraints_10", eq_constraints, ((4 + 2) * 10 + 4,)),
  ("lti_cont", lti_cont, ((2,), (1,))),
  ("lti_cont_10", lti_cont, ((2, 10), (1, 10))),
  ("lti_euler", lambda x, u: euler(lti_cont, x, u, 0.01), ((2,), (1,))),
  ("lti_euler_10", lambda x, u: euler(lti_cont, x, u, 0.01), ((2, 10), (1, 10))),
  ("lti_rk2", lambda x, u: rk2(lti_cont, x, u, 0.01), ((2,), (1,))),
  ("lti_rk2_10", lambda x, u: rk2(lti_cont, x, u, 0.01), ((2, 10), (1, 10))),
  ("lti_rk4", lambda x, u: rk4(lti_cont, x, u, 0.01), ((2,), (1,))),
  ("lti_rk4_10", lambda x, u: rk4(lti_cont, x, u, 0.01), ((2, 10), (1, 10))),
  # pytest.param("linear_sum_grad", linear_sum_grad, (1024,), marks=pytest.mark.skip),
]
argids = [arg[0] for arg in args]


@pytest.mark.parametrize("name, fn, inshape", args, ids=argids)
def test_run(name: str, fn: Codegenable, inshape: OneOrMore[Shape]):
  ins = (
    (Tensor.arange(prod(inshape)).reshape(inshape),)
    if isinstance(inshape[0], int)
    else tuple(Tensor.arange(prod(shape)).reshape(shape) for shape in inshape)
  )
  print(fn(*ins).numpy())


@pytest.mark.parametrize("name, fn, inshape", args, ids=argids)
def test_aot(name: str, fn: Codegenable, inshape: OneOrMore[Shape]):
  prefix = os.path.join(os.path.dirname(__file__), "aot_out")
  os.makedirs(prefix, exist_ok=True)
  aot_src = aot(name, fn, inshape)
  with open(f"{prefix}/{name}.c", "w") as file:
    file.write(aot_src)
  try:
    subprocess.run(
      ["clang", f"{prefix}/{name}.c", "-O3", "-mcpu=native", "-S", "-o", f"{prefix}/{name}.s"],
      check=True,
      text=True,
      capture_output=True,
    )
  except subprocess.CalledProcessError as e:
    raise pytest.fail(f"Compilation failed: {e.stderr}")
