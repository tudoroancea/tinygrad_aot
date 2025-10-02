from dataclasses import dataclass
from functools import cached_property
from itertools import count
from typing import Protocol, cast

from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import get_program
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.renderer import ProgramSpec
from tinygrad.renderer.cstyle import ClangRenderer, CStyleLanguage
from tinygrad.tensor import Tensor
from tinygrad.uop import Ops

from tinygrad_aot.utils import OneOrMore, Shape

__all__ = ["aot", "Codegenable"]


class CustomClangRenderer(ClangRenderer):
  """
  Custom renderer that renders certain ops as builtin clang functions instead of polynomial decompositions.
  NOTE: adding these ops in code_for_op also automatically disables the decompositions because it marks them as 'supported'.
  """

  code_for_op = {
    **({k: v for k, v in CStyleLanguage.code_for_op.items() if k not in [Ops.RECIP]}),
    Ops.SIN: lambda x, dtype: f"__builtin_elementwise_sin({x})",
    Ops.EXP2: lambda x, dtype: f"__builtin_elementwise_exp2({x})",
    Ops.LOG2: lambda x, dtype: f"__builtin_elementwise_log2({x})",
    Ops.TRUNC: lambda x, dtype: f"__builtin_elementwise_trunc({x})",
    Ops.SQRT: lambda x, dtype: f"__builtin_sqrt({x})" if dtype == dtypes.float64 else f"__builtin_sqrtf({x})",
    Ops.TRUNC: lambda x, dtype: f"__builtin_trunc({x})" if dtype == dtypes.float64 else f"__builtin_truncf({x})",
    Ops.FDIV: lambda a, b, dtype: f"({a}/{b})",
  }


def render_kernels(schedule: list[ScheduleItem], var_vals: dict[str, int] | None = None) -> list[ProgramSpec]:
  renderer = CustomClangRenderer()
  assert all(buf.device == renderer.device for si in schedule for buf in si.bufs)
  return [get_program(si.ast, renderer) for si in schedule]


@dataclass(frozen=True)
class BufferClassification:
  inputs: tuple[Buffer, ...]
  outputs: tuple[Buffer, ...]
  intermediates: tuple[Buffer, ...]

  @cached_property
  def all_bufs(self):
    return self.inputs + self.outputs + self.intermediates

  @cached_property
  def buf_names(self):
    """Returns a dict with unique names for each buffer (input/output/intermediate)."""
    input_counter, output_counter, intermediate_counter = count(), count(), count()
    return (
      {buf: f"in{next(input_counter)}_{buf.size}" for buf in self.inputs}
      | {buf: f"out{next(output_counter)}_{buf.size}" for buf in self.outputs}
      | {buf: f"tmp{next(intermediate_counter)}_{buf.size}" for buf in self.intermediates}
    )

  @cached_property
  def intermediate_buf_offsets(self) -> dict[Buffer, int]:
    """Returns a dict with the offset in bytes of each intermediate buffer in a global workspace vector.
    For the moment we assume all the buffers have the same dtype, so that we can ignore alignment causing buffers to not be contiguously laid out.
    """
    assert all(buf.dtype == self.all_bufs[0].dtype for buf in self.all_bufs)
    offsets = {}
    offset = 0
    for buf in self.intermediates:
      from icecream import ic

      ic(buf, offset)
      offsets[buf] = offset
      offset += buf.size
    return offsets

  @cached_property
  def ws_size(self):
    """Returns the size in bytes of the global workspace vector."""
    return sum(buf.size for buf in self.intermediates)


def classify_buffers(schedule: list[ScheduleItem], rendered_kernels: list[ProgramSpec]) -> BufferClassification:
  """Classify all the buffers used throughout the computation between inputs/outputs and intermediate buffers (that have to be heap allocated)."""
  ctx = {"ins": set(), "outs": set()}
  for si, rk in zip(schedule, rendered_kernels):
    ctx["ins"].update(set(si.bufs[i] for i in rk.ins))
    ctx["outs"].update(set(si.bufs[i] for i in rk.outs))

  return BufferClassification(
    inputs=tuple(ctx["ins"] - ctx["outs"]),
    outputs=tuple(ctx["outs"] - ctx["ins"]),
    intermediates=tuple(ctx["ins"] & ctx["outs"]),
  )


def render_schedule(
  name: str,
  schedule: list[ScheduleItem],
  rendered_kernels: list[ProgramSpec],
  bc: BufferClassification,
) -> str:
  """Create a single C source code with all the kernels and the assembled function."""
  c_code = []
  c_code.append('#ifdef __cplusplus\nextern "C" {\n#endif')
  c_code.append("#include <stdlib.h>")
  c_code.append("#include <stdint.h>")  # for uint8_t
  c_code.append("#include <string.h>")  # for memset

  # Add kernel sources
  for rk in rendered_kernels:
    c_code.append(rk.src.replace("restrict", "__restrict__"))

  # declare global function
  base_renderer = cast(CStyleLanguage, Device[bc.inputs[0].device].renderer)
  c_code.append(
    f"void {name}("
    + ", ".join(
      [f"{base_renderer.render_dtype(buf.dtype)}* __restrict__ {bc.buf_names[buf]}" for buf in bc.outputs + bc.inputs]
    )
    + ", uint8_t* __restrict__ ws) {"
  )

  # clear ws memory
  c_code.append(f"\tmemset(ws, 0, {bc.ws_size});")

  # extract intermediate buffers from workspace buffer
  for buf in bc.intermediates:
    dtype_str = base_renderer.render_dtype(buf.dtype)
    c_code.append(f"\t{dtype_str}* {bc.buf_names[buf]} = ({dtype_str}*) ws + {bc.intermediate_buf_offsets[buf]};")

  # call all kernels
  for si, rk in zip(schedule, rendered_kernels):
    c_code.append(f"\t{rk.function_name}({', '.join(bc.buf_names[si.bufs[i]] for i in rk.globals)}, 0);")
    # NOTE: the following doesn't work because in some cases (see bicycle_disc_6 test case) si.bufs
    # contains buffers that are not passed to the kernel. WTF
    # c_code.append(f"\t{rk.function_name}({','.join(bc.buf_names[buf] for buf in si.bufs)}, 0);")

  c_code.append("}")

  c_code.append("#ifdef __cplusplus\n}\n#endif")

  return "\n".join(c_code)


class Codegenable(Protocol):
  def __call__(self, *args: Tensor) -> OneOrMore[Tensor]: ...


def aot(name: str, fn: Codegenable, inshape: OneOrMore[Shape]):
  """Full AOT codegen pipeline"""

  # Get the output tensor to construct the base AST
  ins = (Tensor.empty(inshape),) if isinstance(inshape[0], int) else tuple(Tensor.empty(shape) for shape in inshape)
  outs = fn(*ins)

  # Scheduling (breakdown AST into kernels)
  schedule, var_vals = outs.schedule_with_vars() if isinstance(outs, Tensor) else outs[0].schedule_with_vars(*outs[1:])

  # Render kernels
  rendered_kernels = render_kernels(schedule, var_vals)

  # Buffer classification
  bc = classify_buffers(schedule, rendered_kernels)

  bc.intermediate_buf_offsets

  return render_schedule(name, schedule, rendered_kernels, bc)
