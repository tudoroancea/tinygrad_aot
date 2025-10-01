from dataclasses import dataclass
from functools import cached_property
from itertools import count
from typing import Protocol, cast

from tinygrad.device import Buffer, Device
from tinygrad.engine.realize import get_program
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.renderer import ProgramSpec
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.tensor import Tensor

from tinygrad_aot.utils import OneOrMore, Shape

__all__ = ["aot", "Codegenable"]


def render_kernels(schedule: list[ScheduleItem], var_vals: dict[str, int] | None = None) -> list[ProgramSpec]:
  return [get_program(si.ast, Device[si.bufs[0].device].renderer) for si in schedule]


@dataclass(frozen=True)
class BufferClassification:
  inputs: list[Buffer]
  outputs: list[Buffer]
  intermediates: list[Buffer]

  @cached_property
  def buf_names(self):
    input_counter, output_counter, intermediate_counter = count(), count(), count()
    return (
      {buf: f"in{next(input_counter)}_{buf.size}" for buf in self.inputs}
      | {buf: f"out{next(output_counter)}_{buf.size}" for buf in self.outputs}
      | {buf: f"tmp{next(intermediate_counter)}_{buf.size}" for buf in self.intermediates}
    )


def classify_buffers(schedule: list[ScheduleItem], rendered_kernels: list[ProgramSpec]) -> BufferClassification:
  """Classify all the buffers used throughout the computation between inputs/outputs and intermediate buffers (that have to be heap allocated)."""
  ctx = {"ins": set(), "outs": set()}
  for si, rk in zip(schedule, rendered_kernels):
    ctx["ins"].update(set(si.bufs[i] for i in rk.ins))
    ctx["outs"].update(set(si.bufs[i] for i in rk.outs))

  return BufferClassification(
    inputs=list(ctx["ins"] - ctx["outs"]),
    outputs=list(ctx["outs"] - ctx["ins"]),
    intermediates=list(ctx["ins"] & ctx["outs"]),
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
  c_code.append("#include <stdint.h>")

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
    + ") {"
  )

  # allocate temporary buffers
  tmpcounter = count(0)
  tmpnames = {}
  for tmpbuf in bc.intermediates:
    assert tmpbuf.size > 0, "Intermediate buffer size must be greater than 0"
    tmpnames[tmpbuf] = f"tmp{next(tmpcounter)}_{tmpbuf.size}"
    dtype_str = base_renderer.render_dtype(tmpbuf.dtype)
    c_code.append(f"\t{dtype_str}* {tmpnames[tmpbuf]} = calloc({tmpbuf.size}, sizeof({dtype_str}));")

  # call all kernels
  for si, rk in zip(schedule, rendered_kernels):
    c_code.append(f"\t{rk.function_name}({', '.join(bc.buf_names[si.bufs[i]] for i in rk.globals)}, 0);")
    # NOTE: the following doesn't work because in some cases (see bicycle_disc_6 test case) si.bufs
    # contains buffers that are not passed to the kernel. WTF
    # c_code.append(f"\t{rk.function_name}({','.join(bc.buf_names[buf] for buf in si.bufs)}, 0);")

  # free temporary buffers
  for tmpbuf in bc.intermediates:
    c_code.append(f"\tfree({tmpnames[tmpbuf]});")

  c_code.append("}")

  c_code.append("#ifdef __cplusplus\n}\n#endif")

  return "\n".join(c_code)


class Codegenable(Protocol):
  def __call__(self, *args: Tensor) -> Tensor: ...


def aot(name: str, fn: Codegenable, inshape: OneOrMore[Shape]):
  """Full AOT codegen pipeline"""

  # Get the output tensor to construct the base AST
  ins = (Tensor.empty(inshape),) if isinstance(inshape[0], int) else tuple(Tensor.empty(shape) for shape in inshape)
  out = fn(*ins)

  # Scheduling (breakdown AST into kernels)
  schedule, var_vals = out.schedule_with_vars()

  # Render kernels
  rendered_kernels = render_kernels(schedule, var_vals)

  # Buffer classification
  bc = classify_buffers(schedule, rendered_kernels)

  return render_schedule(name, schedule, rendered_kernels, bc)
