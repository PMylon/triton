from .._core import builtin, _unwrap_if_constexpr
from ._layouts import AMDMFMALayout, AMDWMMALayout
from . import cdna3, cdna4
from . import rdna3, rdna4
from . import gfx1250
from .warp_pipeline import warp_pipeline_stage


@builtin
def sched_barrier(mask, _semantic=None):
    """Emit an s_sched_barrier instruction to constrain the LLVM instruction scheduler.

    The mask is a bitmask controlling which instruction types may NOT cross the barrier:
        0    = block ALL instructions (strongest)
        8    = block MFMA/WMMA only
        256  = block DS reads only
        128  = block all DS ops
    Values can be combined via bitwise OR.
    """
    mask = _unwrap_if_constexpr(mask)
    _semantic.builder.create_sched_barrier(mask)


__all__ = [
    "AMDMFMALayout", "AMDWMMALayout", "cdna3", "cdna4", "rdna3", "rdna4", "gfx1250", "warp_pipeline_stage",
    "sched_barrier"
]
