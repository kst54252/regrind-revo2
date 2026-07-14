"""Hand-object interaction-mesh retargeting.

The retargeter depends on ``pydrake`` (and, for the default solver, a licensed 
Mosek). It is intentionally NOT imported by ``regrind/__init__.py``, so ``import 
regrind`` never requires pydrake.

The ``HandInteractionMeshOneStageRetargeter`` re-export below is guarded so that the
lightweight config modules (``leaphand_constants`` / ``wujihand_constants``) remain
importable even when pydrake is absent.
"""

try:
    from regrind.retargeting.retargeter import HandInteractionMeshOneStageRetargeter

    __all__ = ["HandInteractionMeshOneStageRetargeter"]
except (ImportError, ModuleNotFoundError):
    __all__ = []
