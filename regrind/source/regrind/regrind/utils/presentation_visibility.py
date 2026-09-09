"""Transient, visual-only overrides for Revo2 presentation captures (no asset writes)."""
import re


def hide_revo2_keypoints(stage, root_path=None):
    """Hide marker roots below Robot, or an explicitly scoped presentation model."""
    from pxr import Sdf, Usd, UsdGeom

    if root_path is not None:
        root_path = Sdf.Path(root_path)
        if not root_path.IsAbsolutePath() or root_path == Sdf.Path.absoluteRootPath:
            raise ValueError('Expected a specific absolute model prim path')
        if not stage.GetPrimAtPath(root_path).IsValid():
            raise ValueError(f'Model prim not found: {root_path}')

    paths = []
    # Author only session-layer visibility. Never save an asset or deactivate a prim.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            if root_path is None:
                if 'Robot' not in str(prim.GetPath()).split('/'):
                    continue
            elif not prim.GetPath().HasPrefix(root_path):
                continue
            if not re.fullmatch(r'kp_(?:0\d|1\d|20)_.+', prim.GetName()):
                continue
            if prim.GetParent().GetName() == prim.GetName():
                continue  # nested same-name mesh inherits its marker root's visibility
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                paths.append(str(prim.GetPath()))
    return paths
