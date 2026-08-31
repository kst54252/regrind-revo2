"""Export Revo2 keypoint Xforms from the currently open Isaac Sim stage.

Paste this file into Isaac Sim's Script Editor, or open it there and press Run.
"""

import json
import os
import re

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics


PROJECT_ROOT = os.environ.get(
    "REVO2_PROJECT_ROOT", "/home/wanjunkim/ARSL/regrind-revo2"
)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "tools", "revo2_kinematics", "revo2_keypoints.json"
)
EXPECTED_KEYPOINT_COUNT = 21
KEYPOINT_NUMBER_RE = re.compile(r"^kp_(\d+)(?:_|$)")


def _keypoint_number(prim):
    """Return the numeric part of a name such as kp_03_thumb_tip."""
    match = KEYPOINT_NUMBER_RE.match(prim.GetName())
    return int(match.group(1)) if match else None


def _find_parent_link(keypoint_prim):
    """Find the nearest rigid-body ancestor, falling back to the direct parent."""
    direct_parent = keypoint_prim.GetParent()
    ancestor = direct_parent

    while ancestor and ancestor.IsValid() and not ancestor.IsPseudoRoot():
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            return ancestor
        ancestor = ancestor.GetParent()

    return direct_parent


def export_revo2_keypoints(output_path=OUTPUT_PATH):
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("열려 있는 USD Stage가 없습니다.")

    keypoint_prims = [
        prim
        for prim in stage.Traverse()
        if prim.GetName().startswith("kp_") and prim.IsA(UsdGeom.Xform)
    ]

    malformed = [prim.GetPath().pathString for prim in keypoint_prims if _keypoint_number(prim) is None]
    if malformed:
        print("[WARNING] 이름에서 keypoint 번호를 읽을 수 없어 제외합니다:")
        for path in malformed:
            print(f"  - {path}")

    keypoint_prims = [prim for prim in keypoint_prims if _keypoint_number(prim) is not None]
    keypoint_prims.sort(key=lambda prim: (_keypoint_number(prim), prim.GetPath().pathString))

    numbers = [_keypoint_number(prim) for prim in keypoint_prims]
    duplicate_numbers = sorted({number for number in numbers if numbers.count(number) > 1})
    if duplicate_numbers:
        raise RuntimeError(f"중복된 keypoint 번호가 있습니다: {duplicate_numbers}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    result = {}

    for prim in keypoint_prims:
        parent_link = _find_parent_link(prim)
        if not parent_link or not parent_link.IsValid() or parent_link.IsPseudoRoot():
            print(f"[WARNING] parent link를 찾지 못해 제외합니다: {prim.GetPath()}")
            continue

        # This is keypoint-to-link, not keypoint-to-world. It also remains correct
        # if one or more intermediate Xforms exist below the link.
        relative_matrix, _ = xform_cache.ComputeRelativeTransform(
            prim,
            parent_link,
        )
        translation = relative_matrix.ExtractTranslation()
        number = _keypoint_number(prim)

        result[str(number)] = {
            "name": prim.GetName(),
            "prim_path": prim.GetPath().pathString,
            "parent_link": parent_link.GetName(),
            "xyz": [float(translation[0]), float(translation[1]), float(translation[2])],
        }

    output_path = os.path.abspath(output_path)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")

    count = len(result)
    print(f"[Revo2 keypoint export] 총 keypoint 수: {count}")
    if count != EXPECTED_KEYPOINT_COUNT:
        print(
            f"[WARNING] keypoint가 {EXPECTED_KEYPOINT_COUNT}개여야 하지만 "
            f"{count}개를 찾았습니다."
        )
    print(f"[Revo2 keypoint export] 저장 완료: {output_path}")
    return result


export_revo2_keypoints()
