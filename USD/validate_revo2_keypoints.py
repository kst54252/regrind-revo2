"""Validate Revo2 keypoint local coordinates in the currently open Isaac Sim stage.

Open this file in Isaac Sim's Script Editor and press Run. Change JSON_PATH if
the project is moved to another directory.
"""

import json
import math
import os
import re

import omni.usd
from pxr import Gf, Usd, UsdGeom


# Isaac Sim's Script Editor runs a temporary copy under /tmp and its working
# directory is commonly IsaacLab, so a path relative to cwd is not reliable.
PROJECT_ROOT = os.environ.get(
    "REVO2_PROJECT_ROOT", "/home/wanjunkim/ARSL/regrind-revo2"
)
JSON_PATH = os.path.join(
    PROJECT_ROOT, "tools", "revo2_kinematics", "revo2_keypoints.json"
)
TOLERANCE_M = 1.0e-5
EXPECTED_KEYPOINT_COUNT = 21
KEYPOINT_NUMBER_RE = re.compile(r"^kp_(\d+)(?:_|$)")


def _keypoint_number(item):
    """Return the number from a keypoint name such as kp_03_index_dip."""
    match = KEYPOINT_NUMBER_RE.match(str(item.get("name", "")))
    if match is None:
        raise ValueError(f"keypoint 이름 형식이 잘못되었습니다: {item.get('name')!r}")
    return int(match.group(1))


def _load_keypoints(json_path):
    with open(json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # Accept both the current {number: item} format and a plain item list.
    if isinstance(data, dict):
        items = list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("JSON 최상위 값은 object 또는 list여야 합니다.")

    required = {"name", "prim_path", "parent_link", "xyz"}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"keypoint 항목 {index}가 object가 아닙니다.")
        missing = required - set(item)
        if missing:
            raise ValueError(
                f"{item.get('name', index)!r}에 필수 필드가 없습니다: "
                f"{sorted(missing)}"
            )
        if not isinstance(item["xyz"], (list, tuple)) or len(item["xyz"]) != 3:
            raise ValueError(f"{item['name']!r}의 xyz는 길이 3의 배열이어야 합니다.")

    items.sort(key=lambda item: (_keypoint_number(item), item["name"]))
    return items


def _find_parent_link(keypoint_prim, parent_link_name):
    """Find the named parent link only within this keypoint's ancestor chain."""
    ancestor = keypoint_prim.GetParent()
    while ancestor and ancestor.IsValid() and not ancestor.IsPseudoRoot():
        if ancestor.GetName() == parent_link_name:
            return ancestor
        ancestor = ancestor.GetParent()
    return None


def _format_position(position_m):
    return "[" + ", ".join(f"{value:.9f}" for value in position_m) + "]"


def validate_revo2_keypoints(
    json_path=JSON_PATH,
    tolerance_m=TOLERANCE_M,
):
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("열려 있는 USD Stage가 없습니다.")

    json_path = os.path.abspath(json_path)
    items = _load_keypoints(json_path)
    keypoint_numbers = [_keypoint_number(item) for item in items]
    expected_numbers = list(range(EXPECTED_KEYPOINT_COUNT))

    # Build a fresh cache on every run so poses changed since the previous run
    # are never reused. Default time reads the pose currently authored on Stage.
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

    errors_m = []
    warning_names = []
    failed_names = []
    found_count = 0

    print("=" * 72)
    print("Revo2 keypoint validation")
    print(f"JSON: {json_path}")
    print(f"tolerance: {tolerance_m:.9g} m")
    print("=" * 72)

    for item in items:
        name = item["name"]
        prim = stage.GetPrimAtPath(item["prim_path"])

        if not prim or not prim.IsValid():
            print(f"\n{name}")
            print(f"[WARNING] Stage에서 prim을 찾지 못했습니다: {item['prim_path']}")
            failed_names.append(name)
            continue
        if prim.GetName() != name or not prim.IsA(UsdGeom.Xform):
            print(f"\n{name}")
            print(
                "[WARNING] prim_path의 prim이 예상한 kp_* Xform이 아닙니다: "
                f"{prim.GetPath()}"
            )
            failed_names.append(name)
            continue

        found_count += 1
        parent_link = _find_parent_link(prim, item["parent_link"])
        if parent_link is None:
            print(f"\n{name}")
            print(
                f"[WARNING] 조상에서 parent link {item['parent_link']!r}를 "
                "찾지 못했습니다."
            )
            failed_names.append(name)
            continue

        try:
            local_position = Gf.Vec3d(*(float(value) for value in item["xyz"]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name!r}의 xyz에 숫자가 아닌 값이 있습니다.") from error

        parent_world = xform_cache.GetLocalToWorldTransform(parent_link)
        expected_stage = parent_world.Transform(local_position)

        keypoint_world = xform_cache.GetLocalToWorldTransform(prim)
        actual_stage = keypoint_world.ExtractTranslation()

        delta_stage = expected_stage - actual_stage
        error_m = math.sqrt(sum(float(value) ** 2 for value in delta_stage)) * meters_per_unit
        expected_m = [float(value) * meters_per_unit for value in expected_stage]
        actual_m = [float(value) * meters_per_unit for value in actual_stage]
        errors_m.append(error_m)

        print(f"\n{name}")
        print(f"expected: {_format_position(expected_m)} m")
        print(f"actual:   {_format_position(actual_m)} m")
        print(f"error:    {error_m:.9f} m")

        if error_m > tolerance_m:
            warning_names.append(name)
            print(f"[WARNING] error가 tolerance {tolerance_m:.9g} m를 초과했습니다.")

    print("\n" + "=" * 72)
    print(f"Stage에서 발견한 keypoint: {found_count}/{EXPECTED_KEYPOINT_COUNT}")

    if errors_m:
        max_error_m = max(errors_m)
        mean_error_m = sum(errors_m) / len(errors_m)
        print(f"max error:  {max_error_m:.9f} m")
        print(f"mean error: {mean_error_m:.9f} m")
    else:
        max_error_m = float("inf")
        mean_error_m = float("inf")
        print("max error:  N/A")
        print("mean error: N/A")

    count_ok = (
        len(items) == EXPECTED_KEYPOINT_COUNT
        and found_count == EXPECTED_KEYPOINT_COUNT
    )
    numbering_ok = keypoint_numbers == expected_numbers
    all_compared = len(errors_m) == EXPECTED_KEYPOINT_COUNT
    passed = (
        count_ok
        and numbering_ok
        and all_compared
        and not warning_names
        and not failed_names
    )

    if len(items) != EXPECTED_KEYPOINT_COUNT:
        print(
            f"[WARNING] JSON keypoint 수가 {EXPECTED_KEYPOINT_COUNT}개가 아닙니다: "
            f"{len(items)}개"
        )
    if not numbering_ok:
        print(
            f"[WARNING] keypoint 번호가 0~{EXPECTED_KEYPOINT_COUNT - 1}의 "
            f"연속된 번호가 아닙니다: {keypoint_numbers}"
        )
    if failed_names:
        print(f"[WARNING] 비교하지 못한 keypoint: {', '.join(failed_names)}")
    if warning_names:
        print(f"[WARNING] tolerance 초과 keypoint: {', '.join(warning_names)}")

    if passed:
        print("[PASS] 21개 keypoint가 모두 tolerance 이내입니다.")
    else:
        print("[FAIL] keypoint 검증에 실패했습니다.")
    print("=" * 72)

    return passed


validate_revo2_keypoints()
