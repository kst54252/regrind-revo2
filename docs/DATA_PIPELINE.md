# Data pipeline and coordinate frames

[Repository map](architecture.md) · [Current status](current-status.md)

## 단계별 출력

```text
dataset/<sequence>/
  -> outputs/preprocessed/dexycb/<sequence>/dexycb_right_hand_preprocessed.npz
  -> outputs/retargeted/dexycb/<sequence>/revo2_retargeted.h5
  -> outputs/isaac/dexycb/<sequence>/world_trajectory.h5
  -> outputs/isaac/dexycb/<sequence>/rb3_revo2_reference.h5
```

`outputs/visualizations/dexycb/<sequence>/`에는 전처리 및 리타게팅 결과를 확인하는
HTML이 생성됩니다. HTML page는 ignore되지만 일부 manifest는 추적될 수 있으므로
visualization 디렉터리 전체를 일괄 삭제하지 않습니다.

## 핵심 배열

| 배열 | shape | 설명 |
|---|---:|---|
| `mano_joint_coords` | `(T, 21, 3)` | 원본/오른손 변환 MANO points |
| `revo2_joints` | `(T, 6)` | Revo2 관절 궤적 |
| `robot_keypoints` | `(T, 21, 3)` | Revo2 FK semantic points |
| `wrist_pos`, `wrist_quat` | `(T,3)`, `(T,4)` | wrist SE(3) |
| `rb3_joints` | `(T, 6)` | RB3 strict-IK 결과 |
| `reference_joints` | `(T, 12)` | RB3 다음 Revo2 순서 |

Quaternion order는 파일 메타데이터에 저장됩니다. Retargeted output은 `xyzw`,
world trajectory는 `wxyz`, final RB3+Revo2 reference는 `xyzw`입니다. 로더는
메타데이터를 읽고 필요한 API convention으로 변환합니다.

## 좌표계

DexYCB 두 번째 optical camera는 `+X=image right`, `+Y=image down`,
`+Z=forward`입니다. world 변환은 물체 local axis가 아니라 고정 camera gravity
axis를 사용합니다.

```text
camera -Y = Isaac world +Z
```

첫 캔 mesh의 최저점은 world `Z=0`에 놓이며 object origin의 기본 XY는 현재
`(0.4, 0.0)`입니다. sequence별 yaw는 `prepare_isaac_references.py` 한 곳에서
관리합니다. 손, wrist, 물체에는 같은 `T_world_camera`가 적용됩니다.

세부 변환식과 단일 파일 CLI는
[`tools/dexycb_world_transform/README.md`](../tools/dexycb_world_transform/README.md)를
참고하세요.
