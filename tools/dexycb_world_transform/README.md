# DexYCB-to-Isaac world conversion

하나의 rigid camera-to-world transform을 계산하고 같은 `T_world_camera`를 전체
object, wrist, MANO trajectory에 적용한다. 원본 파일은 읽기만 하며 출력은 새
HDF5 또는 NPZ로 저장한다.

단독 CLI의 기본값은 `--camera-frame-convention object_upright`이며 아래와
같다. **배치 파이프라인의 기본값과 다르다.** Batch는 `dexycb_y_down`,
XY=(0.40, 0.00) 및 sequence별 yaw를 명시한다
([파이프라인 좌표계](../../docs/DATA_PIPELINE.md)).

```text
first object origin = [0.50, 0.00, -mesh_z_min]
first object rotation = identity
table plane = world Z = 0
```

계산식은 다음과 같다.

```text
T_world_camera = T_world_object_desired @ inverse(T_camera_object0)
T_world_object[t] = T_world_camera @ T_camera_object[t]
T_world_wrist[t]  = T_world_camera @ T_camera_wrist[t]
p_world           = R_world_camera @ p_camera + t_world_camera
```

여러 DexYCB 동작처럼 첫 물체 자세가 서로 다른 경우에는 물체 local Z로 중력축을
결정하면 안 된다. 이 프로젝트의 batch 변환은 두 번째 optical camera의 고정 축을
사용한다.

```text
camera +X = image right
camera +Y = image down
camera +Z = forward
camera -Y = Isaac world +Z
```

`--camera-frame-convention dexycb_y_down`을 사용하면 캔이 눕거나 서 있어도 이 축은
변하지 않는다. `--world-yaw-deg`는 중력축을 보존한 채 테이블 평면만 회전한다.
첫 물체의 XY origin을 요청 위치에 두고, 실제 첫 mesh 자세의 최저점이 world Z=0에
닿도록 translation을 계산한다.

## 변환

현재 workspace에서는 IsaacLab 가상환경에 `trimesh`, `h5py`, `scipy`가 모두
설치되어 있다.

```bash
/home/wanjunkim/IsaacLab/.venv/bin/python \
  tools/dexycb_world_transform/transform_trajectory.py \
  outputs/retargeted/dexycb/20200709_143747_left/revo2_retargeted.h5 \
  --mesh 007_tuna_fish_can/textured_simple.obj \
  --camera-frame-convention dexycb_y_down \
  --desired-x 0.4 --desired-y 0.0 --world-yaw-deg 150 \
  --out outputs/isaac/dexycb/20200709_143747_left/world_preview_NEW.h5
```

위 예시는 현재 `20200709_143747_left`의 배치 설정이다. 다른 sequence는
`prepare_isaac_references.py::ISAAC_ALIGNMENT`를 확인한다. 입력 원본은 읽기만
하지만 출력 경로는 덮어쓸 수 있으므로 기존 reference 대신 새 이름을 사용한다.

입력 quaternion은 `--input-quat-order auto|wxyz|xyzw`로 지정한다. `auto`는
파일의 `quat_convention` 또는 `quaternion_order`를 사용하며, 메타데이터가 없으면
`wxyz`이다. 출력 quaternion은 항상 `wxyz`다.

단일 물체를 강제로 upright 정렬하는 기존 모드에서는 local +Z를 유지하는
quaternion을 지정할 수 있다.

```bash
--desired-object-quat-wxyz W X Y Z
```

annotation pose frame과 OBJ mesh frame이 실제로 다른 asset에만 고정 model-frame
calibration을 right-compose할 수 있다.

```bash
--object-model-frame-rpy-deg ROLL PITCH YAW
```

이 옵션을 손/물체 궤적의 상하 방향을 바꾸는 용도로 사용하면 안 된다. 중력 방향은
`dexycb_y_down` 모드의 고정 camera 축으로 결정한다. `object_upright`는
별도 모드로, 첫 물체 자세를 강제로 정렬한다.

## 한 프레임 시각화

```bash
/home/wanjunkim/IsaacLab/.venv/bin/python \
  tools/dexycb_world_transform/visualize_world_frame.py \
  outputs/isaac/dexycb/20200709_143747_left/world_preview_NEW.h5 \
  --mesh 007_tuna_fish_can/textured_simple.obj \
  --frame 0
```

이미지 저장만 할 경우 `--save result.png --no-show`를 추가한다.

## 출력

```text
object_pos_world    (T,3)
object_quat_world   (T,4)       wxyz
wrist_pos_world     (T,3)
wrist_quat_world    (T,4)       wxyz
mano_joint_world    (T,21,3)
T_world_camera      (4,4)
```

후속 처리에 쓸 수 있도록 입력에 존재하는 `revo2_joints`,
`object_points_local`, `fps`, `frame_index`도 그대로 보존한다.
