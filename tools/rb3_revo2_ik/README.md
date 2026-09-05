# RB3-730 + Revo2 kinematics and Isaac replay

REGRIND가 만든 Revo2 wrist trajectory를 RB3-730의 bounded numerical IK로 풀어
12-DoF kinematic reference를 생성한다.

## 좌표계와 모델

- 입력 pose: `wrist_pos`와 `wrist_quat` (`xyzw`), REGRIND world 기준
- RB3 root: 기본값은 REGRIND world의 원점/identity
- IK target: 조립 USD의
  `/World/revo2_right/Geometry/world/right_hand_base_link`
- 실제 mount transform: RB3 `link6` 기준 translation `[0, 0, 0.141304972] m`,
  orientation identity (Revo2를 link6 축과 나란히 수직 장착)
- adapter asset: `USD/n_wc_v3_temp.stl`을 변환한
  `USD/revo2_vertical_adapter/revo2_vertical_adapter.usd`; link6 플랜지 끝면
  `Z=0.100 m`부터 시작하며 높이 41.304972 mm
- RB3 기본 `tcp`로 actual wrist frame을 근사하지 않는다.

관절 축, offset, limit, mount transform은 `USD/rb3_revo2_vertical.usda`의 composed
stage에서 추출해 `rb3_model.json`에 저장했다. 모듈 로딩 시 원본 USD SHA-256이
달라졌으면 경고한다.

## 실행

```bash
python3 tools/rb3_revo2_ik/build_reference_trajectory.py \
  outputs/trajectories/dexycb/20200928_144714/revo2_dexycb_retargeting.h5 \
  --out outputs/trajectories/dexycb/20200928_144714/rb3_revo2_reference.h5
```

RB3가 REGRIND world에서 다른 pose로 놓여 있다면 `--base-position X Y Z`와
`--base-quat-xyzw X Y Z W`를 함께 지정한다. 첫 프레임 initial guess는
`--initial-q`로 바꿀 수 있다.

REGRIND wrist frame과 실제 장착 Revo2 palm frame 사이에 고정 회전 보정이
필요하면 모든 프레임에 right-compose되는 옵션을 사용한다. 예를 들어 손바닥을
local Z 기준으로 180도 뒤집을 때:

```bash
python3 tools/rb3_revo2_ik/build_reference_trajectory.py WORLD_TRAJECTORY.h5 \
  --out REFERENCE.h5 \
  --input-quat-convention wxyz \
  --target-wrist-local-rpy-deg 0 0 180
```

이 보정은 첫 프레임 initial guess가 아니라 실제 IK target pose 전체에
적용되며, 출력의 `target_wrist_local_rpy_correction_deg`와
`target_wrist_local_quat_xyzw_correction`에 기록된다.

### Floating policy rollout을 실제 캔 위치에 배치

Floating-hand RL rollout은 로봇팔과 독립적인 좌표에서 생성된다. 아래 launcher는
첫 캔 pose를 원하는 RB3 world pose로 옮기는 하나의 rigid transform을 계산하고,
object/wrist/MANO 전체에 동일하게 적용한 뒤 strict IK를 수행한다.

```bash
./scripts/floating_to_rb3.sh \
  --rollout outputs/floating/20200709_143747_left/rollout.h5 \
  --object-start 0.40 0.00 0.020469 \
  --out outputs/floating/20200709_143747_left/reference_12dof.h5
```

초기 캔 orientation을 바꾸려면 `--object-quat X Y Z W`를 추가한다. 생략하면 rollout의
초기 orientation을 유지한다. 출력에는 적용한 rotation matrix와 translation이
`floating_alignment_rotation`, `floating_alignment_translation`로 기록된다.

Floating task의 `--random-placement`로 이미 캔 위치를 샘플링한 rollout은 다시
옮길 필요가 없다. 이 경우 `--object-start`를 생략하면 기록된 캔·손목 좌표를 그대로
strict IK에 사용한다.

```bash
./scripts/floating_to_rb3.sh \
  --rollout outputs/floating/random_can_rollout.h5 \
  --out outputs/floating/random_can_reference_12dof.h5
```

## 완전 신전 자세에서 천천히 접근

첫 strict-IK 자세로 바로 teleport하지 않고 RB3 완전 신전/손가락 open 상태에서
천천히 접근하는 구간을 앞에 붙인다.

```bash
/home/wanjunkim/IsaacLab/.venv/bin/python \
  tools/rb3_revo2_ik/prepend_ik_approach.py REFERENCE.h5 \
  --out REFERENCE_WITH_APPROACH.h5 \
  --duration 5.0
```

기본 시작 관절은 RB3/Revo2 모두 0 rad이다. 첫 자세는 strict IK 결과를
사용하고, 2-pi 주기 관절은 시작 자세에서 가장 가까운 동등 branch로 정리한다.
접근 구간은 endpoint velocity/acceleration이 0인 5차 minimum-jerk 관절 경로다.
물체와 MANO skeleton은 접근 중 첫 프레임 위치에 고정되고, 접근이 끝나면 원래
동작이 이어진다.

각 프레임은 이전 프레임의 유한한 bounded solution으로 warm start한다. 허용
오차 안의 해가 여러 개면 이전 joint configuration과 가장 가까운 해를 선택한다.
정확한 pose가 workspace 밖이면 `ik_success=False`로 기록하면서도, 관절 제한 안의
best-fit 해를 저장하므로 전체 sequence를 replay할 수 있다. 실패 프레임을 이전
프레임 값으로 복사하지 않는다.

## 출력 핵심 필드

- `rb3_joints`: `(T, 6)`
- `revo2_joints`: `(T, 6)`
- `reference_joints`: `(T, 12)`, RB3 다음 Revo2 순서
- `wrist_pos`, `wrist_quat`, `object_pos`, `object_quat`
- `fk_wrist_pos`, `fk_wrist_quat`
- `ik_success`, `optimizer_success`, `failed_frame_indices`
- `position_error_m`, `orientation_error_rad`
- `joint_limit_violation`, `finite_solution`, `solver_message`
- joint name/order, limits, base pose, tolerance, mount frame metadata

## Isaac Sim replay

Script Editor에 코드를 붙여 넣지 않고 프로젝트 루트의 터미널에서 실행한다.

```bash
./tools/rb3_revo2_ik/run_replay_gui.sh --list-sequences
./tools/rb3_revo2_ik/run_replay_gui.sh --sequence 20200709_143747_left
```

Isaac Sim GUI와 작은 control window가 열리며 Play/Pause/Reset, 한 프레임 전후 이동,
임의 프레임 이동, 검증 summary를 사용할 수 있다. 기본값은 loop 재생이다. 다른 속도나
일시 정지 상태로 열려면 각각 `--speed 0.5`, `--paused`를 추가한다.

Replay와 RB3+Revo2 RL 환경은
`config/workcell/rb3_revo2_table.json`의 실제 작업 셀 치수를 공유한다.

- 로봇 받침대: `0.50 x 0.50 x 0.70 m`
- 책상: `X 0.80 x Y 1.60 m`, 상면 높이 `0.72 m`
- 로봇은 받침대 상면 정중앙에 고정
- 책상의 가까운 X 모서리(`X=0.25 m`)와 받침대의 `+X` 모서리가 맞닿음
- trajectory 좌표는 책상 상면을 `Z=0`으로 유지하므로 공통 바닥은 `Z=-0.72 m`,
  RB3 설치면은 `Z=-0.02 m`

책상은 40 mm 상판과 네 다리, 받침대와 바닥은 모두 충돌 형상으로 생성된다.
RB3 설치 높이가 기존보다 20 mm 낮으므로 strict IK 결과의 `rb3_base_position`도
`[0, 0, -0.02]`이어야 한다. `build_reference_trajectory.py`는 이 값을 작업 셀
설정에서 자동으로 읽는다.

- 실제 RB3-730 + Revo2: 6 arm joints와 6 hand leader joints를 프레임별로 직접 적용
- 자홍색 점/선: retargeting 전의 DexYCB MANO21 skeleton
- tuna fish can mesh: dataset object pose trajectory
- 하늘색 선: wrist reference path
- 빨강/초록 점: Isaac FK wrist / reference wrist
- 주황색 점/선: object reference와 path

Revo2 follower joint도 USD mimic ratio로 함께 설정한다. 기본 실행은 순수 kinematic
teleport replay이며 physics contact와 position controller는 아래 옵션에서만 실행한다.
RL은 어느 모드에서도 실행하지 않는다.

### 중력/접촉으로 캔 움직이기

캔을 dataset pose로 프레임마다 옮기지 않고 첫 자세에 한 번만 생성한 뒤, 로봇과의
접촉 및 중력으로만 움직이려면 다음 모드를 사용한다.

```bash
./tools/rb3_revo2_ik/run_replay_gui.sh \
  --sequence 20200709_143747_left \
  --physics-object
```

이 모드에서는 RB3+Revo2 관절을 USD position drive로 제어하고 캔에 기본 0.15 kg
질량, cylinder collider, 마찰 0.8, 중력 9.81 m/s²를 적용한다. world Z=0의 실제
0.80 x 1.60 m 책상 상판 collider를 사용한다. 캔 reference line/marker는 기본으로 숨기며, 물체 mesh pose는
초기화/reset을 제외하고 절대 덮어쓰지 않는다. `Reset`은 로봇과 캔을 첫 자세로 되돌린다.

질량과 마찰은 필요하면 변경할 수 있다.

```bash
--object-mass 0.15 --object-friction 0.8
```

physics 모드는 기본적으로 한 번만 재생한다. 반복 실험은 `Reset` 후 `Play`를 누르거나
명시적으로 `--loop`를 추가한다. 현재 단계는 성공을 강제하지 않으며, 캔을 놓치거나
밀어내는 결과도 실제 접촉 결과로 그대로 표시한다.

## 테스트

```bash
./scripts/run_tests.sh
```

## IK 실패 원인 진단

기존 full-pose 결과를 변경하지 않고 random joint sampling workspace와
position-only IK를 비교한다.

```bash
MPLBACKEND=Agg /home/wanjunkim/IsaacLab/.venv/bin/python \
  tools/rb3_revo2_ik/diagnose_ik_failures.py \
  outputs/trajectories/dexycb/20200928_144714/dexycb_isaac_world.h5 \
  outputs/trajectories/dexycb/20200928_144714/rb3_revo2_reference_world_strict.h5 \
  --out-dir outputs/diagnostics/rb3_ik_world \
  --workspace-samples 100000
```

출력에는 workspace/trajectory 그림, 연속 실패 구간별 확대 그림,
`failed_frames.csv`, position-only 결과와 workspace samples를 담은
`rb3_ik_diagnostics.h5`가 포함된다. 기본 position tolerance는 `1e-4 m`,
joint-limit 근접 기준은 `0.05 rad`다.

인터랙티브 로봇 자세와 workspace viewer:

```bash
PYTHONPATH=tools/rb3_revo2_ik /home/wanjunkim/IsaacLab/.venv/bin/python \
  tools/rb3_revo2_ik/visualize_ik_diagnostic_interactive.py \
  outputs/trajectories/dexycb/20200928_144714/dexycb_isaac_world.h5 \
  outputs/trajectories/dexycb/20200928_144714/rb3_revo2_reference_world_strict.h5 \
  outputs/diagnostics/rb3_ik_world/rb3_ik_diagnostics.h5 \
  --out outputs/diagnostics/rb3_ik_world/rb3_ik_diagnostic_interactive.html
```

HTML은 Plotly JavaScript를 내장한 단일 파일이라 다른 폴더 없이 공유할 수 있다.

## Upright yaw 탐색

첫 물체 XY와 바닥 높이는 고정한 채, strict IK 성공률과 관절 연속성이 좋은
테이블 위 장면 방향만 탐색할 수 있다.

```bash
/home/wanjunkim/IsaacLab/.venv/bin/python \
  tools/rb3_revo2_ik/search_world_yaw.py RETARGETING.h5 \
  --mesh 007_tuna_fish_can/textured_simple.obj \
  --desired-x 0.4 --desired-y 0.0 \
  --target-wrist-local-rpy-deg 0 0 180
```
