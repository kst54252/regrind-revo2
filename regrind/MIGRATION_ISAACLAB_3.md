# REGRIND Isaac Lab 3.0 마이그레이션 요약

이 문서는 초기 포팅의 변경·검증 이력이다. 현재 명령은
[루트 README](../README.md), 데이터별 quaternion 규약은
[데이터 파이프라인](../docs/DATA_PIPELINE.md)을 따른다.

## 초기 포팅 대상 환경

- Isaac Lab develop 3.0
- Isaac Sim 6.0
- Python 3.12
- RSL-RL 5.4.1

이후 2026-09-07 실제 진단에서 확인한 배포 패키지는 Isaac Sim **6.0.1.0**,
`isaaclab` **13.3.0**, `isaaclab_physx` **3.1.1**이고 editable checkout의
`VERSION`은 **3.0.0**이었다. 서로 다른 버전 표기를 동일시하지 않는다.
정확한 설치 경로와 확인 방법은 [actuator 진단](../docs/ARM_ACTUATOR_DIAGNOSIS.md#installed-implementation-and-arm-configuration)에 보존되어 있다.
이 목록은 lockfile이나 모든 PC에 대한 호환성 보장이 아니다.

## 주요 변경 사항

### 1. Quaternion 규약 통일

- Isaac Lab 3.0 런타임 경계에서 quaternion을 `WXYZ`에서 `XYZW`로 변환했다.
  모든 저장 파일을 XYZW로 바꾼 것은 아니다. 전처리/world 파일은 WXYZ,
  retargeted/final reference는 XYZW이므로 파일 metadata를 우선한다.
- 초기 identity quaternion을 `(0, 0, 0, 1)`로 변경했다.
- retargeting 결과 저장 시 `WXYZ -> XYZW` 경계 변환을 수행하고 HDF5에 `quat_convention=xyzw`를 기록한다.
- 기존 `WXYZ` trajectory도 로드 시 자동 변환되도록 하여 기존 데이터 호환성을 유지했다.
- quaternion 보간, Euler 변환, 회전 거리 계산도 모두 `XYZW` 기준으로 수정했다.

### 2. Warp-first 데이터 API 대응

- articulation, rigid object, contact sensor 데이터 접근에 `.torch` view를 명시했다.
- 예: `joint_pos.torch`, `root_pos_w.torch`, `net_forces_w.torch`, `force_matrix_w.torch`.
- `Articulation` 전용 타입이 필요하지 않은 event 코드는 `BaseArticulation`을 사용하도록 변경했다.

### 3. Asset 상태 및 제어 API 변경

- 구형 root state 일괄 쓰기를 현재 API의 pose/velocity 쓰기로 분리했다.
  - `write_root_link_pose_to_sim_index()`
  - `write_root_link_velocity_to_sim_index()`
- joint target은 `set_joint_position_target_index()`를 사용하도록 변경했다.
- root 속도는 혼합된 `root_state_w` slicing 대신 명시적인 `root_link_vel_w`를 사용한다.
- 관련 replay, command reset, action 코드를 동일한 방식으로 맞췄다.

### 4. PhysX event 및 wrench 적용

- 기존 task의 event/wrench 경로를 현재 Isaac Lab asset API로 포팅했다.
  이후 추가한 진단 코드는 검증된 PhysX tensor API로 native drive/접촉값을
  읽기도 한다. 저장소 전체에서 tensor API 사용을 금지하거나 제거했다는 뜻은 아니다.
- gravity curriculum을 현재의 stateful `ManagerTermBase` 및 `randomize_physics_scene_gravity` API로 변경했다.
- 기존 Cartesian PD 식, gain, target 계산과 root force 적용은 유지했다.
- Isaac Sim 6에서 저관성 dummy root에 회전 토크를 직접 적용하면 손이 폭주하는 문제 때문에, 회전 토크만 body mass 비율로 분배한다. 전체 토크 합은 기존 `tau`와 같다.

### 5. 설정 및 import 경로 갱신

- `isaacsim.core` 및 구형 utility import를 현재 `isaaclab.sim` 경로로 변경했다.
- mutable config 기본값은 필요한 곳에서 `default_factory` 형태로 변경했다.
- Leap/Wuji 및 object 초기 pose 설정을 `XYZW` 규약에 맞췄다.

### 6. RSL-RL 5.x 학습 코드 포팅

- 구형 단일 `policy`/`CustomActorCritic` 구조를 별도의 `actor`와 `critic` 모델 설정으로 변경했다.
- actor/critic은 현재 `MLPModel`, `RslRlMLPModelCfg`를 사용한다.
- actor는 `GaussianDistributionCfg(init_std=0.5)`를 사용한다.
- 초기 residual policy mean이 정확히 0이 되도록 `MLPModel`을 얇게 상속하고 actor 마지막 linear layer만 zero initialization한다.
- actor 관측은 `policy`, critic 관측은 `critic` group으로 명시했다.
- `on_policy_runner` monkey patch와 `rsl_rl.networks` import를 제거했다.
- train/play에서 `handle_deprecated_rsl_rl_cfg()`를 적용하고, play export는 runner의 현재 JIT/ONNX export 메서드를 사용한다.
- 현재 AppLauncher에서도 기존 `--headless` 실행 명령이 동작하도록 호환 처리했다.

## 초기 포팅 당시 검증 결과

- quaternion 및 zero-initialized actor 단위 검증 통과
- `git diff --check` 및 Python 문법 검사 통과
- 64개 environment로 PPO 학습 iteration 15, 총 24,576 step까지 정상 진행 확인
- 생성된 checkpoint의 play load, inference, JIT/ONNX export 확인

## 참고

- REGRIND의 reward, observation, action scale, PPO hyperparameter 등 원본 학습 로직은 가능한 그대로 유지했다.
- 학습 중 출력되는 inotify watch 부족 및 scissors 내부 prim 탐색 경고는 남아 있지만 PPO 시작과 업데이트를 막지는 않는다.
