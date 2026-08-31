# REGRIND Isaac Lab 3.0 마이그레이션 요약

## 대상 환경

- Isaac Lab develop 3.0
- Isaac Sim 6.0
- Python 3.12
- RSL-RL 5.4.1

## 주요 변경 사항

### 1. Quaternion 규약 통일

- Isaac Lab 3.0 런타임 규약에 맞춰 quaternion을 `WXYZ`에서 `XYZW`로 통일했다.
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

- 제거된 `omni.physics.tensors` 직접 사용을 없애고 현재 Isaac Lab asset/event API를 사용한다.
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

## 검증 결과

- quaternion 및 zero-initialized actor 단위 검증 통과
- `git diff --check` 및 Python 문법 검사 통과
- 64개 environment로 PPO 학습 iteration 15, 총 24,576 step까지 정상 진행 확인
- 생성된 checkpoint의 play load, inference, JIT/ONNX export 확인

## 참고

- REGRIND의 reward, observation, action scale, PPO hyperparameter 등 원본 학습 로직은 가능한 그대로 유지했다.
- 학습 중 출력되는 inotify watch 부족 및 scissors 내부 prim 탐색 경고는 남아 있지만 PPO 시작과 업데이트를 막지는 않는다.
