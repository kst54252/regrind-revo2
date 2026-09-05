# Floating Revo2 + tuna can REGRIND PPO

학습과 로봇팔 배치를 분리합니다. PPO는 RB3 없이 떠 있는 Revo2 손만 제어하고,
policy rollout이 완성된 뒤에만 RB3-730 strict IK를 실행합니다.

```text
reference wrist/object/Revo2 trajectory
  -> floating Revo2 physics + residual PPO
  -> physical floating rollout HDF5
  -> desired tuna-can start pose로 한 번의 rigid SE(3) alignment
  -> RB3-730 strict IK (previous-frame warm start + joint limits)
  -> RB3 6 + Revo2 6 = 12-DoF replay
```

현재 `20200709_143747_left`의 학습 기본 입력은
`outputs/isaac/dexycb/20200709_143747_left/rb3_revo2_reference_stable.h5`입니다.
성공한 floating rollout의 초기 PhysX 안정화 2 frame을 제거한 뒤, 첫 캔을 upright로
정렬하고 실제 tuna mesh의 최저점이 table-frame `Z=0`에 정확히 닿도록 전체
object/wrist/MANO trajectory에 하나의 rigid transform을 적용했습니다. 기존
`rb3_revo2_reference.h5`는 비교와 복구를 위해 그대로 보존됩니다.

동일 파일을 다시 생성하려면 다음 명령을 사용합니다.

```bash
./scripts/floating_to_rb3.sh \
  --rollout outputs/floating/random_can_replay/20200709_143747_left_random_rollout.h5 \
  --out outputs/isaac/dexycb/20200709_143747_left/rb3_revo2_reference_stable.h5 \
  --drop-leading-frames 2 \
  --level-object-on-table \
  --object-start 0.4 0.0 0.0
```

학습 reset의 joint noise는 유지하지만 object Z/rotation reset noise는 0입니다.
따라서 XY random placement는 mesh-table 접촉과 upright 자세를 바꾸지 않습니다.
RSI가 활성화된 학습에서는 중간 reference phase로 reset될 수 있으며, 이는 해당
phase의 object pose를 사용하는 기존 REGRIND 동작입니다. 첫 phase로만 확인하는
deterministic Play에서는 RSI와 모든 reset perturbation이 꺼집니다.

## Floating task

| Task | 환경 수 | randomization/curriculum | 용도 |
|---|---:|---|---|
| `Regrind-Floating-Revo2-TunaCan-Play-v0` | 1 | OFF | policy/GUI/rollout export |
| `Regrind-Floating-Revo2-TunaCan-Smoke-v0` | 16 | ON | 짧은 PPO 통합 검사 |
| `Regrind-Floating-Revo2-TunaCan-v0` | 4096 | ON | full training |

Action은 총 12차원입니다.

- `root_pose[0:6]`: floating wrist position/orientation residual
- `joint_pos[0:6]`: Revo2 leader joint residual
- Revo2 distal 5축: 독립 action이 아니며 기존 mimic 관계를 사용
- RB3 joint: 학습 환경과 policy observation/action에 존재하지 않음

Root action은 공개 LeapHand/WujiHand task의 `SE3ImpedanceActionTerm`을 재사용합니다.
위치/회전 scale은 각각 `1.0 * control_dt`, `3.2 * control_dt`, impedance gain은
position `300/30`, rotation `3/0.3`입니다. Finger action도 reference target에 대한
relative residual 방식을 유지합니다.

### IK 가능 영역 내 캔 위치 randomization

학습 환경은 episode reset마다 캔의 첫 XY를 `X=[0.40, 0.50] m`,
`Y=[-0.20, 0.20] m`에서 균등 샘플링합니다. 캔만 이동하는 것이 아니라
object/wrist reference 전체를 동일하게 평행 이동하므로 grasp 상대 자세가
보존됩니다. 이 보수적 책상 영역은 수직 Revo2 어댑터와 RB3 설치 높이
`Z=-0.02 m`에서 5x5 grid로 전체 trajectory를 검사해 strict full-pose IK
`25/25` 성공을 확인했습니다. 캔 yaw와 Z는 변경하지 않습니다.

학습 config에서는 기본 활성화되고 deterministic Play config에서는 꺼집니다.
평가에서 활성화하려면 `--random-placement`를 사용합니다.

Floating task의 위치 observation은 sampled placement offset을 제거한 canonical
object-trajectory frame으로 들어갑니다.

```text
p_obs = p_world - placement_offset
```

이 변환은 actor의 object/wrist/action-base position과 critic의 fingertip position에
동일하게 적용됩니다. 따라서 캔과 손 reference를 함께 평행 이동하면 기존 고정 위치와
policy 입력이 같아집니다. 물체가 실제로 reference에서 미끄러지거나 손목이 target에서
벗어난 상대 오차는 제거되지 않습니다. Observation shape `(67,)/(94,)`도 유지되므로
기존 floating-hand checkpoint를 그대로 사용할 수 있습니다. Simulator 상태와 rollout
HDF5는 canonical 좌표가 아니라 실제 table/world 좌표로 저장됩니다.

```bash
./scripts/rl.sh play --random-placement --checkpoint CHECKPOINT --num_envs 1
./scripts/rl.sh zero --random-placement --gui
```

기존 checkpoint의 위치 일반화 확인:

```bash
./scripts/rl.sh play \
  --sequence 20200709_143747_left \
  --random-placement \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/RUN/model_2999.pt \
  --rollout-path outputs/floating/random_can_rollout.h5
```

실제 RB3 관절 궤적은 sampled floating rollout마다 strict IK로 다시 생성합니다.

```bash
./scripts/rl.sh play \
  --random-placement \
  --checkpoint CHECKPOINT \
  --rollout-path outputs/floating/random_can_rollout.h5

./scripts/floating_to_rb3.sh \
  --rollout outputs/floating/random_can_rollout.h5 \
  --out outputs/floating/random_can_reference_12dof.h5
```

## Observation, reward, RSI

Actor observation은 `(num_envs, 67)`, critic privileged observation은
`(num_envs, 94)`, action은 `(num_envs, 12)`입니다.

- Actor: object pose, wrist pose history, Revo2 joint history, previous action,
  trajectory phase, action-base wrist/joint target
- Critic 추가 정보: object linear/angular velocity, 실제 Revo2 fingertip 5개 위치,
  Revo2 joint velocity
- Reward: tuna local surface point 50개의 world tracking, object linear/angular
  velocity, wrist position/orientation, residual magnitude/rate/out-of-bounds,
  early termination

RSI reset은 임의 reference frame을 고른 후 floating root wrist pose/velocity,
Revo2 leader와 mimic follower, tuna pose/velocity를 그 reference state 근처에
초기화합니다. Tuna can은 rigid object이며 articulated-object joint 항목은 없습니다.

공개 REGRIND의 observation delay/noise, mass/friction/actuator randomization,
gravity curriculum과 robot/object random push curriculum을 재사용합니다. RB3 전용
randomization 항목은 floating task에서 제거됩니다. Tuna rotational symmetry reward,
tactile sensor, 새 network나 새 RL 알고리즘은 추가하지 않았습니다.

## PPO

`config/revo2_floating/agents/rsl_rl_ppo_cfg.py`는 공개 LeapHand/WujiHand baseline과
동일한 RSL-RL PPO 값을 사용합니다.

- 24 rollout steps/env, 최대 20,000 iterations
- actor/critic `[1024, 512, 256, 128]`, ELU, observation normalization
- zero-initialized actor output, Gaussian initial std `0.5`
- PPO 5 epochs, 4 mini-batches, learning rate `1e-3`, adaptive schedule
- `gamma=0.998`, `lambda=0.95`, clip `0.2`, entropy `0.002`

## 실행

16-env smoke test:

```bash
./scripts/rl.sh train \
  --sequence 20200709_143747_left \
  --num_envs 16 \
  --max_iterations 2 \
  --headless \
  --logger tensorboard \
  --run_name floating_smoke
```

Full training:

```bash
./scripts/rl.sh train \
  --sequence 20200709_143747_left \
  --full \
  --num_envs 4096 \
  --max_iterations 1000 \
  --headless \
  --logger tensorboard \
  --run_name floating_full_1000
```

Zero residual로 floating reference 확인:

```bash
./scripts/rl.sh zero \
  --sequence 20200709_143747_left \
  --gui --real_time
```

Policy를 GUI로 보고, 동시에 environment 0의 실제 wrist/object/Revo2 상태를 저장합니다.
Episode가 중간에 실패하면 자동 reset 뒤의 상태를 이어 붙이지 않고 그 지점에서 저장을
중단합니다.

```bash
./scripts/rl.sh play \
  --sequence 20200709_143747_left \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/RUN/model_999.pt \
  --rollout-path outputs/floating/20200709_143747_left/rollout.h5 \
  --real_time
```

Headless export는 위 명령에 `--headless`를 추가합니다. Rollout에는
`wrist_pos/quaternion`, `revo2_joints`, `object_pos/quaternion`, policy action,
reference targets, phase와 MANO21이 기록됩니다. Quaternion order는 `xyzw`입니다.

### Random can → floating policy → RB3 IK → workcell GUI

다음 wrapper는 strict-IK 검증 영역에서 캔 XY를 새로 샘플링하고, 기존
floating-hand checkpoint를 headless로 실행한 뒤 RB3 IK를 풀고, 로봇 베이스와
책상이 포함된 Isaac Sim GUI를 엽니다. 기본은 캔을 trajectory로 순간 이동시키지
않고 중력과 로봇 접촉으로만 움직이는 physics-object mode입니다.

Floating rollout의 처음 2프레임은 측정된 약 11도 기울어진 캔이 중력으로 수직
안정화되는 reset settling 구간입니다. Physics replay는 이 2프레임을 제거하고 이미
안정된 frame 2를 새 시작점으로 사용합니다. 그 자세의 작은 수치 오차만 제거한 뒤
회전된 실제 mesh 최저점을 책상 상판 `Z=0`에 정확히 둡니다. 이 보정은 캔만 바꾸지 않고
object/wrist/MANO 전체에 하나의 rigid transform으로 적용한 다음 RB3 IK를 다시
풀기 때문에 손과 캔의 상대 자세는 유지됩니다.

```bash
./scripts/random_can_full_replay.sh
```

Physics replay는 floating-hand 학습과 같은 기본 120 Hz에서 trajectory의 저장
`dt`에 해당하는 고정 개수의
physics update만 진행합니다. Timeline 시간이 증가하지 않는 Isaac 구성에서도 Play가
frame 0에서 멈추지 않습니다. 마지막 target은 기본 0.5초 유지해 관절이 감속한 뒤
pause합니다. 유지 시간은 `--terminal-hold 1.0`, physics 주기는
`--physics-hz 120`처럼 변경할 수 있습니다. Revo2 drive gain과 tuna-can/table
contact offset도 floating 학습 환경과 동일한 값으로 transient replay Stage에 적용합니다.

기본 `--robot-control kinematic`에서는 RB3+Revo2 관절을 저장된 reference에 정확히
적용하면서 tuna can만 dynamic rigid body로 둡니다. 따라서 캔은 trajectory로
teleport되지 않고 로봇 collision, 중력, 마찰로만 움직이지만, articulation drive
추종 실패가 마지막에 한 번에 표시되는 현상은 없습니다. 30 Hz reference 관절은
120 Hz physics substep 네 개로 선형 보간되며, 매 state write마다 같은 drive target도
설정해 손가락이 stale target 쪽으로 되튕기는 현상을 막습니다. 실험용 완전 동역학
arm position control은 `--robot-control position`으로 선택할 수 있습니다.

다른 checkpoint나 느린 재생 속도를 지정할 수도 있습니다.

```bash
./scripts/random_can_full_replay.sh \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/RUN/model_2999.pt \
  --speed 0.5
```

캔도 reference pose를 그대로 따라가는 순수 기구학 검증은
`--kinematic-object`를 추가합니다. 두 모드 모두 원본 MANO21 skeleton을 함께
표시합니다.

현재 `20200709_143747_left`는 원래 학습 phase 60을 보존하면서 마지막 20개
불필요 프레임을 제거해 40프레임으로 종료됩니다. 따라서 기존 checkpoint의
처음 40프레임 phase 입력은 변경되지 않습니다.

마지막으로 검증한 random sample의 잘린 40-frame 결과는 다음 파일에도 저장되어
있습니다.

```text
outputs/floating/20200709_143747_left/model_2999_random_rollout.h5
outputs/floating/20200709_143747_left/model_2999_random_reference_12dof.h5
```

## 실제 물체 pose로 정렬하고 RB3 결합

`--object-start`는 tuna mesh origin의 RB3-world 좌표입니다. Orientation을 생략하면
floating rollout의 첫 orientation을 유지하며, 바꾸려면 `--object-quat X Y Z W`를
지정합니다. 이때 object만 옮기지 않고 wrist, object, MANO 전체에 동일한 rigid
transform을 적용합니다.

```bash
./scripts/floating_to_rb3.sh \
  --rollout outputs/floating/20200709_143747_left/rollout.h5 \
  --object-start 0.40 0.00 0.020469 \
  --out outputs/floating/20200709_143747_left/reference_12dof.h5
```

Strict IK 출력에서 다음 조건을 확인합니다.

- `IK success rate: 100%`
- `failed frame indices: []`
- `joint_limit_violation=False`, `finite_solution=True`
- position/orientation error가 설정 tolerance 이하

통합 GUI:

```bash
./tools/rb3_revo2_ik/run_replay_gui.sh \
  --trajectory outputs/floating/20200709_143747_left/reference_12dof.h5
```

## 로그

Floating log는 `logs/rsl_rl/floating_revo2_tuna/<timestamp>_<run_name>/`에 생성됩니다.

```bash
/home/wanjunkim/IsaacLab/.venv/bin/tensorboard \
  --logdir logs/rsl_rl/floating_revo2_tuna --port 6006
```

Smoke test 정상 기준은 16개 environment, actor 67/critic 94/action 12, finite reward,
RSI reset, `Learning iteration`과 PPO loss 출력, `model_*.pt` 생성입니다. 실제 확인한
1-iteration smoke test는 16 env, 384 total steps로 PPO update까지 완료했습니다.

기존 combined RB3+Revo2 RL task는 회귀 호환을 위해 유지합니다. 예전 동작이 필요하면
`scripts/rl.sh` 명령에 `--legacy-arm-rl`을 추가합니다.
