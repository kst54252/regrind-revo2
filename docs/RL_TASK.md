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
