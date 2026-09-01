# RB3-730 + Revo2 + tuna can REGRIND PPO 환경

공개 REGRIND의 manager-based observation, reward, termination, RSI, RSL-RL PPO,
domain randomization과 gravity/push curriculum을 RB3-730 6축 + Revo2 leader
6축에 대응시킨 환경입니다. 새 PPO 구현이나 network architecture는 추가하지
않았습니다.

## Task 모드

| Task | 환경 수 | randomization/curriculum | 용도 |
|---|---:|---|---|
| `Regrind-RB3-Revo2-TunaCan-Play-v0` | 1 | OFF | deterministic policy/GUI 검증 |
| `Regrind-RB3-Revo2-TunaCan-Smoke-v0` | 16 | ON | 짧은 PPO 통합 검사 |
| `Regrind-RB3-Revo2-TunaCan-v0` | 4096 | ON | full training |

기존 `Regrind-RB3-Revo2-Tuna-*` 이름은 호환 alias로 유지합니다. Action은
RB3 6축 다음 Revo2 leader 6축의 12차원이며 다음 식을 유지합니다.

```text
q_target = q_ref + action_scale * residual
```

기본 residual scale은 arm `0.05 rad`, hand `0.15 rad`입니다. Revo2 distal
5축은 독립 action이 아니라 기존 mimic 비율로 계산합니다. Tuna can은 0.15 kg
rigid body이며 articulated-object joint observation/reward/randomization은 없습니다.

## PPO baseline

`agents/rsl_rl_ppo_cfg.py`는 LeapHand/WujiHand 공개 config와 같은 값을 사용합니다.

- rollout: 24 steps/env
- training: 20,000 iterations, checkpoint 500 iteration 간격
- actor/critic MLP: `[1024, 512, 256, 128]`, ELU, observation normalization
- actor output: REGRIND `ZeroInitMLPModel`, Gaussian initial std `0.5`
- PPO: 5 epochs, 4 mini-batches, learning rate `1e-3`, adaptive schedule
- `gamma=0.998`, `lambda=0.95`, clip `0.2`, entropy `0.002`
- desired KL `0.01`, max gradient norm `1.0`

Runner와 PPO update는 기존 `regrind/scripts/rsl_rl/train.py`와 RSL-RL
`OnPolicyRunner`를 그대로 사용합니다.

## Observation과 reward

Actor는 `(num_envs, 76)`, critic privileged observation은 `(num_envs, 109)`,
action은 `(num_envs, 12)`입니다. Actor에는 object pose, wrist/joint history,
previous action, phase와 reference joint target이 들어갑니다. Critic에는 실제
Revo2 touch-link 5개의 위치, joint velocity, object linear/angular velocity가
추가됩니다. MANO21 semantic point 전체를 fingertip observation으로 사용하지 않습니다.

Training actor observation에는 기존 REGRIND 함수의 noise와 0~2 timestep continuous
latency가 적용됩니다. Object pose, wrist pose, joint state별 delay buffer를 사용하고
critic은 undelayed privileged state를 받습니다. 공개 코드에 별도 action-latency
구현이 없어 새 action delay는 만들지 않았습니다.

Reward는 기존 함수를 그대로 사용합니다.

- tuna local surface point `(50,3)`의 world-space keypoint tracking
- object linear/angular velocity tracking
- wrist position/orientation tracking
- residual magnitude, action-rate, out-of-bounds 항목
- early termination penalty

Tuna can 회전 대칭 보정은 하지 않습니다. Success는 기존 REGRIND 기준처럼
early failure 없이 reference 마지막 frame에 도달한 episode이며 TensorBoard의
`Episode_Termination/success`로 기록됩니다.

## RSI, randomization과 curriculum

Training reset은 terminal frame을 제외한 reference frame을 균일하게 선택하고
RB3/Revo2 joint state, mimic follower state, tuna pose/velocity를 기록합니다. 작은
joint/object reset perturbation도 training에서 활성화되고 Play에서는 꺼집니다.

Randomization range는 `RandomizationRangesCfg`에서 조정합니다.

| 항목 | RB3 | Revo2 |
|---|---:|---:|
| body friction | 0.6–1.0 | 0.7–1.3 |
| body mass scale | 0.98–1.02 | 0.9–1.1 |
| actuator gain scale | 0.95–1.05 | 0.8–1.2 |

Tuna mass는 0.85–1.15배, friction은 0.5–1.2, CoM은 XY ±2 mm/Z ±1 mm이고
table friction은 0.6–1.2입니다. RB3/Revo2 actuator nominal stiffness, damping,
effort와 velocity limit은 각각 별도 asset actuator group에 있고,
`ActuatorBaselineCfg` multiplier로 조정할 수 있습니다.

Gravity와 random push는 공개 REGRIND stage/threshold를 그대로 사용합니다.
Gravity는 0에서 시작해 20,000 common step 이후 단계적으로 증가하여 최종
`-9.81 m/s²`에 도달합니다. Robot/object random velocity push는 130,000 step부터
시작합니다. 현재 gravity fraction은 `Metrics/reference/curriculum_level`로 기록됩니다.

## 실행

모든 RL 명령은 `scripts/rl.sh`를 사용합니다. 기본 sequence는
`20200709_143747_left`이며 `--sequence` 또는 `--reference`로 변경할 수 있습니다.

16-env, 2-iteration smoke test:

```bash
cd /home/wanjunkim/ARSL/regrind-upload
./scripts/rl.sh train \
  --sequence 20200709_143747_left \
  --num_envs 16 \
  --max_iterations 2 \
  --headless \
  --logger tensorboard \
  --run_name smoke_16env
```

Viewer를 보려면 `--headless`를 빼면 됩니다. Full 4096-env 학습:

```bash
./scripts/rl.sh train \
  --sequence 20200709_143747_left \
  --full \
  --num_envs 4096 \
  --headless \
  --logger tensorboard \
  --run_name full_4096
```

W&B를 사용할 때:

```bash
./scripts/rl.sh train \
  --sequence 20200709_143747_left \
  --full \
  --num_envs 4096 \
  --headless \
  --logger wandb \
  --log_project_name regrind-rb3-revo2
```

학습된 checkpoint를 randomization 없는 Play task에서 재생:

```bash
./scripts/rl.sh play \
  --sequence 20200709_143747_left \
  --load_run <run-folder> \
  --checkpoint model_500.pt
```

Zero-residual deterministic 검증:

```bash
./scripts/rl.sh zero \
  --sequence 20200709_143747_left \
  --gui \
  --skeleton \
  --real_time
```

`1000 iterations × 24 rollout steps`는 curriculum common step 약 24,000에
해당하므로 현재 schedule에서는 중력이 약 `0~-1 m/s²`인 초기 단계입니다. 학습
checkpoint와 같은 조건으로 재생하려면 play 명령에
`--auto_gravity_from_ckpt`를 추가합니다. Full gravity와 random-push 단계까지
학습하려면 curriculum threshold에 맞는 iteration 수가 필요합니다.

## 로그 확인

결과는 `logs/rsl_rl/rb3_revo2_tuna/<timestamp>_<run_name>/`에 생성됩니다.

```bash
/home/wanjunkim/IsaacLab/.venv/bin/tensorboard \
  --logdir logs/rsl_rl/rb3_revo2_tuna \
  --port 6006
```

다음 항목이 기록됩니다.

- `Train/mean_reward`, `Train/mean_episode_length`
- 모든 `Episode_Reward/*` component
- object keypoint/position/orientation 및 wrist error
- residual `action_rms`, action-rate
- `Episode_Termination/success`
- gravity와 `curriculum_level`
- `Loss/value`, `Loss/surrogate`, `Loss/entropy`, `Loss/learning_rate`

Smoke test 정상 기준은 16개 environment, actor 76/critic 109/action 12,
NaN 검사 통과, reward와 RSI reset 발생, `Learning iteration` 및 PPO loss 출력,
TensorBoard event와 `model_*.pt` 생성입니다.

아직 tactile sensor, symmetry-aware reward, 실제 로봇 deployment, 새 network/RL
알고리즘은 포함하지 않습니다.
