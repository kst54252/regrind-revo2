# REGRIND RB3-Revo2: DexYCB Retargeting, Replay, and Residual RL

DexYCB의 사람 손-물체 동작을 6-DoF Revo2 손으로 리타게팅하고, RB3-730의
strict IK를 거쳐 12-DoF 궤적을 Isaac Sim에서 재생하는 프로젝트입니다.

현재 파이프라인은 다음 범위를 다룹니다.

```text
DexYCB second camera
  -> 오른손 MANO21 전처리
  -> REGRIND interaction-mesh retargeting
  -> Revo2 6-DoF + semantic keypoints 21개
  -> Isaac/RB3 world 좌표 변환
  -> RB3-730 strict IK
  -> RB3 6축 + Revo2 6축 replay

RL/deployment branch
  -> RB3 없이 floating Revo2 wrist SE(3) + finger 6축 residual PPO
  -> policy rollout의 wrist/object/Revo2 trajectory 저장
  -> 실제 캔 시작 pose로 전체 rollout rigid alignment
  -> RB3 strict IK
  -> RB3 6축 + Revo2 6축 통합 replay
```

## 빠른 시작

프로젝트 루트에서 실행합니다.

```bash
# 전체 데이터 전처리, 리타게팅, world 변환, strict IK
./scripts/run_pipeline.sh

# 준비된 sequence 확인
./scripts/run_isaac_replay.sh --list-sequences

# 순수 kinematic replay
./scripts/run_isaac_replay.sh --sequence 20200709_143747_left

# 캔을 중력과 접촉으로만 움직이는 실험 모드
./scripts/run_isaac_replay.sh --sequence 20200709_143747_left --physics-object

# 회귀 테스트
./scripts/run_tests.sh
```

기본 Python 경로는 `/home/wanjunkim/IsaacLab/.venv/bin/python`입니다. 다른 환경은
`ISAAC_SIM_PYTHON=/path/to/python`으로 지정할 수 있습니다.

## 디렉터리

| 경로 | 역할 | Git 관리 |
|---|---|---|
| `dataset/` | 원본 DexYCB 데이터, 읽기 전용 | 제외 |
| `regrind/` | 기반 REGRIND 코드 | 별도 nested repository |
| `tools/dexycb_batch/` | 전체 파이프라인 orchestration | 포함 |
| `tools/dexycb_world_transform/` | camera-to-world 변환 | 포함 |
| `tools/revo2_kinematics/` | Revo2 FK와 semantic keypoint | 포함 |
| `tools/rb3_revo2_ik/` | RB3 IK, 진단, Isaac replay | 포함 |
| `scripts/` | 사람이 사용하는 대표 실행 명령 | 포함 |
| `tests/` | 모든 회귀 테스트 | 포함 |
| `docs/` | 구조, 데이터, 실행 설명 | 포함 |
| `USD/` | RB3/Revo2 USD와 Stage | 포함 |
| `007_tuna_fish_can/` | YCB tuna can asset | 필요한 경량 asset만 포함 |
| `outputs/` | 전처리·리타게팅·IK·시각화 결과 | 제외 |

자세한 파일 관계는 [프로젝트 구조](docs/PROJECT_STRUCTURE.md), 좌표계와 데이터
형식은 [데이터 파이프라인](docs/DATA_PIPELINE.md), Isaac 실행은
[Isaac Sim 리플레이](docs/ISAAC_SIM_REPLAY.md)를 참고하세요.

## 현재 검증 범위

- Revo2 FK: 입력 `(6,)`, 출력 semantic keypoints `(21, 3)`
- RB3 strict IK: joint limits와 이전 프레임 warm start 적용
- 준비된 5개 sequence: strict IK 332/332 frames 성공
- Isaac replay: RB3/Revo2 관절, 원본 MANO21, tuna can mesh 표시
- `--physics-object`: 캔 pose를 매 프레임 덮어쓰지 않고 중력/contact 사용

물리 grasp 성공 자체는 아직 보장하지 않습니다. tactile sensor, symmetry-aware
reward와 실제 로봇 deployment는 현재 범위 밖입니다.

## Floating Revo2 REGRIND RL

학습 환경에는 RB3를 넣지 않습니다. 공개 REGRIND와 같은 방식으로 floating wrist의
SE(3) residual 6차원과 Revo2 leader joint residual 6차원을 합친 12차원 action을
사용합니다. 학습된 floating trajectory는 실행 위치의 tuna can pose에 rigid alignment한
후 RB3 strict IK로 12-DoF robot reference로 변환합니다.

```bash
# Floating-hand zero-residual GUI
./scripts/rl.sh zero --sequence 20200709_143747_left --gui --real_time

# PPO smoke/full training
./scripts/rl.sh train --sequence 20200709_143747_left --num_envs 16 --max_iterations 2 --headless
./scripts/rl.sh train --sequence 20200709_143747_left --full --num_envs 4096 --headless

# Floating policy GUI + physical rollout HDF5 export
./scripts/rl.sh play --sequence 20200709_143747_left \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/RUN/model_999.pt \
  --rollout-path outputs/floating/20200709_143747_left/rollout.h5

# Actual can start pose alignment + RB3 strict IK
./scripts/floating_to_rb3.sh \
  --rollout outputs/floating/20200709_143747_left/rollout.h5 \
  --object-start 0.40 0.00 0.020469 \
  --out outputs/floating/20200709_143747_left/reference_12dof.h5

# Combined RB3+Revo2 GUI
./tools/rb3_revo2_ik/run_replay_gui.sh \
  --trajectory outputs/floating/20200709_143747_left/reference_12dof.h5
```

이전 RB3+Revo2 동시 residual task는 삭제하지 않았으며 필요한 경우 RL 명령에
`--legacy-arm-rl`을 붙여 사용할 수 있습니다.

`train_rb3_revo2_ppo.sh`, `play_rb3_revo2_ppo.sh`, `run_rl_zero_replay*.sh`는
기존 사용자를 위한 얇은 호환 wrapper이며 실제 launcher 로직은 `scripts/rl.sh`와
`scripts/_common.sh`에 한 번만 정의됩니다.

자세한 범위와 검증 기준은 [RL task 문서](docs/RL_TASK.md)를 참고하세요.
