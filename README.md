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
  -> deployment에서는 매 step 실제 object/hand state로 policy 추론
  -> wrist residual을 bounded RB3 strict IK로 변환
  -> RB3 6축 + Revo2 6축 폐루프 physics 실행
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
현재 포팅 환경과 원본 REGRIND 설치 환경은 다릅니다.
[환경 이력](regrind/MIGRATION_ISAACLAB_3.md)을 먼저 확인하세요. Python 경로만
바꾸면 Isaac 의존성이 설치되는 것은 아닙니다. 다른 PC에서는 데이터셋·체크포인트·
필요한 reference 및 로컬 진단 입력을 별도로 준비해야 합니다.

### 어떤 경로를 실행할 것인가

| 용도 | 대표 경로 / 필수 입력 → 출력 |
|---|---|
| 전처리 → retargeting → RB3 reference | `scripts/run_pipeline.sh`: `dataset/` → `outputs/{preprocessed,retargeted,isaac}/dexycb/`; 세부 단계는 [데이터 문서](docs/DATA_PIPELINE.md) |
| 독립 Revo2 FK / RB3 IK | [Revo2 FK](tools/revo2_kinematics/README.md), [RB3 IK](tools/rb3_revo2_ik/README.md): 모델·keypoints·world trajectory → FK / 12-DoF reference |
| Floating 학습 / 평가 | `scripts/rl.sh train` / `play`: reference / checkpoint → `logs/rsl_rl/floating_revo2_tuna/` / 선택한 rollout HDF5 |
| 기존 arm baseline 정책 | `scripts/rl.sh play-arm`: floating checkpoint + reference → 실제 mounted 상태 기반 정책·IK 실행 |
| 궤적만 시각화 | `scripts/run_isaac_replay.sh --trajectory FILE.h5`: reference → Isaac viewer; policy 평가와 구분 |
| 고정 초기 상태 비교 | `bash scripts/evaluate_mounted_interface.sh --mode legacy --checkpoint FILE.pt --states BANK.jsonl --output NEW_DIR --episodes 20 --headless`: mode는 floating/legacy/simple 중 선택; [실행 계약/비교](docs/MINIMAL_MOUNTED_INTERFACE.md) |
| 검증 중인 arm 후보 GUI | `bash scripts/play_arm_candidate.sh NEW_OUTPUT_DIR`: [고정 candidate 및 속도 한계](docs/ARM_REALTIME_EXECUTION.md); 기존 baseline을 대체하지 않음 |

전체 pipeline은 생성물을 다시 작성하므로 단순 실행 확인에 사용하지 마세요.
`--sequence` 필터가 전처리 전체를 제한하지 않는 점은
[현재 상태](docs/current-status.md)에 설명되어 있습니다. 비교 실행에는 매번 새 출력
디렉터리를 사용하세요. 핵심 진단은 [실행/진단 색인](scripts/README.md)에서 선택합니다.

발표용 스켈레톤/Isaac 영상, 작업 셀 사진, 병렬 학습 영상 및 비교 영상은
[미디어 실행 안내](scripts/README.md#presentation-media)를 참고하세요.

## 디렉터리

| 경로 | 역할 | Git 관리 |
|---|---|---|
| `dataset/` | 원본 DexYCB 데이터, 읽기 전용 | 제외 |
| `regrind/` | 기반 REGRIND 패키지·스크립트 | 현재 root repository에 포함 |
| `tools/dexycb_batch/` | 전체 파이프라인 orchestration | 포함 |
| `tools/dexycb_world_transform/` | camera-to-world 변환 | 포함 |
| `tools/revo2_kinematics/` | Revo2 FK와 semantic keypoint | 포함 |
| `tools/rb3_revo2_ik/` | RB3 FK/IK, reference 생성, Isaac 실행·런타임 계측 | 포함 |
| `tools/arm_diagnostics/` | 저장된 팔 추종·접촉·정책 비교 결과 분석 (시뮬레이터 실행 아님) | 포함 |
| `scripts/` | 사람이 사용하는 대표 실행 명령 | 포함 |
| `tests/` | 주 simulator-independent 회귀 테스트 | 포함 |
| `docs/` | 구조, 데이터, 실행 설명 | 포함 |
| `USD/` | RB3/Revo2 USD와 Stage | 포함 |
| `007_tuna_fish_can/` | YCB tuna can asset | 필요한 경량 asset만 포함 |
| `outputs/` | 전처리·리타게팅·IK 결과는 일부 추적, 대형 HTML은 제외 | 혼합 |

자세한 파일 관계는 [프로젝트 구조](docs/PROJECT_STRUCTURE.md), 좌표계와 데이터
형식은 [데이터 파이프라인](docs/DATA_PIPELINE.md), Isaac 실행은
[Isaac Sim 리플레이](docs/ISAAC_SIM_REPLAY.md)를 참고하세요.

## 현재 검증 범위

- Revo2 FK: 입력 `(6,)`, 출력 semantic keypoints `(21, 3)`
- RB3 strict IK: joint limits와 이전 프레임 warm start 적용
- 과거 전처리 검증: 당시 준비된 5개 sequence의 strict IK 332/332 frames 성공
  (이후 trim/reference 수정본의 현재 프레임 수를 의미하지 않음)
- Isaac replay: RB3/Revo2 관절, 원본 MANO21, tuna can mesh 표시
- `--physics-object`: 캔 pose를 매 프레임 덮어쓰지 않고 중력/contact 사용

물리 grasp 성공 자체는 아직 보장하지 않습니다. tactile sensor, symmetry-aware
reward와 실제 로봇 deployment는 현재 범위 밖입니다.

## Floating Revo2 REGRIND RL

학습 환경에는 RB3를 넣지 않습니다. 공개 REGRIND와 같은 방식으로 floating wrist의
SE(3) residual 6차원과 Revo2 leader joint residual 6차원을 합친 12차원 action을
사용합니다. 학습된 floating trajectory는 실행 위치의 tuna can pose에 rigid alignment한
후 RB3 strict IK로 12-DoF robot reference로 변환합니다.
이는 **저장된 rollout의 오프라인 변환**입니다. `play-arm`은 저장 행동을 재생하지
않고 실제 mounted 상태에서 매번 정책을 추론합니다. 원래 baseline과 선택형
120 Hz 후보의 제어 주기·근사 IK 허용 여부는 [RL 실행 문서](docs/RL_TASK.md#online-rb3-deployment)에서 구분합니다.

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

# 같은 floating checkpoint를 RB3+Revo2+책상에서 온라인 폐루프 실행
# 매 episode마다 캔 XY는 IK 안전 영역에서 다시 샘플링됨
./scripts/rl.sh play-arm --sequence 20200709_143747_left \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/RUN/model_4999.pt \
  --num_envs 1 --real_time

# Actual can start pose alignment + RB3 strict IK
./scripts/floating_to_rb3.sh \
  --rollout outputs/floating/20200709_143747_left/rollout.h5 \
  --object-start 0.40 0.00 0.020469 \
  --out outputs/floating/20200709_143747_left/reference_12dof.h5

# Combined RB3+Revo2 GUI
./scripts/run_isaac_replay.sh \
  --trajectory outputs/floating/20200709_143747_left/reference_12dof.h5
```

이전 RB3+Revo2 동시 residual task는 삭제하지 않았으며 필요한 경우 RL 명령에
`--legacy-arm-rl`을 붙여 사용할 수 있습니다.

중복된 이전 실행 이름은 제거했습니다. 학습·평가·제로 에이전트·디버그는
`scripts/rl.sh train|play|zero|debug`로 실행하세요. GUI는 `zero --gui`,
스켈레톤 표시는 `zero --gui --skeleton`입니다.
이전 이름과 새 이름의 대응은 [정리 기록](docs/cleanup-plan.md#readable-layout-cleanup)에 있습니다.

자세한 범위와 검증 기준은 [RL task 문서](docs/RL_TASK.md)를 참고하세요.
