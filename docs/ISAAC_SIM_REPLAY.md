# Isaac Sim replay

## 시작

```bash
./scripts/run_isaac_replay.sh --list-sequences
./scripts/run_isaac_replay.sh --sequence 20200709_143747_left
```

GUI에는 RB3-730, Revo2, tuna fish can mesh, retargeting 전 MANO21 skeleton,
wrist target/FK marker가 함께 표시됩니다. 작은 control window에서 Play, Pause,
Reset, 이전/다음 프레임, 특정 프레임 이동을 사용할 수 있습니다.

## 두 replay 모드

### Kinematic (기본)

각 프레임의 12개 joint position을 직접 적용하고 object는 recorded trajectory를
따라갑니다. 궤적과 FK 대응을 확인하는 모드이며 접촉 결과를 평가하지 않습니다.

```bash
./scripts/run_isaac_replay.sh --sequence 20200709_143747_left
```

### Dynamic object experiment

캔은 첫 pose에 한 번만 생성되고 이후 gravity, table, robot contact로 움직입니다.
`--robot-control kinematic`은 전체 관절 상태를 고정해 FK를 검사합니다.
`--robot-control position`은 팔과 손 모두 drive로 움직이며 추종 오차가 발생할 수 있습니다.
`--robot-control arm-kinematic`은 팔만 IK 상태에 고정하고 손가락은 drive로 움직이는
접촉 비교용 모드입니다. 기본값은 `kinematic`입니다.

## 실제 작업 셀

RB3 replay와 legacy arm RL 환경은 `config/workcell/rb3_revo2_table.json`을
공유합니다. 책상 상면을 world `Z=0`으로 두고, 72 cm 높이의 공통 바닥은
`Z=-0.72 m`입니다. 70 cm 받침대 상면과 RB3 설치점은 `Z=-0.02 m`이며,
받침대 크기는 50 x 50 cm입니다. 책상은 로봇의 `+X` 방향에 붙고 크기는
`X 0.8 x Y 1.6 m`입니다. 이 기준으로 생성한 reference에는
`rb3_base_position=[0,0,-0.02]`가 저장됩니다.

```bash
./scripts/run_isaac_replay.sh \
  --sequence 20200709_143747_left \
  --physics-object
```

이 모드는 현재 grasp 성공을 보장하지 않습니다. trajectory가 open-loop kinematic
reference이고 controller/contact tuning을 하지 않았기 때문입니다.

## Floating 학습 기록과 통합 재생 비교

같은 동작 자체를 확인할 때는 **물리 캔 옵션 없이** 저장된 손과 캔 상태를 재생합니다.
이때 캔의 움직임도 학습 기록을 표시하므로 새로 수행한 물리 파지 검증은 아닙니다.

```bash
./tools/rb3_revo2_ik/run_replay_gui.sh \
  --trajectory outputs/floating/20200709_143747_left/stable_model_2999_reference_12dof.h5 \
  --demo-skeleton --no-loop
```

같은 명령에 `--physics-object --robot-control arm-kinematic`을 추가하면 캔과
손가락을 다시 물리 시뮬레이션합니다. 이는 현재 상태를 입력받는 floating 정책의
실시간 추론과 다르므로, 학습 기록과 동일한 파지 결과를 보장하지 않습니다.

신규 rollout은 6개 leader 상태와 별개로 `revo2_follower_joints (T,5)`,
`revo2_joint_drive_target (T,6)`, `revo2_fingertip_pos (T,5,3)`을 기록합니다.
21점 MANO는 저장된 순서 메타데이터를 읽고 손가락 연결 순서로 변환합니다.
학습 시 SI 게인과 USD 도 단위 게인의 변환 및 force drive 설정을 적용하며,
물리는 렌더링 횟수와 무관하게 `1 / --physics-hz` 간격으로 진행합니다.

2026-09-05, `stable_model_2999`의 38프레임 검증 결과:

- strict IK 38/38 성공, 전체 관절 상태 재생 시 손목 최대 위치 오차 `3.20e-7 m`.
- 실제 학습 손끝 5개와 통합 손끝의 최대 위치 차이 `3.72e-7 m`.
- 게인/물리 설정 수정 후 NaN/Inf 없이 재생됨.
- 물리 캔 파지는 아직 실패: 기록의 마지막 캔 원점 Z는 약 `0.2302 m`,
  새 접촉 시뮬레이션에서는 약 `0.012636 m`에 남음.
- 따라서 IK/손 형상 일치 검증과 물리 파지 성공을 구분해야 함.

자동 검증은 위 명령에 `--headless --exit-after-replay`를 추가합니다.

## 주요 옵션

- `--speed 0.5`: 재생 속도
- `--paused`: 정지 상태로 시작
- `--start-frame N`: 시작 프레임
- `--dt SECONDS`: trajectory dt override
- `--no-demo-skeleton`: MANO21 숨기기
- `--object-mass KG`, `--object-friction VALUE`: dynamic can 물성
- `--loop` / `--no-loop`: 반복 재생 설정

구현 및 진단 명령 전체는
[`tools/rb3_revo2_ik/README.md`](../tools/rb3_revo2_ik/README.md)를 참고하세요.
