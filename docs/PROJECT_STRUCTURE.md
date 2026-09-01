# Project structure

이 문서는 “어디에서 무엇을 수정해야 하는가”를 빠르게 찾기 위한 지도입니다.

```text
regrind-upload/
├── README.md
├── scripts/                    # 사용자용 실행 진입점(rl.sh 포함)
├── tests/                      # 기능별 회귀 테스트
├── docs/                       # 프로젝트 수준 문서
├── tools/
│   ├── dexycb_batch/           # 여러 sequence를 잇는 orchestration
│   ├── dexycb_world_transform/ # camera -> Isaac world 변환
│   ├── revo2_kinematics/       # 6-DoF FK -> 21 semantic points
│   └── rb3_revo2_ik/           # RB3 IK, 진단, Isaac replay
├── regrind/                    # upstream 기반 리타게팅 코드
├── USD/                        # robot assets와 assembled Stage
├── 007_tuna_fish_can/          # YCB object asset
├── dataset/                    # 원본 데이터(수정 금지)
└── outputs/                    # 언제든 재생성 가능한 결과
```

## 코드 책임

- `tools/dexycb_batch/preprocess_dataset.py`: 두 번째 카메라 데이터 추출, MANO21
  순서 검증, 왼손 capture의 오른손 변환
- `tools/dexycb_batch/retarget_all.py`: REGRIND를 sequence별로 호출하고 HTML 생성
- `tools/dexycb_batch/prepare_isaac_references.py`: world 변환과 strict IK 연결
- `tools/dexycb_world_transform/transform_trajectory.py`: 하나의 rigid
  `T_world_camera`를 object, wrist, MANO에 동일 적용
- `tools/revo2_kinematics/revo2_kinematics.py`: GUI와 무관한 Revo2 FK
- `tools/rb3_revo2_ik/rb3_kinematics.py`: RB3 FK/IK와 joint limits
- `tools/rb3_revo2_ik/build_reference_trajectory.py`: frame-by-frame strict IK
- `tools/rb3_revo2_ik/launch_replay_gui.py`: 터미널에서 Isaac Sim GUI 시작
- `tools/rb3_revo2_ik/replay_reference_isaac_sim.py`: Stage replay/controller 구현

`scripts/`는 안정적인 공개 진입점이며, 실제 구현은 `tools/`에 둡니다. 이렇게 하면
내부 파일을 기능별로 유지하면서 사용자는 긴 경로를 외울 필요가 없습니다.

RL 관련 실행은 `scripts/rl.sh {train,play,zero,debug}`로 통합되어 있습니다.
예전 이름의 shell script는 인자와 동작을 그대로 전달하는 호환 wrapper입니다.
전체 명령 목록은 [`scripts/README.md`](../scripts/README.md)에 있습니다.

## 변경 원칙

1. `dataset/` 원본은 읽기만 합니다.
2. 생성 데이터는 `outputs/<stage>/<dataset>/<sequence>/` 아래에 둡니다.
3. 새 테스트는 구현 파일 옆이 아니라 `tests/`에 추가합니다.
4. 사용자가 반복 실행할 명령은 `scripts/` wrapper로 노출합니다.
5. robot-specific 상수와 joint 순서는 해당 kinematics 모듈에 둡니다.
