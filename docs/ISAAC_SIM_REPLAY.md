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

로봇은 position drive target으로 움직이고 캔은 첫 pose에 한 번만 생성됩니다.
이후 캔은 gravity, table, robot contact로만 움직입니다.

```bash
./scripts/run_isaac_replay.sh \
  --sequence 20200709_143747_left \
  --physics-object
```

이 모드는 현재 grasp 성공을 보장하지 않습니다. trajectory가 open-loop kinematic
reference이고 controller/contact tuning을 하지 않았기 때문입니다.

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
