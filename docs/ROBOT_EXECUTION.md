# 실제 로봇 연결을 위한 입출력 경계

2026-09-15. **SDK 연결 코드/모의 검증 단계이며, 실물 연결·구동 완료가 아니다.**
공식 `rbpodo` / `bc-stark-sdk`를 별도 환경에 설치하고 RB3 상태 채널과
사용자가 지정한 Revo2 RS-485의 **읽기 전용** adapter를 추가했다.
[SDK 연결 명령](#공식-sdk-설치와-읽기-전용-연결)을 사용한다.
실제 IP·시리얼 포트·baudrate·ID·보정값은 아직 없고 USB 시리얼 장치도
현재 PC에 나타나지 않는다. 서보 활성화/실물 운동은 실행하지 않았다.
`ExecutionSession`의 `hardware` motion backend는 계속 시작 전에 거부한다.
읽기 전용 probe는 이 세션을 사용하지 않으며 구동 해제 CLI는 없다.

기존 [floating / mounted 실행](RL_TASK.md#approved-video-controller-and-timed-transfer),
정책/normalizer, reference, timestep, gains, assets는 변경하지 않았다.
새 경계는 opt-in이며 현재 `rl.sh play-arm`을 대체하지 않는다.

## 분리한 구조

```text
실제 장치 상태 또는 Isaac runtime 상태 (이름·시각·출처 포함)
    → RobotState
    → 호출자: 기존 관측/history/정책/reference decoder
    → DecodedTarget (Revo2 base world pose + 손 leader 6개)
    → ExecutionSession: 기존 FK/IK + 명령 검사
    → JointCommand (RB3 6 + Revo2 leader 6)
    → RobotBackend.read_state / send / stop
         ├ MockBackend: 실행 검증 완료, 이상적인 모의 장치
         ├ SimulationCallbacks: 기존 simulator 입출력 함수를 연결하는 경계
         └ 실제 motion backend: 실행 차단 (읽기 전용 SDK probe는 별도 제공)
```

| 파일 | 역할 |
|---|---|
| `tools/robot_execution/contracts.py` | 상태, pose, decoded target, joint command와 이름 매핑 |
| `tools/robot_execution/session.py` | measured-state 시작, IK warm start, 기한/한계 검사, 실패 latch |
| `tools/robot_execution/backends.py` | 공통 protocol, mock, simulator callback 경계 |
| `tools/robot_execution/isaac_state.py` | 기존 mounted runtime getter를 재사용하는 읽기 전용 adapter |
| `tools/robot_execution/sdk_readers.py`, `sdk_probe.py` | 공식 SDK 상태 reader / 제한 시간 내 읽기 전용 연결 검사 |
| `tools/robot_execution/dry_run.py` | 실제 FK/IK를 모의 장치에 연결하는 실행 테스트 |
| `config/robot_execution/mock.json` | 모의 테스트 전용 주기/제한값; 실물 사양 아님 |
| `tests/test_robot_execution.py` | 계약/실패 처리 회귀 테스트 |

## 재사용 지점과 아직 연결하지 않은 부분

- FK/IK: `tools/rb3_revo2_ik/rb3_kinematics.py::RB3730Kinematics`와 기존
  `WarmStartIK`. `link6 → mount → right_hand_base_link`가 이미 포함되므로
  mount offset을 다시 적용하지 않는다. `kinematics`와 `solver`를 주입한다.
- 관절 이름: RB3는 kinematics model, 손은 `Revo2Kinematics.joint_names`의
  순서. 손 follower나 semantic point를 독립 명령으로 보내지 않는다.
- 상태 reader: `rb3_revo2_commands.py::current_hand_wrist_*`,
  `current_object_*`, `robot.data.joint_pos/joint_vel` 경로를 재사용한다.
  기존 `ArmExecutionTrace.state()`처럼 env origin을 더해 world로 통일한다.
  USD XformCache나 desired pose로 actual 값을 대체하지 않는다.
- 정책: 기존 `FrozenPolicyAdapter`와 `ObservationManager`가 소유하는
  67-D actor/history/normalizer를 여기서 재구현하지 않았다. 기존 action은
  **wrist residual 6 + leader residual 6**이며 RB3 joint residual이 아니다.
  실물 상태→관측 이력 adapter와 실제 물체 추적은 후속 연결이 필요하다.
- Decoder: `SE3ImpedanceActionTerm.process_actions`와 기존 hand decoder가
  reference/residual을 결합한 **뒤의** target을 받는 경계다. raw action이나
  이미 합쳐진 target에 residual을 다시 더하면 안 된다.
- `SimulationCallbacks`의 actuator writer/stop은 호출자가 제공해야 한다.
  읽기 adapter의 단위 테스트만 완료했고 새 session을 Isaac actuator에
  설치한 물리 실행은 아직 하지 않았다. 기존 Isaac 경로는 그대로 사용한다.

## 단위, frame, 시간

- 내부 단위: m / rad / s, quaternion **XYZW**. 단위 quaternion만 받는다.
  실제 SDK의 degree/count/부호/영점/순서는 driver 입출력에서 변환해야 한다.
- `Pose.frame`은 caller의 보정된 world 이름이다. FK의 `base_position`과
  `base_quaternion_xyzw`도 같은 world 기준으로 구성해야 한다.
  문자열 일치만으로 실물 calibration을 검증했다고 볼 수 없다.
- 실물 wrist는 보정된 encoder FK 또는 외부 측정이다. 출처를
  `encoder_fk` / `external_tracking`으로 기록한다. `mock_fk`는 실물/Isaac
  상태로 사용할 수 없다. 실물 mount/model 오차의 별도 검증이 필요하다.
- 시각은 host monotonic clock 기준 **취득 시각**이다. 장치 시각은 clock
  동기화/offset 보정이 필요하다. 캐시 읽기 시각으로 취득 시각을 덮지 않는다.
  물체도 별도 취득 시각을 갖고 freshness 검사를 받는다.
- Isaac reader는 physics와 runtime buffer update **이후**의 같은 timestep에서
  `sample(sequence, sample_time)`로 호출한다. 그 스냅샷을 다음 제어 tick에서
  사용한다. reader 자체가 physics를 step하거나 fresh buffer임을 보증하지 않는다.
- policy/phase 30 Hz와 제어 120 Hz는 다른 주기다. session은 scheduler가
  아니며 callback을 호출한다고 매번 policy/phase를 갱신해서는 안 된다.
  기존 보간/response shaping은 caller의 기존 경로가 소유한다.
  이 실험 경계는 호출 간격이 설정 주기의 ±10% 밖이면 거부한다. 늦은
  명령을 한꺼번에 보내 따라잡지 않는다. 실시간 성능 보장 수치는 아니다.
- `v_path = (q_cmd[k]-q_cmd[k-1])/configured_period`는 명령 경로 검사값이다.
  최소 session의 **velocity target은 0**이며 v_path를 자동 전송하지 않는다.
  승인된 video controller의 v_path/response 설정과 동일한 controller라는
  주장이 아니다. 새 controller 또는 하드웨어 성능 비교에 사용하지 않는다.

## 세션 동작과 한계

`start()`는 측정 관절값으로 warm start/명령 이력을 초기화한다. reference
첫 pose로 teleport하지 않는다. `step(target_factory)`는 최신 실제 상태를
읽어 callback에 제공하고, 목표와 FK frame을 확인한 다음 기존 IK를 호출한다.
이전 accepted command를 initial seed, 현재 actual arm을 alternate seed로 쓴다.
손도 target/actual을 분리한다. 처음부터 먼 target이면 자동 접근하지 않고 거부한다.

검사: 유한값, 관절 이름, 측정 sequence/시각, readiness/protective stop,
물체 측정 freshness(기본 필수), 관절 position/speed, command/actual 간격,
raw joint step/경로 속도/경로 가속도, IK 성공, target/송신 deadline.
각도 wrap, 목표의 조용한 clipping, phase 시간 늘리기로 통과시키지 않는다.
송신 일부 실패/timeout/IK 실패는 양 장치 stop 요청 및 fault latch로 처리한다.
stop 요청 자체가 실패하면 `STOP REQUEST FAILED`로 구분한다. 자동 재시작은 없다.

**이 검사는 안전 인증된 제어기가 아니다.** 충돌·접촉·힘 제한·실물 homing은
해결하지 않는다. Python이 멈추거나 프로세스가 종료되면 이 코드가 stop을
호출하지 못할 수 있다. 실제 controller-side watchdog, 물리 비상정지,
운전자 승인/작업영역 검증이 별도로 필수다. simulator gains/effort limits를
실물 SDK에 그대로 복사하지 않는다. 토크 측정/포화는 이 경계에서 미지원이다.

실제 motion backend 연결 전에 추가 확인할 내용:

1. RB3와 Revo2 SDK/firmware, 통신 경로, 실제 position stream API와 지원 주기.
2. 관절 이름/순서/단위/영점/부호, Revo2 leader→실물 coupling 동작.
3. 실물 joint/속도/가속도 한계, 양 장치 command acknowledgement 및 일부 실패 처리.
4. 장치 자체 watchdog과 controlled stop, 별도 물리 비상정지/enable 절차.
5. robot base/mount/camera/object frame 보정과 측정 timestamp/지연.
6. 실제 관측 adapter와 원본 policy의 관측/action 동등성, history/phase reset.
7. 읽기 전용 연결 → 작은 비접촉 운동 → 저속 접근 → 감독하 접촉 검증.

공식 SDK의 조회 API만 구현했으며 실물 주소/baudrate/ID는 추측하지 않았다.

## 공식 SDK 설치와 읽기 전용 연결

| 장치 | 공식 출처 / 설치 고정 버전 | 이번에 사용하는 API |
|---|---|---|
| RB3-730 | [RainbowRobotics/rbpodo](https://github.com/RainbowRobotics/rbpodo/tree/v0.16.14), `rbpodo==0.16.14` | `CobotData(address, 5001).request_data(timeout)` |
| Revo2 RS-485 | [BrainCo 공식 SDK 안내](https://www.brainco-hz.com/docs/revolimb-hand/en/revo2/get_sdk.html), [공식 예제](https://github.com/BrainCoTech/brainco-hand-sdk), `bc-stark-sdk==2.0.3` | `modbus_open` → `get_device_info`, `get_finger_unit_mode`, `get_motor_status` → `modbus_close` |

격리 환경은 `tools/robot_execution/.venv/`, 고정 의존성은
`requirements-sdk.txt`다. Python 3.12에서 설치/import/pip check를 검증했다.
Isaac 가상환경에는 설치하지 않았고, 이 `.venv/`는 기존 gitignore에 포함된다.

```bash
bash scripts/robot_sdk.sh install
bash scripts/robot_sdk.sh doctor  # import와 로컬 포트 목록만; 장치에 접속하지 않음

# 로컬 파일에 실제 주소/포트/속도/ID를 입력. null을 예제 기본값으로 대체하지 않는다.
cp config/robot_execution/hardware.example.json config/robot_execution/hardware.local.json

# 오른손 RS-485만 먼저 조회 (설정 입력 + 실제 장치 연결 후)
bash scripts/robot_sdk.sh probe --device revo2 \
  --config config/robot_execution/hardware.local.json --output NEW_REVO2_READ.json

# RB3 상태 채널만 / 두 장치 모두 조회
bash scripts/robot_sdk.sh probe --device rb3 \
  --config config/robot_execution/hardware.local.json --output NEW_RB3_READ.json
bash scripts/robot_sdk.sh probe --device all \
  --config config/robot_execution/hardware.local.json --output NEW_COMBINED_READ.json
```

`hardware.local.json`은 gitignore에 추가했다. 모든 output은 새 경로만 허용한다.
기본 2회, 최대 10회만 조회하며 robot discovery, baudrate sweep, broadcast,
제어 채널(5000) 접속을 하지 않는다. `get_device_info`의 오른손/기종/식별값을
검사하며 실제 장치의 unit mode, ID, 통신 속도, gains, limits를 변경하지 않는다.
모드 전환·초기화·교정·손 펴기/주먹 쥐기 예제도 실행하지 않는다.
Revo2는 [전원 투입 시 자동 교정 운동이 있을 수 있음](https://www.brainco-hz.com/docs/revolimb-hand/en/revo2/parameters.html)에 유의한다.

RB SDK 생성자의 연결 timeout이 Python에 노출되지 않으므로 probe는 별도
프로세스의 전체 실행 시간도 제한한다. 장치 오류/시간 초과 시 조회 실패로
종료한다. 이는 조회 프로세스 정리일 뿐, 물리 비상정지 기능이 아니다.

### 실제 값과 아직 해석할 수 없는 값

- RB3: 설치된 SDK의 `SystemState.sdata.jnt_ang`는 encoder **degree**로
  명시되어 있어 rad 변환값도 기록한다. 모델의 영점·부호 보정과는 별개다.
  `jnt_ref`는 desired 값으로 분리한다. [고정 버전 정의](https://github.com/RainbowRobotics/rbpodo/blob/v0.16.14/include/rbpodo/data_type.hpp).
- 같은 정의의 `tcp_pos`에는 reference로 덮일 수 있다는 경고가 있어 actual
  mounted wrist로 쓰지 않는다. velocity는 이 상태 채널에 없어 `null`이며
  0으로 만들어 넣지 않는다. `jnt_cur`는 A이고 토크 포화 지표가 아니다.
- Revo2: [물리 단위 문서](https://www.brainco-hz.com/docs/revolimb-hand/en/revo2/faq.html)는
  position 0.1°, speed °/s, current mA를 설명하지만, 설치된 SDK 2.0.3
  `DeviceContext.get_motor_status` docstring은 통합 정규화 범위를 설명한다.
  따라서 `positions/speeds/currents_sdk_raw`와 unit mode를 함께 저장하고
  실물/펌웨어별 검증 전에는 rad나 mA로 가정하지 않는다.
- SDK 배열 순서는 Thumb, ThumbAux, Index, Middle, Ring, Pinky다.
  **프로젝트 thumb metacarpal/proximal 순서와 그대로 같다고 가정하지 않는다.**
  이름 대응, 부호, 영점과 각 채널의 값→각도 변환을 실물로 검증해야 한다.
- host request/response 시각과 RB device timer를 따로 보존한다. Revo2 조회는
  device timestamp가 없으므로 수신 시각을 센서 취득 시각이라고 표기하지 않는다.
  두 장치를 동시에 읽은 것으로도 취급하지 않는다.

이 조회 결과는 `Telemetry`이며 아직 `RobotState`로 자동 변환하지 않는다.
필수 속도/손목 보정/물체 추적/시각 동기화가 빠진 상태를 policy actual
observation에 섞지 않기 위한 경계다. 실제 policy 연결과 motion send는 후속 작업이다.

### SDK 실행 증거와 제한

- 설치/import 확인: `/tmp/regrind_sdk_doctor_20260915.json`.
- 실물 설정 검사 실행: `/tmp/regrind_sdk_probe_20260915.json`.
  `bash scripts/robot_sdk.sh probe --config config/robot_execution/hardware.example.json`
  은 exit 2: RB3 address 누락, Revo2 port 누락. 실제 연결 성공으로 보고하지 않는다.
- SDK 의존성 check 통과. RS-485 장치 미표시를 확인했으며 실물 주소 스캔은 하지 않았다.
- 전체 회귀: 229개 중 227개 통과, SDK 격리환경 전용 2개는 기본 Isaac
  interpreter에서 skip. 전용 환경에서 그 2개를 포함한 SDK 테스트 12개가
  별도로 통과했다. 로그: `/tmp/regrind_sdk_regressions_20260915.log`,
  `/tmp/regrind_sdk_native_final_20260915.log`.
- 아래 native tests는 **실제 설치된 SDK**를 localhost 모의 RB 서버 및 PTY
  가상 시리얼에 연결한다. `reqdata\n`, Modbus FC03/04 읽기만 전송함을 검사한다.
  Revo2 테스트의 identity는 synthetic이며 실제 firmware/장치 식별 검증은 아니다.
  PTY는 USB serial low-latency ioctl을 지원하지 않는 경고가 나지만 조회는 통과했다.

```bash
tools/robot_execution/.venv/bin/python -m unittest tests.test_robot_sdk -v
# localhost 소켓/PTY 사용 권한이 있는 터미널에서:
tools/robot_execution/.venv/bin/python -m unittest tests.test_robot_sdk_native -v
```

## 실행 / 검증

```bash
# 프로젝트 root. 기존 파일 덮어쓰기 금지: 매 실행 다른 output 사용.
bash scripts/check_robot_execution.sh \
  --output outputs/diagnostics/robot_execution_check/commands.jsonl

# Isaac GUI/GPU/장치 없이 계약 검증
/home/wanjunkim/IsaacLab/.venv/bin/python -m unittest tests.test_robot_execution -v
./scripts/run_tests.sh
```

JSONL 첫 줄은 config/joint order/검증 범위, 이후 각 줄은 command,
`actual_before_send`, path speed, IK 오차다. command에는 생성 시각/유효 기한,
state/command sequence, reference frame이 있다. 오류는 fault 행에 기록한다.
원본 policy, reference, checkpoint를 수정하거나 로그를 덮어쓰지 않는다.

실행한 dry run: 120개 명령 / **가상 시간** 1초, 기존 FK→IK와 모의 송신 완료.
최종 로그: `/tmp/regrind_robot_io_20260915_final.jsonl` (로컬 임시 검증 기록).
변경 전 193개 / 변경 후 217개 회귀 테스트 통과(새 계약 테스트 24개).
테스트 로그: `/tmp/regrind_robot_io_20260915_tests.log`.
이상적 mock 장치이므로 작은 FK/IK 오차는 실제 추종 정밀도나 파지 성공이 아니다.
정책 추론, Isaac 접촉, 실물 통신, 120 Hz wall-clock deadline은 이 실행으로
검증하지 않았다. 신규 모듈은 기존 train/play 실행에 설치하지 않았다.
