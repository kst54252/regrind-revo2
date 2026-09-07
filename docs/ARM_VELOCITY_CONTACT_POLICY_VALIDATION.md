# 속도 목표 변경: 접촉 ON 및 실제 정책 평가

2026-09-07. [OFF 비교](ARM_VELOCITY_TARGET_COMPARISON.md)의 후속 검증.
**이번 20개 동일 초기 배치에서는 더 잘 잡지 못했다. 기존 19/20 → v_path 7/20.**
팔 추종 개선과 파지 성능 개선은 일치하지 않았다. 기존 기본값 `zero`를 유지한다.
추가 원인 분리, gains/limits 튜닝, 재학습, 필터링은 수행하지 않았다.

## 실험 1: 캔 접촉 ON, 동일한 108개 명령

[기존 실패 episode 15](ARM_CONTACT_COMPARISON.md)의 초기 상태/PhysX prefix를
그대로 재현했다. 108 physics timestep(1/120 s, 총 .9 s) 동안 policy/IK 호출 없이
동일한 팔 위치·손가락 위치/속도·feedforward 명령을 전달했다. v_path만 기존
ON actuator 로그에서 같은 timestep으로 가져왔다. 기록 이후 명령은 만들지 않았다.

새 zero 실행의 관절 위치/속도, wrist pose, can root state는 이전 ON baseline과
**모든 sample에서 bit-for-bit 동일**했다. 두 조건의 gains/limits/dt/초기 상태,
명령 및 simulator 위치 setter 전달값의 동일성도 assert했다.

| 지표 | zero | v_path |
|---|---:|---:|
| 손목 위치 mean / p95 / max [mm] | 16.056 / 40.516 / 44.602 | 8.257 / 21.190 / 26.583 |
| 손목 회전 mean / p95 / max [rad] | .118241 / .226505 / .340548 | .023378 / .057251 / .072030 |
| wrist1 mean / max 관절 오차 [rad] | .071076 / .411727 | .002154 / .004537 |
| wrist3 mean / max 관절 오차 [rad] | .135890 / .734730 | .005809 / .018974 |
| wrist1 최대 실제 가속도 [rad/s²] | 51.72 | 276.13 |
| wrist3 최대 실제 가속도 [rad/s²] | 211.80 | 612.99 |
| 캔 최대 상승 [mm] | 23.386 | 21.535 |
| 마지막 캔 XYZ [m] | [.431817, .157167, .012636] | [.446106, .176245, .012636] |
| 기존 종료 조건의 최초 발생 | .9 s: object_deviation | .9 s: object_deviation |

손목 mean 위치/회전 오차는 48.57% / 80.23% 감소했지만 양쪽 모두 캔은 바닥으로
돌아왔다. 이 짧은 고정 재생은 **전체 파지 성공 판정이 아니다**.
기존 종료 term 함수를 원래 control boundary마다 읽기 평가했다. reference frame과
episode length를 평가 시에만 맞추고 즉시 복원했으며, termination/reset을 실행하거나
명령을 연장하지 않았다. zero의 .9 s object_deviation은 원본 실행과 일치한다.

v_path에서 wrist3의 짧은 속도 spike가 생기고 최대 가속도가 증가했다.
.05 rad/s deadband 밖 실제 속도 부호 전환은 wrist1 3→5, wrist3 5→7이었다.
움직이는 목표의 방향 전환도 포함되므로 이 횟수만으로 지속 진동을 판정하지 않는다.
관절 위치 제한 위반/NaN/Inf는 없고 실제 속도는 양쪽 모두 9.5 rad/s 미만이었다.

[ON 시간 그래프](../outputs/diagnostics/arm_velocity_on_analysis_20260907/velocity_target_comparison.png),
[ON 수치](../outputs/diagnostics/arm_velocity_on_analysis_20260907/summary.json).

## 실험 2: 동일 초기 배치 20개, 각 실행의 실제 관측으로 정책+IK

기존 `arm_execution_random20_20260907.jsonl`은 팔/물체 상태는 있지만 손가락을
포함한 full reset state가 없었다. 따라서 이번에 zero 정책 평가를 새로 실행하여
**전체 상태를 가진 공통 20개 초기 배치**를 저장했다. 새 zero 실행은 기존 random20의
2,924 sample에서 q_cmd/q_actual/관절 속도/wrist pose/물체 위치와 정확히 일치했다.
즉 기존 배치는 재현됐지만 완전한 초기 상태 비교의 근거는 이번 full-state 기록이다.

v_path 실행에서는 저장된 placement offset을 기존 reset 경로에 넣고, 그 reset으로
생성된 실제 상태를 검증했다. 단순히 같은 seed였다는 근거만 사용하지 않았다.

- 모든 20회 reset 및 첫 physics step 직전에서 전체 17 joint position/velocity,
  robot/object root pose+velocity, runtime wrist pose, reference frame,
  applied arm target, placement offset의 **최대 차이 0**.
- 위치 목표 이력은 reset의 새 팔 목표로 시작한다. 그 뒤 보간된 최종 팔 목표의
  raw 차분 / physics dt를 계산한다. 이전 episode 목표나 기록된 v_path를 재사용하지 않는다.
- 첫 차분 predecessor는 `initialization.applied_arm_target`이다. `before_state`는
  apply_actions 이후에 읽으므로 그 안의 목표 버퍼는 이미 현재 q_cmd이다.
- 실제 native velocity setter 전달값을 모든 2,484 v_path sample에서 이 차분과
  대조해 통과했다. 속도 setter는 step당 한 번, 손가락 속도/FF 명령은 기존 zero.
- 동일 checkpoint, 위치 보간법, 손가락 제어 구현, gains/limits/dt/중력/충돌 설정.
  policy는 양쪽 모두 `policy(actual_obs)`이며, 대응되는 control step 중 초기 20회
  행동은 같고 이후 601회 행동은 달랐다. 행동/IK 결과를 강제로 같게 만들지 않았다.
- 저장 상태는 명시적인 runtime state이다. 다른 episode 이력 이후 PhysX 내부의
  비공개 contact cache까지 직렬화·동일성 검증했다고 주장하지 않는다.

### 성공/실패 기준과 배치별 결과

기존 `TerminationsCfg`를 그대로 사용했다. `success=demo_end_reached`, 실패는
object 50-keypoint 평균 오차 > .15 m(`object_deviation`) 또는 wrist-object 거리
> .5 m(`hand_far_from_object`) 등 기존 terms이다. 성공 term은 별도의 접촉 센서로
파지를 판정하는 지표가 아니라, 조기 실패 없이 reference 끝에 도달한 지표이다.

- zero: **19 성공, 1 실패**; v_path: **7 성공, 13 실패**.
- 실패→성공: 없음.
- 성공→실패: **0, 1, 2, 3, 6, 10, 11, 13, 14, 17, 18, 19** (0-based).
- 기존 실패 15는 그대로 실패. 모든 실패의 활성 term은 `object_deviation`.
  `hand_far_from_object`, timeout 실패나 IK 실패/fallback은 없었다.

초기 Z는 모두 .012636 m. 마지막 상승은 초기 can Z 기준이며, 실패 조기 종료 시점도
포함한다. 손목 오차는 FK(q_cmd) vs PhysX Revo2 base(stage C).

| 배치 | 초기 X,Y [m] | zero → v_path | 마지막 캔 상승 [mm] | mean 손목 위치 오차 [mm] |
|---|---|---|---:|---:|
| 0 | .43264, -.07406 | 성공 → 실패 | 241.3 → 0.0 | 17.93 → 8.83 |
| 1 | .43584, .01920 | 성공 → 실패 | 239.0 → 30.3 | 17.30 → 7.57 |
| 2 | .45246, -.06908 | 성공 → 실패 | 244.6 → 0.0 | 18.82 → 9.36 |
| 3 | .42197, -.09640 | 성공 → 실패 | 245.2 → 23.6 | 18.40 → 9.48 |
| 4 | .49205, -.16134 | 성공 → 성공 | 241.1 → 237.0 | 23.12 → 13.57 |
| 5 | .43281, -.06401 | 성공 → 성공 | 244.8 → 237.8 | 18.02 → 9.73 |
| 6 | .49150, -.08201 | 성공 → 실패 | 239.8 → 31.0 | 20.42 → 11.34 |
| 7 | .47364, -.14598 | 성공 → 성공 | 241.7 → 246.4 | 21.74 → 12.95 |
| 8 | .44421, -.14261 | 성공 → 성공 | 242.9 → 225.9 | 20.42 → 11.30 |
| 9 | .44460, .05152 | 성공 → 성공 | 236.0 → 242.4 | 17.37 → 9.36 |
| 10 | .45776, .01004 | 성공 → 실패 | 238.8 → 0.0 | 17.76 → 9.11 |
| 11 | .41688, .07168 | 성공 → 실패 | 237.3 → 0.0 | 16.76 → 9.11 |
| 12 | .42276, -.18558 | 성공 → 성공 | 242.9 → 246.9 | 21.23 → 12.03 |
| 13 | .46440, .09414 | 성공 → 실패 | 267.6 → 0.0 | 15.24 → 8.90 |
| 14 | .43011, -.16787 | 성공 → 실패 | 243.1 → 0.0 | 20.85 → 10.88 |
| 15 | .45724, .15631 | 실패 → 실패 | 0.0 → 0.0 | 16.06 → 7.87 |
| 16 | .45687, -.18679 | 성공 → 성공 | 240.5 → 147.3 | 22.43 → 13.30 |
| 17 | .47934, .09808 | 성공 → 실패 | 262.4 → 70.2 | 15.91 → 10.63 |
| 18 | .47546, -.15932 | 성공 → 실패 | 240.5 → 15.4 | 22.07 → 12.61 |
| 19 | .47183, .10960 | 성공 → 실패 | 253.5 → 0.0 | 16.05 → 9.24 |

배치별 회전 오차, wrist1/wrist3를 포함한 각 관절 오차·속도·가속도·부호 전환·
위치 limit 검사는 [전체 JSON](../outputs/diagnostics/arm_policy_velocity_analysis_20260907/summary.json)의
`conditions.<mode>.episodes.<index>`에 있다.

### 추종 개선과 움직임 악화는 함께 관측됨

| 전체 수집 sample 지표 | zero | v_path |
|---|---:|---:|
| sample 수 | 2924 | 2484 |
| stage-C 위치 mean / max [mm] | 18.932 / 56.141 | 10.530 / 37.980 |
| stage-C 회전 mean / max [rad] | .125307 / .810665 | .033025 / .118923 |
| 실제 IK 목표 대비 위치 mean / max [mm] | 22.530 / 66.820 | 13.567 / 53.730 |
| 실제 IK 목표 대비 회전 mean / max [rad] | .133601 / .810203 | .034348 / .132089 |
| wrist1 관절 오차 mean / max [rad] | .086921 / 1.803539 | .005162 / .290196 |
| wrist3 관절 오차 mean / max [rad] | .187143 / 2.140157 | .009113 / .282894 |
| wrist1 최대 실제 가속도 [rad/s²] | 365.61 | 1920.32 |
| wrist3 최대 실제 가속도 [rad/s²] | 277.17 | 2255.68 |

조기 종료 때문에 sample 수/episode 길이가 다르다. 이 전체 mean/max를 동일한
길이의 고정 입력 비교나 순수 인과 분해로 해석하지 않는다. 위 배치별 수치와
접촉 ON 고정 명령 실험을 함께 봐야 한다.

가속도는 인접 physics timestep의 실제 joint velocity 차분이며 reset을 가로질러
미분하지 않았다. v_path 배치 17, .975 s에서 wrist3 속도는 +9.98835 → -8.80896
rad/s로 바뀌어 -2255.68 rad/s²가 기록됐다. 실제 새로운 큰 속도 전환이 관측됐다.
정책 평가의 최대 제출 velocity target은 wrist1 18.8006, wrist3 18.4695 rad/s이고
실제 속도는 약 10 rad/s 부근까지 갔다. 요청대로 목표에 추가 clipping/필터링을
넣지 않았으며, simulator의 기존 10 rad/s limit도 변경하지 않았다.
이 관측으로 drive torque 포화나 원인 하나를 단정하지 않는다.
양쪽 모두 NaN/Inf와 관절 위치 limit 위반은 없었다.

[20개 배치 비교 그래프](../outputs/diagnostics/arm_policy_velocity_analysis_20260907/paired_policy_comparison.png).
이번 20회는 특정 checkpoint/trajectory/배치의 대응 실험이지, 일반적인 성공률 추정이 아니다.

## 코드와 실행 명령

선택 옵션은 `./scripts/rl.sh play-arm --arm-velocity-target {zero,v_path}`.
default `zero`는 기존 위치-only 명령 경로를 유지한다. `RB3WristIKAction.apply_actions`
에서 최종 보간 target과 이전 실제 제출 target의 차분을 사용하며, reset 시 새 목표로
이력을 초기화하고 velocity target을 0으로 지운다. 손가락 제어 구현은 변경하지 않았다.

변경/추가 경로:

- `regrind/scripts/rsl_rl/play.py`: live 선택 옵션, paired 초기 상태 옵션, ON fixed replay 허용.
- `.../mdp/rb3_revo2_actions.py`: opt-in physics-rate 팔 속도 목표와 reset 처리.
- `tools/rb3_revo2_ik/replay_arm_contact.py`: 기존 termination terms의 read-only 경계 검사.
- `tools/rb3_revo2_ik/analyze_arm_velocity.py`: 같은 접촉 조건 비교, can trajectory/종료 flag.
- `tools/rb3_revo2_ik/trace_arm_execution.py`: full reset/명령/config 계측 확장.
- `tools/rb3_revo2_ik/paired_arm_states.py`: 저장 placement 적용 및 실제 초기 상태 검사.
- `tools/rb3_revo2_ik/analyze_arm_policy_velocity.py`, `scripts/analyze_arm_policy_velocity.sh`.
- `tests/test_arm_velocity_live.py`, `tests/test_arm_velocity_analysis.py`.

아래 `_repeat` 이름으로 재실행 가능하다. 경로가 이미 존재하면 새 이름으로 바꿔야 한다.
실제 실행 파일명은 `_repeat` 대신 `_20260907`을 사용했다.

```bash
common=(--sequence 20200709_143747_left
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt
  --num_envs 1 --headless --random-placement)

# 1. 접촉 ON 고정 재생: 108 commands만 사용
replay=(--arm-contact-replay outputs/diagnostics/arm_contact_source_20260907.jsonl
  --arm-contact-episode 15 --arm-contact-condition present --arm-actuator-diagnostic)
./scripts/rl.sh play-arm "${common[@]}" "${replay[@]}" \
  --arm-contact-output outputs/diagnostics/arm_velocity_on_zero_repeat.jsonl
./scripts/rl.sh play-arm "${common[@]}" "${replay[@]}" \
  --arm-contact-output outputs/diagnostics/arm_velocity_on_path_repeat.jsonl \
  --arm-velocity-path-trace outputs/diagnostics/arm_actuator_present_20260907.jsonl
bash scripts/analyze_arm_velocity.sh \
  outputs/diagnostics/arm_velocity_on_zero_repeat.jsonl \
  outputs/diagnostics/arm_velocity_on_path_repeat.jsonl \
  --original outputs/diagnostics/arm_actuator_present_20260907.jsonl \
  --output-dir outputs/diagnostics/arm_velocity_on_analysis_repeat

# 2. live policy+IK: 공통 초기 상태 저장 후 변경 방식 평가
./scripts/rl.sh play-arm "${common[@]}" --eval_episodes 20 --arm-velocity-target zero \
  --arm-execution-trace outputs/diagnostics/arm_policy_velocity_zero20_repeat.jsonl \
  --arm-execution-full-state
./scripts/rl.sh play-arm "${common[@]}" --eval_episodes 20 --arm-velocity-target v_path \
  --arm-evaluation-states outputs/diagnostics/arm_policy_velocity_zero20_repeat.jsonl \
  --arm-execution-trace outputs/diagnostics/arm_policy_velocity_path20_repeat.jsonl \
  --arm-execution-full-state
bash scripts/analyze_arm_policy_velocity.sh \
  outputs/diagnostics/arm_policy_velocity_zero20_repeat.jsonl \
  outputs/diagnostics/arm_policy_velocity_path20_repeat.jsonl \
  --output-dir outputs/diagnostics/arm_policy_velocity_analysis_repeat
```

## 산출물과 검증

`outputs/diagnostics/` 아래 새 파일만 생성했고 이전 로그는 보존했다.

- `arm_velocity_on_{zero,path}_20260907.jsonl`: fixed ON 108-step 로그.
- `arm_velocity_on_analysis_20260907/`: summary/시간 그래프.
- `arm_policy_velocity_{zero20,path20}_20260907.jsonl`: full live traces와 공통 초기 상태.
- `arm_policy_velocity_analysis_20260907/`: summary, placements.csv, 배치 비교 PNG,
  각 조건의 기존 execution analyzer 결과(`zero/`, `v_path/`).
- Console: `/tmp/arm_velocity_on_{zero,path,analysis}_20260907.log`,
  `/tmp/arm_policy_velocity_{zero20,path20}_20260907.log`,
  `/tmp/arm_policy_velocity_analysis_complete_20260907.log`.

네 Isaac 실행 모두 완료. 실제 native 속도 전달, 동일 reset/첫 물리 상태 및 설정
대조 통과. `validate-regrind-change`에 따라 **61/61 회귀 테스트**, Python compile,
shell syntax, focused diff 및 `git diff --check` 검증.
초기 offline 분석은 첫 미분 predecessor를 post-apply 버퍼에서 읽어 assertion이
실패했으며, reset snapshot을 사용하도록 분석 코드만 바로잡았다. 실패 console 로그
`/tmp/arm_policy_velocity_analysis_20260907.log`도 보존했다. 시뮬레이션/명령은 재작성하지 않았다.

성능 악화가 확인되어 v_path는 선택 실험 옵션으로만 남겨 두었다. 정상 실행의
기본값은 zero이며 이 작업에서 추가 튜닝이나 후속 원인 실험은 진행하지 않았다.
