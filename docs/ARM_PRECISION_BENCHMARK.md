# RB3 정적·저속 정밀 추종: 실행 완료

후속 고정-gains 명령 비교: [저고도 저속 위치·속도 시험](ARM_LOW_HEIGHT_COMMAND_COMPARISON.md).

2026-09-07. **선택용 3개 자세와 보류한 검증 동작 모두 목표를 통과했다.**
기준 포함 4개 gains 설정만 실행하고 탐색을 종료했다. 정책/파지 최적화나 재학습은
하지 않았으며, 기존 training/play 기본값은 바꾸지 않았다.

| 조건 | 정적 최대 위치 [mm] | 정적 최대 회전 [°] | 이동 P95 / 최대 위치 [mm] | 이동 최대 회전 [°] |
|---|---:|---:|---:|---:|
| 선택용 baseline | 21.593 | 2.514 | 21.573 / 21.594 | 2.515 |
| 후보 c1 | 3.453 | .469 | 3.448 / 3.453 | .473 |
| 후보 c2 | 1.448 | .222 | 1.446 / 1.448 | .224 |
| **선택 c3** | **.745** | **.134** | **.743 / .745** | **.135** |
| 보류 검증 baseline | 22.329 | 2.555 | 22.322 / 22.329 | 2.555 |
| **보류 검증 best(c3)** | **.773** | **.106** | **.773 / .774** | **.107** |

정적 기준은 마지막 고정 1초의 최대 위치 ≤1 mm 및 최대 회전 ≤.5°.
이동 기준은 고정된 이동 구간 P95 위치 ≤2 mm, 회전은 더 엄격하게 **최대 ≤.5°**로
판정했다. 각 동작의 시간축과 목표는 동일하며 보정/시간 이동을 적용하지 않았다.

[전후 비교 그래프](../outputs/diagnostics/precision_20260907/comparison/before_after.png) ·
[비교 검증 JSON](../outputs/diagnostics/precision_20260907/comparison/comparison.json).

## 선택한 실험 gains

암 관절 이름으로 매핑한다. Kp 단위는 N·m/rad, Kd는 N·m·s/rad인 기존 implicit
PhysX **force drive**이다. 가속도 drive로 바꾸지 않았다.

| 관절 | 기준 Kp → 선택 Kp | 기준 Kd → 선택 Kd | 유지한 effort limit [N·m] |
|---|---:|---:|---:|
| base | 300 → 700 | 20 → 35 | 10 |
| shoulder | 500 → 20000 | 20 → 260 | 100 |
| elbow | 500 → 12000 | 20 → 140 | 100 |
| wrist1 | 300 → 900 | 20 → 45 | 100 |
| wrist2 | 200 → 2400 | 20 → 70 | 100 |
| wrist3 | 50 → 250 | 10 → 20 | 10 |

팔 velocity limit은 전부 10 rad/s로 유지. 중력 보상/기타 feedforward는 **없음**,
실제 simulator effort setter는 모든 timestep에서 0이었다. 부착된 손까지 포함하고
effort limit을 우회하지 않는 보상 경로를 검증하지 않았으므로 보상 후보를 만들지 않았다.
실제 drive-only torque 측정값이 없으므로 solver torque saturation은 **UNKNOWN**.

baseline의 정적 shoulder/elbow 오차가 약 .015–.025 rad로 base보다 훨씬 컸다.
그래서 이 두 관절의 강성/감쇠를 중심으로 단계적으로 높였고 손목 관절은 별도 값으로
조정했다. 모든 관절을 같은 배율로 올리거나 가장 큰 gains를 계속 탐색하지 않았다.
c2는 이동 기준만 통과, c3가 두 기준을 통과해 고정한 뒤 검증 자세를 실행했다.

저장 설정: [rb3_precision_best.json](../config/experiments/rb3_precision_best.json).
후보 이력: [rb3_precision_candidates.json](../config/experiments/rb3_precision_candidates.json).
이 파일들은 실험 runner만 읽고 정상 환경 config에서는 참조하지 않는다.

## 튜닝 전에 고정한 벤치마크

[rb3_precision_benchmark.json](../config/experiments/rb3_precision_benchmark.json)에
정확한 joint pose, 초기 상태, 구간 길이, 판정 창과 안정성 임계값을 저장했다.
기존 검증된 FK/IK로 작업 공간의 자세를 한 번 선택하고 joint 좌표를 고정했다.
runtime에서는 IK를 다시 풀거나 목표를 actual pose에 맞춰 바꾸지 않는다.

| 자세 | 목표 Revo2 base world XYZ [m] | 용도 |
|---|---|---|
| A | [.43, -.08, .32] | 선택용 |
| B | [.48, .02, .36] | 선택용 |
| C | [.40, .12, .30] | 선택용 |
| D | [.46, -.13, .38] | 보류 검증 전용 |

초기 q는 A에 `[-.03,.03,-.02,.02,0,-.02]` rad를 더한 값, 초기 관절 속도는 0.
각 **실행 시작에만** 이 상태를 설정한다. 각 자세로 직접 reset/teleport하지 않는다.

- 2초 동안 초기 자세에 물리적으로 정착.
- 선택용: 초기→A→B→C, 각 이동 6초 + 유지 3초, 총 29초/3480 physics step.
- 검증용: 초기→D→A, 각 이동 6초 + 유지 3초, 총 20초/2400 step.
- 선택용 이동 측정: (2,8], (11,17], (20,26] s 전체.
- 선택용 정적 측정: (10,11], (19,20], (28,29] s. 검증은 앞의 두 창만 사용.
- 위치와 속도는 같은 quintic `s(u)=10u³−15u⁴+6u⁵` 및 해석적 미분으로 생성.
  endpoint 속도·가속도는 0. 정책 명령을 차분하거나 필터링하지 않는다.
- 명령 속도 ≤.5 rad/s, 가속도 ≤1 rad/s²를 **해석적 최대값**으로 검사한다.
  이는 시험 설계 범위이지 제조사 제원이 아니다. 실행 중 궤적을 늦추지 않았다.
- 각 step은 `q(t_k), dq(t_k)`를 `t_k−dt`에 전달하고 `t_k`의 runtime 상태와
  비교한다. 한 step 끝의 목표를 알려진 smooth trajectory에서 선계산하는 동일한
  discrete command 규칙이다. 사후 시간 이동이나 actual에 맞춘 목표 fitting은 없다.

baseline은 **기존 gains + 동일한 해석적 위치/속도 입력**이다. 기존 정책의 zero
velocity 실행과 다른 벤치마크이므로 정책 결과(19/20)와 직접 비교하는 실험이 아니다.

## 보존한 물리와 비교 검증

기존 `Online-Play` 환경/asset과 PhysX `current_hand_wrist_pos/quat`를 그대로 썼다.
같은 world Revo2 base를 기존 `RB3730Kinematics.forward_batch(q_target)`와 비교한다.
환경 manager의 policy/action/reference 업데이트는 측정 중 호출하지 않는다.

- 손 및 연결부의 질량/관성을 유지. 손가락은 모두 0 rad의 **고정 위치 목표**를
  기존 손 drive로 유지한다. 손의 실제 관절을 매 step 덮어쓰거나 용접하지 않는다.
  손가락의 물리적 탄성 오차는 baseline 최대 .01072 rad, 선택 설정 .00890 rad였다.
- gravity `[0,0,-9.81]`, dt=1/120 s, implicit force drive, solver/iteration/sleep/
  stabilization 설정, collision 설정을 그대로 유지했다. 원래 self-collision=false도 유지.
- native runtime의 effort/velocity/position limits, 전 link mass/inertia 및
  gravity-disabled 플래그를 기록했다. 손 Kp/Kd도 그대로다.
- 모든 비교에서 benchmark hash, 초기 전체 관절/속도 및 robot/object root state,
  물리 config, native limits/mass/inertia, 모든 q/v/a target·target pose·timestamp가
  정확히 동일함을 비교 스크립트가 확인했다. 바뀐 것은 팔 Kp/Kd뿐이다.
- 접촉 보고만 추가했다. 기존 충돌을 비활성화하지 않았으며 전 24개 실제 articulation
  link를 센서가 포함하는지 확인했다. c3와 held-out에서는 **정확한 prim path 집합**까지
  native articulation link 목록과 대조했다. 측정 구간 전체의 보고된 접촉력은 0 N.
  비활성화된 self-collision pair까지 충돌 안전성을 입증한 것은 아니다.

설치 contact view의 wildcard 탐색 과정에서 동일 이름의 하위 mesh prim에 대한
`Failed to find contact report API` 경고가 있었다. 실제 센서 body는 올바른 24개
rigid link였고, c3/검증 metadata의 `contact_body_paths`와 `articulation_link_paths`
집합 일치 검사를 통과했다. 누락된 실제 link를 0 N으로 간주한 것이 아니다.

## 안정성, 정착시간, 오버슈트와 한계

최종 hold 창의 모든 관절 peak-to-peak ≤.0005 rad 및 속도 ≤.005 rad/s를
비진동 판정 조건으로 미리 고정했다. 후보 네 개 모두 이 조건을 통과했다.
선택 설정 최종 hold의 peak-to-peak는 약 3.73e-9 rad 이하였고, simulator의 원래
sleep/stabilization 설정을 유지한 결과다. 외란 후 회복이나 장시간 안정성 검증은 아니다.

- 선택 설정: 이동 중 최대 실제 속도 .40693 rad/s, 가속도 .21103 rad/s².
- 검증 설정: 이동 중 최대 실제 속도 .11572 rad/s, 가속도 .06398 rad/s².
- **초기 2초 정착 구간의 가속도는 시험 명령 bound보다 높다.** baseline 최대
  17.397 rad/s², best 최대 5.607 rad/s². 1 rad/s²를 넘는 sample은 각각 22/4개이며,
  고정된 이동 평가 구간에서는 양쪽 모두 0개였다. 전체 시간에 실제 가속도 ≤1이라고
  주장하지 않는다. 전체 실제 속도는 .5 rad/s 이하, native position/velocity limit
  위반 및 NaN/Inf는 모든 실행에서 없었다.
- baseline은 유지 구간에서 1 mm/.5°에 정착하지 못했다(null). best는 목표 도착 후
  유지 구간 첫 sample(8.33 ms)부터 계속 조건을 만족했다. 이미 감속하며 도착한
  결과이지 큰 위치 step에 대한 8.33 ms 정착 성능을 뜻하지 않는다.
- 목표 endpoint를 접근 방향으로 넘은 최대 joint 오차는 선택용 best .000993 rad,
  검증용 .001855 rad. 이 값은 static bias도 포함하며, fitted 평형점을 기준으로 한
  overshoot가 아니다. 자세별 6개 관절 오차·overshoot·정착시간은 각 summary의 `holds`에 저장.
- 큰 Kp/Kd는 접촉·빠른 명령 변화의 응답을 바꿀 수 있다. 이 실험은 정적/저속·비접촉에
  한정되며 **원래 속도 정책 추종, 파지 성공, 실제 RB3 하드웨어 적용을 검증하지 않는다**.
  이전 [정책 비교의 파지 악화](ARM_VELOCITY_CONTACT_POLICY_VALIDATION.md)를 무효화하지 않는다.

## 재현 명령

프로젝트 루트에서 실행한다. 존재하는 output은 덮어쓰지 않으므로 새 폴더 이름을 쓴다.
실제 완료 실행은 아래 `precision_repeat` 대신 `precision_20260907`을 사용했다.

```bash
# 현재 gains 및 세 후보: 고정된 선택용 벤치마크
for candidate in baseline c1 c2 c3; do
  bash scripts/benchmark_arm_precision.sh --headless --candidate "$candidate" \
    --output "outputs/diagnostics/precision_repeat/$candidate"
done

# 선택한 설정을 바꾸지 않고 held-out 검증
bash scripts/benchmark_arm_precision.sh --headless \
  --candidates config/experiments/rb3_precision_best.json --candidate best --held-out \
  --output outputs/diagnostics/precision_repeat/best_heldout
bash scripts/benchmark_arm_precision.sh --headless --candidate baseline --held-out \
  --output outputs/diagnostics/precision_repeat/baseline_heldout

bash scripts/compare_arm_precision.sh outputs/diagnostics/precision_repeat \
  --output-dir outputs/diagnostics/precision_repeat/comparison
```

각 실행 폴더에는 `metadata.json`(고정 입력/설정/runtime 값), `trace.jsonl`(매 physics
step 상태/명령/오차/접촉력), `summary.json`, `precision.png`가 있다.
비교 폴더에는 `comparison.json`과 `before_after.png`가 있다.
Console 로그는 `/tmp/precision_{baseline_launch2,c1,c2,c3,best_heldout,baseline_heldout}_20260907.log`.
첫 baseline 시도는 AppLauncher의 headless 옵션 차이로 시작 전에 실패했으며,
`/tmp/precision_baseline_20260907.log`를 보존했다. 정상 학습/play 코드는 수정하지 않았다.

추가 파일: `config/experiments/rb3_precision_*.json`,
`tools/rb3_revo2_ik/{precision_trajectory,benchmark_arm_precision}.py`,
`tools/arm_diagnostics/{analyze_arm_precision,compare_arm_precision}.py`,
`scripts/{benchmark_arm_precision,compare_arm_precision}.sh`, `tests/test_arm_precision.py`, 이 문서.

`validate-regrind-change`에 따라 실제 Isaac 실행 6회 완료, 고정 입력/설정 비교 통과,
**65/65 회귀 테스트**, Python compile, shell syntax, focused status/diff 및 whitespace 검사를 수행했다.
