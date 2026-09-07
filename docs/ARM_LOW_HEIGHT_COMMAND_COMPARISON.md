# 저고도 저속 명령 비교 — 실행 완료

2026-09-07. [정밀 gains 검증](ARM_PRECISION_BENCHMARK.md)의 c3를 **그대로 고정**하고
선형 위치 보간+속도 0과 부드러운 위치·속도 동시 명령을 PhysX에서 비교했다.
원래 목표 대비 오차도 개선됐지만, 실제 RL 파지에 적용하거나 성공을 검증한 결과는 아니다.

## 범위와 명령 계약

- 기존 `Online-Play` asset, FK, joint-name mapping, PhysX runtime base 읽기,
  접촉 reporter를 재사용. 정상 train/play/action 코드는 이번 작업에서 변경하지 않았다.
- Kp `[700,20000,12000,900,2400,250]`, Kd `[35,260,140,45,70,20]`.
  순서 base/shoulder/elbow/wrist1/wrist2/wrist3. 손 자세 목표 0 rad 고정.
  보상 없음; effort/velocity limits, 중력, 질량·관성, 충돌, solver, dt 모두 보존.
- 위치·속도·가속도의 원본은 기존 해석적 quintic. 명령 속도 ≤0.5 rad/s,
  가속도 ≤1 rad/s²라는 시험 한도를 유지했다. 제조사 한도가 아니다.
- A `[.40,-.12,.16]`, B `[.44,-.10,.14]`, C `[.40,-.06,.18]` m.
  기존 벤치마크 A의 손목 방향을 유지. 보류한 D `[.42,-.14,.15]` m.
  원래 파지의 8–11 cm가 아닌 **14–18 cm 비접촉 영역**까지의 검증이다.
- 시작 상태는 A+기존 작은 joint offset. 시작에만 joint state 설정, 이후 물리 drive.
  2초 정착, 이동마다 6초, 정지마다 3초. 정지 마지막 1초, 이동 전체를 평가.
  선택 29초/3480 tick, 보류 20초/2400 tick. 시행 중 속도를 더 늦추지 않았다.
- `linear_zero`: control boundary에서 현재 원본 q waypoint를 받아, 직전 waypoint에서
  4 physics tick 동안 선형 보간. simulator 속도 목표는 0. 기존 온라인 명령의
  보간·zero 속도 구조를 재현한 **합성 입력 시험**이며 기존 실패 episode 재생은 아니다.
- `smooth_pv`: 같은 구간을 원본 quintic의 정확한 미분 정보로 재구성하고 q/dq 동시 전달.
  구현은 이미 고정된 quintic을 4 tick 지연해 평가한다. control boundary에서 끝나는
  직전 구간만 사용하며, 두 방식의 control endpoint q는 일치한다.
  두 방식 모두 원본 대비 33.333 ms 지연. 분석에서 시간 이동하지 않는다.
- 이는 **오프라인 미분 정보를 아는 시험**이다. 임의 정책 출력에 이 미분 정보가 있다고
  가정하지 않으며, 온라인 smoothing 모듈 또는 RL 정책 변경을 구현한 것이 아니다.
  위치 보간과 속도 목표를 함께 바꿨으므로 각 효과를 단독으로 분리한 실험도 아니다.

## 실제 결과

다음 위치·회전은 **지연하지 않은 원본 목표 vs 실제 runtime Revo2 base**이다.

| 지표 | linear_zero | smooth_pv |
|---|---:|---:|
| 선택 이동 위치 평균 [mm] | .797 | .683 |
| 선택 이동 위치 P95 / 최대 [mm] | 1.297 / 1.341 | 1.059 / 1.088 |
| 선택 이동 회전 최대 [deg] | .2145 | .1261 |
| 보류 이동 위치 P95 / 최대 [mm] | .763 / .769 | .692 / .696 |
| 보류 이동 회전 최대 [deg] | .1225 | .1163 |
| 선택 정지 마지막 1초 최대 위치 [mm] | .6374 | .6371 |
| 보류 정지 마지막 1초 최대 위치 [mm] | .6223 | .6225 |

원본 위치 P95는 선택 **18.4%**, 보류 **9.2%** 감소. 정적 차이는 미미하며 보류
정적 오차는 약 .00018 mm 증가했다. 모든 지표가 일괄 개선됐다고 해석하면 안 된다.
원본 기준 이동 최대 1.088 mm이므로 **모든 시각에서 1 mm 이하 달성은 아니다**.

실제 전달 목표 vs 실제 base만 비교하면 선택 P95 `.836 → .636 mm`,
보류 `.634 → .623 mm`. 원본→전달 목표 편차는 선택 최대 `.7244 mm`,
보류 `.3127 mm`이며 양쪽에서 거의 같다. 이 편차를 숨겨 정밀도를 주장하지 않았다.

### 안정성과 한계

| 이동 구간 실제 값 | linear_zero | smooth_pv |
|---|---:|---:|
| 선택 최대 joint 속도 [rad/s] | .06467 | .08077 |
| 선택 최대 joint 가속도 [rad/s²] | .07132 | .04708 |
| 보류 최대 joint 속도 [rad/s] | .02669 | .03375 |
| 보류 최대 joint 가속도 [rad/s²] | .03089 | .04756 |

보류 가속도는 약 54% 증가했다. 다만 두 방식 모두 정지 peak-to-peak/속도 기준,
관절 위치·속도 제한 및 finite 검사를 통과했다. 실제 속도 증가는 목표 속도를 더
따라간 결과와 일치하지만, 그것만으로 인과를 확정하지 않는다.
초기 2초 정착은 이동 통계에서 분리했다. 초기 실제 가속도 최대는 네 실행 모두
**4.537 rad/s²**로, 1 rad/s²의 명령 설계 한도보다 크다. 실제 가속도가 전체 실행에서
1 이하였다고 주장하지 않는다. 전체 통계는 각 `summary.json`에 별도 보존했다.
선형 보간 corner의 순간 가속도는 정의되지 않는다. 비교 JSON의 path 가속도는
물리 tick의 backward finite difference이며 연속시간 한도 충족 증명이 아니다.
`linear_zero` trace의 legacy `a_target=0`은 미사용 placeholder이며 simulator 명령이 아니다.

24개 실제 articulation link에 reporter가 연결됐음을 검증했고, 네 실행에서
net contact force peak 0 N, limit violation 0, NaN/Inf 없음. net-force 관측이지
모든 collision pair의 기하학적 간격을 증명한 것은 아니다. 원래 self-collision=false 유지.
실제 solver drive-only torque는 미검증이므로 포화 여부 **UNKNOWN**.

## 재현과 산출물

새 output root를 사용해야 한다. 기존 실행 폴더를 덮어쓰지 않는다.

```bash
for mode in linear_zero smooth_pv; do
  bash scripts/benchmark_arm_precision.sh --headless \
    --benchmark config/experiments/rb3_low_precision_benchmark.json \
    --candidates config/experiments/rb3_precision_best.json --candidate best \
    --command-mode "$mode" \
    --output "outputs/diagnostics/low_precision_repeat/$mode"
done
bash scripts/compare_arm_command_precision.sh \
  outputs/diagnostics/low_precision_repeat/linear_zero \
  outputs/diagnostics/low_precision_repeat/smooth_pv \
  --output outputs/diagnostics/low_precision_repeat/comparison
```

보류 검증은 같은 두 실행에 `--held-out`을 추가하고 output에 `_heldout`을 붙인다.
실제 실행 root는 `outputs/diagnostics/low_precision_20260907/`.

- `linear_zero/`, `smooth_pv/`, `linear_zero_heldout/`, `smooth_pv_heldout/`:
  `metadata.json`, `trace.jsonl`, `summary.json`, `precision.png`.
- [선택 비교 그래프](../outputs/diagnostics/low_precision_20260907/comparison/comparison.png),
  [수치](../outputs/diagnostics/low_precision_20260907/comparison/comparison.json).
- [보류 비교 그래프](../outputs/diagnostics/low_precision_20260907/comparison_heldout/comparison.png),
  [수치](../outputs/diagnostics/low_precision_20260907/comparison_heldout/comparison.json).
- console: `/tmp/low_precision_{linear_zero,smooth_pv,linear_zero_heldout,smooth_pv_heldout}_20260907.log`.

비교 스크립트는 동일 benchmark hash, gain, physics, native mass/inertia/limits,
초기 robot/object/joint state, 원본 q/pose/time 배열, 명령 경계 위치를 assert한다.
통계는 per-joint 오차·속도·가속도, hold settling/overshoot를 포함한다.
`./scripts/run_tests.sh`: **69 tests 통과**. 기본 analytical 경로 보존, 공통 지연,
관절 wrapping 없음, 저고도 FK와 미분 일치 회귀를 추가했다.

추가/변경: 실험 config, 기존 `precision_trajectory.py`/`benchmark_arm_precision.py`,
`compare_arm_command_precision.py`, root 비교 wrapper, `test_arm_precision.py`, 이 문서.
일반 학습·평가 기본값은 그대로다. 다음 단계가 필요하다면, 온라인 명령의 인과적
위치·속도 생성과 그 지연을 별도로 검증해야 하며 이 오프라인 결과로 대체할 수 없다.
