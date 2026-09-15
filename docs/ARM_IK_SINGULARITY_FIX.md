# 120 Hz: small wrist motion, large IK joint motion — verified opt-in fix

**결론:** 물체가 로봇의 관절 한계 밖으로 나가서 생긴 문제가 아니라 손목
특이점 근처의 IK 민감도 문제였다. 관절 끝 한계는 유지하고, 속도와 명령
가속도를 함께 고려하는 IK를 추가했다. 실제 40개 초기 배치의 파지와 마지막
접촉 유지를 보존하면서 문제 구간의 급변/추종 오차를 줄였다. 기본 경로는
보존했고 아래 명령으로 새 옵션을 실행한다.

## Verified cause and scope

Source: `ik120_improvement_20260907/old20_baseline`, placement11 at 1.041667 s.
The input moves 2.317 mm / .002692 rad, but wrist1/wrist3 change -.79817/+.79750 rad.
Wrist2 is .015789 rad (~.90°). The verified weighted pose Jacobian has singular
values min .00618 / max 6.068. This supports a near-wrist-singularity mechanism,
not a large floating-hand motion or a quaternion jump.

The nearest joint limit is elbow, .27543 rad (15.78°) away. The two swinging
wrist joints are >5.7 rad from their position limits. Base-to-wrist distance is
.4345 m and the elbow is bent, not at full extension. There is no evidence of
joint-position clipping at this event. Offline shrinking **every** joint's IK
range by .1 rad still yields the same -.79817/+ .79750 rad step: merely reducing
the end limits does not remove this internal singularity.

실제 wrist1/wrist3 축 사이 각도도 .90466°다. Jacobian의 거의 무효한
관절 변화 방향은 `[.0054, -.0004, -.0009, -.7066, -.0019, +.7075]`로,
대부분 wrist1과 wrist3의 반대 회전이다. 두 회전은 손의 방향에서 거의
상쇄되므로 **손 움직임이 작아 보였다는 관찰이 맞다**. 수학적 IK 해가
가능하다는 것과 그 해에 한 timestep 안에 도달할 수 있다는 것은 다르다.

## Minimal opt-in change

Reuse the existing nonlinear least-squares IK and analytic Jacobian. Instead of
solving unrestricted IK then clipping six coordinates independently, solve with
the joint box `model/runtime position bounds ∩ [previous command ± speed_limit*dt]`.
The previous command, not the previous unconstrained IK solution, centers the box.
No angle wrapping, mass/asset/gain/physical limit/phase/policy change.

The first velocity-only prototype intentionally allowed a **reported pose approximation** near singularities:
acceptance budget 5 mm / .05 rad (2.86°), whereas `ik_strict_pose_success` retains
the original .1 mm / .001 rad check. Beyond the budget or on solver failure,
hold the previous accepted command and log failure; do not pretend the target
was reached. Reset still uses the existing exact IK, maintaining identical initial
states. Native speed limits remain 10 rad/s, not an increased physical limit.

Offline placement11, identical recorded IK inputs: maximum step .083333 rad,
position error .425 mm, rotation .01412 rad (.81°), no budget rejection. The
wrist follows a continuous branch into negative wrist2 rather than demanding
the original abrupt rotation. This is a recorded-input, policy-free result.

## 최종 수정: 속도 + 명령 가속도 제약

속도만 제한한 첫 live 후보는 40/40 파지를 유지했으나 다른 구간에서
실제 가속도 2399 rad/s²가 발생해 채택하지 않았다. 거기서 범위를 넓힌
추가 gain 튜닝 대신, 같은 IK 탐색 범위에 다음 제약을 추가했다.

```
v_next = (q_next - q_cmd_previous) / dt
|v_next| <= native_velocity_limit
|v_next - v_cmd_previous| <= 250 * dt
q_lower <= q_next <= q_upper
```

250 rad/s²는 이번 **명령 생성용 시험값**, 제조사 사양이나 native actuator
설정이 아니다. 기존 nonlinear least-squares와 Jacobian으로 이 범위 안의
최소 pose error 해를 구한다. 관절별로 IK 결과를 나중에 잘라 생기는 누적
오차를 피하고, 손목 축이 겹치는 동안 wrist2가 음수 쪽으로 연속적으로
넘어가는 해를 사용할 수 있게 했다. 새 물리 제어기나 알고리즘 프레임워크는 없다.

가속도 제약 모드에서는 수렴한 유효 범위의 최선 해를 전달한다. 자세 오차가
평가 기준을 넘었다고 갑자기 q를 유지하면 속도가 순간적으로 0이 되어 오히려
가속도 제약을 깨기 때문이다. `.1 mm/.001 rad` strict 결과와
`5 mm/.05 rad` pose 기준은 별도로 기록한다. 최종 40회에서는 후자를 넘지
않았지만, 다른 입력에서도 항상 만족하도록 강제하는 보장은 아니다.
유효 범위가 없거나 solver가 수렴하지 않으면 명확한 오류로 진단 실행을
중단한다. reset에서 이전 명령 속도 이력을 0으로 지우며 exact IK 초기화는 유지한다.

## 실제 120 Hz 결과

`old20_finalbaseline ↔ old20_smooth`, `held20_finalbaseline ↔ held20_smooth`.
기존과 별도 20개 초기 상태를 각각 재사용해 모두 실제 closed-loop policy로
실행했다. 첫 actual arm/hand/object 상태, phase, reset buffers와 actor
observation/action까지 일치했다. 이후 행동은 각각의 actual observation으로
계산했다. RL 재학습, gains, native 관절 한계, dt, asset/질량/접촉 설정 변경 없음.

| 지표 | 기존20: baseline → 수정 | 별도20: baseline → 수정 |
|---|---:|---:|
| 기존 task success | 20/20 → 20/20 | 20/20 → 20/20 |
| 마지막 lift/contact 보조 기준 | 20/20 → 20/20 | 20/20 → 20/20 |
| IK 입력→actual 손목 위치 mean [mm] | 1.050 → .941 | .917 → .871 |
| 위 위치 P95 [mm] | 3.076 → 2.819 | 2.798 → 2.597 |
| 위 위치 max [mm] | **17.282 → 5.852** | **10.289 → 5.998** |
| IK 입력→actual 회전 max [rad] | .12124 → .03885 | .07998 → .04762 |
| 실제 팔 가속도 max [rad/s²] | **636.68 → 289.62** | **729.27 → 281.56** |
| 명령 가속도 max [rad/s²] | 640.05 → 250.001 | 726.23 → 250.001 |

문제의 기존 배치11에 한정하면 최대 IK step은 **.79817→.08334 rad**,
IK 입력→actual 손목 위치 최대 오차는 **17.282→3.190 mm**였다.
특이점이 있는 후반 t>=.9 s만 보면 수정 후 최대 1.586 mm다.
가속도 제한의 .001 rad/s² 정도 초과는 float32 최종 명령 차분의 반올림이다.

중요한 trade-off: 수정 후 raw IK pose error 최대는 위치 **1.169 mm**,
회전 **.04849 rad (2.78°)**다. 원래 strict pose 기준을 넘은 timestep은
70+60=130 / 5928개. 이 짧은 근사 오차를 숨기지 않았으며, 수학적 exact
IK를 고집했을 때보다 **실제 로봇 손목 추종**은 좋아졌다. 저속 1-mm 정밀도
보장이나 모든 배치의 일반적 100% 파지 성공률로 해석하지 않는다.

Solver failure/명령 거절=0, ARM 위치 한계 위반=0, NaN/Inf=0.
기존 index leader/follower의 soft-limit 초과는 최대 .00185 rad로 남아 있다.
검증된 solver drive torque는 없으므로 torque saturation은 여전히 UNKNOWN이다.
기존 실행도 새 코드로 다시 실행했고 **5928 timestep의 관절/속도/손목/물체/
IK·native command/정책 행동이 이전 baseline과 bit-for-bit 동일**했다.
입력 손목 target은 기존 .1 s response 이후 pose다. 모든 비교는 timestamp를
맞췄고 후처리 time shift나 policy phase 지연/동작 시간 연장은 하지 않았다.

## 실행 / 파일

수정안 GUI (새 output 경로 사용):

```bash
bash scripts/play_arm_candidate.sh outputs/diagnostics/ik120_smooth_view \
  --transfer-config config/experiments/rb3_smooth_bounded_ik.json \
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt
```

위 명령은 이 보고서의 5,000회 모델을 명시한다. `--checkpoint`를 생략하면
현재 공통 10,000회 모델을 사용하지만, 이 보고서의 수치는 그 모델의 검증 결과가 아니다.

기존 transfer candidate 동작은 위 추가 `--transfer-config`만 빼면 된다.
원래 arm baseline은 `scripts/rl.sh play-arm`이다.
속도만 제한한 실패 후보 설정은 정리 과정에서 삭제했다. 과거
`rb3_velocity_bounded_ik.json`은 Git 커밋 `531b9c8`에서 복구할 수 있다.

Headless 비교 재현: 기존 `scripts/evaluate_mounted_interface.sh`에
`--mode simple --episodes 20 --headless --fast-ik --recovery-capture`와
동일 checkpoint를 전달한다. baseline transfer config는
`rb3_transfer_recovery_candidate.json`, 수정안은 `rb3_smooth_bounded_ik.json`.
기존20 state bank는 기본값, 별도20은
`--states outputs/diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl`.
checkpoint/reference 경로와 원래 명령은 아래 이전 보고서를 참조한다.

```bash
/home/wanjunkim/IsaacLab/.venv/bin/python -m tools.arm_diagnostics.analyze_ik_tracking \
  outputs/diagnostics/ik120_velocity_bound_20260907 \
  old20_finalbaseline old20_smooth held20_finalbaseline held20_smooth \
  --paired --plot-episode 11
./scripts/run_tests.sh
```

- 구현: `tools/rb3_revo2_ik/velocity_bounded_ik.py`.
- 연결: `mdp/simple_mounted_interface.py`, `evaluate_mounted_interface.py`의
  opt-in 설정/명령 및 pose-success 구분. 기존 decoder/FK/model은 그대로다.
- 분석: `tools/arm_diagnostics/analyze_ik_tracking.py`에 IK input→actual(F) 오차와 특이점 구간 그래프 추가.
- 테스트: `test_velocity_bounded_ik.py`, `test_policy_rate_ik.py`의 reset 회귀.
- 로그/그래프: `outputs/diagnostics/ik120_velocity_bound_20260907/`.
  `singularity_old20_finalbaseline_old20_smooth_ep11.png`가 요청 구간 비교다.
- `metadata.json`, `physics.jsonl`, `policy.json`, `ik120_summary.json`에
  초기 상태, 실제 명령/상태, pose approximation 및 성공 판정을 보존했다.

완료: 실제 비교, `./scripts/run_tests.sh` **112개 통과**, `git diff --check` 통과.
변경 후 `validate-regrind-change` 절차로 baseline 재현 및 경계 계약을 확인했다.
이 시뮬레이션용 선택 옵션을 실물에 바로
적용하거나 기본 경로로 무조건 승격하지 않는다.

See [previous 120 Hz comparison](ARM_IK120_IMPROVEMENT.md) for unchanged
checkpoint, reference, banks, effort-signal provenance and previous gain rejection.

## 2026-09-11: 같은 특이점 입력으로 여러 회피 방법 재비교

**새로운 오프라인 IK 비교를 실행했다.** 위의 과거 40회 물리 파지 검증과 별개다.
사진으로 보여준 배치11의 실제 `ik_input_pos/quat` 148개와 reset 1개를 사용했다.
nominal reference나 actual wrist로 목표를 대체하지 않았다.
입력은 **5,000회 checkpoint의 과거 rollout**이며, 현재 기본 10,000회 정책의
새 평가 결과가 아니다. 기본 실행·정책·gains·asset은 변경하지 않았다.

### 고정 조건과 평가 의미

- source: `outputs/diagnostics/ik120_improvement_20260907/old20_baseline`.
- 120 Hz, 148 timestep / 1.233333초. 늘이거나 시간축을 이동하지 않았다.
- 캔 초기 중심: `[0.41688213, 0.07168479, 0.01263600] m`.
- 기존 FK/IK, mount, joint ordering, analytic Jacobian과 branch/bounded 도구 재사용.
- 기존 속도 한계 10 rad/s, strict IK 0.1 mm / 0.001 rad.
- seed 24개, 각 solve 최대 300회 평가. yaw 후보는
  −60°, −30°, −15°, +15°, +30°, +60°, +90°로 미리 고정.
- 가속도 제한 후보의 250 rad/s²는 **명령용 시험 제약**이다. 물리 한계는 그대로다.
- joint step/velocity는 raw joint coordinate 차분. ±π로 감싸 급변을 숨기지 않는다.
  가속도는 reset 속도 0에서 시작하는 backward difference.
- 원본 `command_time_s`는 **30 Hz 정책 시각**이다. physics-rate IK 명령 시각은
  post-step `time_s - physics_dt`로 별도 저장한다.
- 초기 pose만 저장된 reset q의 FK로 표현한다. 나머지 IK 입력은 그대로다.
- 작업대 검사는 팔 중심선 대 table/pedestal/leg AABB이며 관절 경로 중간점도
  최대 0.02 rad 간격으로 검사했다. **전체 mesh·손가락·self collision 검증이 아니다.**

### 문제 배치11 결과

각 수치는 전체 148개 간격의 최대값. 손목 오차는 **목표↔FK**이지 실제 actuator
tracking error가 아니다. 수치 오차 수준은 위치/회전 모두 strict 기준 이하다.

| 방법 | 최대 joint step [rad] | 최대 속도 [rad/s] | 최대 명령 가속도 [rad/s²] | FK 위치/회전 오차 최대 | 판단 |
|---|---:|---:|---:|---|---|
| 기록된 raw IK / 같은 초기 자세 warm start | 0.798175 | 95.781 | 5789.39 | 수치 오차 수준 | 급변 재현 |
| 모든 관절 끝 제한 0.1 rad 축소 | 0.798175 | 95.781 | 5789.39 | 수치 오차 수준 | 내부 손목 특이점에는 효과 없음 |
| 초기 branch 탐색, 작업대 검사 전 | 0.018901 | 2.268 | 110.60 | 수치 오차 수준 | 팔 중심선이 작업대/받침대 고체를 통과 → 제외 |
| 작업대 검사 포함 branch/경로 선택 | 0.128090 | 15.371 | 514.92 | 수치 오차 수준 | 좋아지지만 10 rad/s 초과 |
| IK 내부 속도 제한 | 0.083333 | 10.000 | 2396.70 | 0.425 mm / 0.809° | 근사 pose, 높은 가속도 잔존 |
| IK 내부 속도+가속도 제한 | 0.083333 | 10.000 | 250.00 | 1.077 mm / 2.150° | 급변 완화, strict pose 불충족 |
| **전체 task +60° yaw + 초기 branch 선택** | **0.010733** | **1.288** | **65.43** | **수치 오차 수준** | **이번 오프라인 후보 중 가장 좋음** |

기존 후처리 `q_cmd`도 0.083333 rad로 잘렸지만 FK 위치/회전 오차가
**16.845 mm / 6.839°**였다. 나중에 관절별로 자르기와, 처음부터 관절 제약
안에서 손목 pose 오차를 최소화하기는 결과가 다르다. 기록된 float32 명령의
속도 최대 10.0000048 rad/s는 미소 반올림 초과이며 실제 속도 측정이 아니다.

+60°에서 최대 step은 **98.66% 감소**했다. wrist1/wrist3 축의 최소 분리각은
**0.784° → 83.759°**, 동일 가중치 Jacobian의 최소 singular value는
**0.00618 → 0.70138**이다. 제한 IK는 특이점 근처를 근사해서 통과하는
**완화**이고, yaw 후보는 해당 손목 특이점에서 멀어지는 **회피**다.
모든 종류의 singularity가 없다는 보장은 아니다.

원래 최대 급변 sample t=1.041667초(명령 t=1.033333초)에서 wrist1/wrist3의
step은 −0.798175/+0.797501 rad였다. +60° 후보의 같은 간격은
−0.001293/+0.005110 rad다.

### +60° 고정 후 별도 배치 확인

배치11에서 고른 각도를 바꾸지 않고 기록 배치0/5/19를 새로 계산했다.
초기 branch는 각 배치에서 같은 24-seed 절차로 선택한다. 아래 네 경우 모두
+60°에서 strict IK, native 속도, 관절 한계, 간이 작업대/경로 검사를 통과했다.
원본 배치0은 152 timestep(1.266667초), 나머지는 148 timestep(1.233333초)이다.
각 배치 안에서 모든 방법이 동일 길이의 원본 기록을 사용하며 임의로 자르지 않았다.

| 기록 배치 | 캔 XY [m] | 원래 최대 속도 → +60° [rad/s] | 원래 최대 가속도 → +60° [rad/s²] |
|---|---|---:|---:|
| 11, 방향 선택용 | [0.4169, 0.0717] | 95.781 → 1.288 | 5789.39 → 65.43 |
| 0, 추가 확인 | [0.4326, −0.0741] | 1.829 → 1.318 | 85.00 → 66.99 |
| 5, 추가 확인 | [0.4328, −0.0640] | 1.946 → 1.287 | 99.70 → 64.61 |
| 19, 추가 확인 | [0.4718, 0.1096] | 10.965 → 1.248 | 267.75 → 58.98 |

파지 성공 횟수가 아니며, 원래 문제가 없는 배치도 포함한다.
위치마다 최선의 yaw가 달라질 수 있어 +60°를 보편적 해답으로 고정하지 않는다.

### 무엇을 바꾸는 방법인가

캔 중심 `c`는 유지하고, world +Z 축 기준으로 **손목/물체 전체 task**를 회전한다.
위에서 내려다보면 양수는 반시계 방향이다.

$$
p'_t = c + R_z(\theta)(p_t-c),\qquad
R'_t = R_z(\theta)R_t .
$$

손목 회전만 돌리는 것이 아니다. 상대 파지 기하는 보존하지만 world 목표와
초기 관절 자세가 달라진다. 배치11 +60°의 초기 관절 차이 최대는 1.038 rad다.
자유 branch 후보도 reset 자세가 다르므로 같은 reset 비교로 취급하지 않는다.
시작 자세로 이동하는 접근 경로는 이번에 검사하지 않았다.

**다음 우선순위는 위치별 접근 yaw + 초기 branch 선택**이다. 기존 정책이 회전에
자동으로 불변이라고 가정할 수 없으므로 실제 RL 경로에 넣으려면 reference,
object, observation의 좌표계를 일관되게 처리하고 closed-loop 접촉 평가가 필요하다.
그 검증 없이 새 기본 경로로 바꾸지 않았다. 이번에는 Isaac physics를 실행하지 않아
실제 추종 오차·파지 성공·drive torque는 미확인이다.

### 재현과 파일

```bash
bash scripts/compare_singularity_methods.sh \
  --source outputs/diagnostics/ik120_improvement_20260907/old20_baseline \
  --episode 11 --out outputs/diagnostics/singularity_methods_repeat/episode11

for episode in 0 5 19; do
  bash scripts/compare_singularity_methods.sh \
    --source outputs/diagnostics/ik120_improvement_20260907/old20_baseline \
    --episode "$episode" --yaws 60 \
    --out "outputs/diagnostics/singularity_methods_repeat/validation_ep$episode"
done
./scripts/run_tests.sh
```

기존 output이 있으면 덮어쓰지 않고 종료한다. 원본 trace/state bank가 필요하며
소형 출력만으로 frozen policy의 동적 재평가를 대신할 수는 없다.

- 구현: `tools/arm_diagnostics/compare_singularity_methods.py`, root launcher,
  `tests/test_singularity_methods.py`.
- 실제 출력: `outputs/diagnostics/singularity_methods_20260911/episode11_run/` 및
  `validation_ep0/`, `validation_ep5/`, `validation_ep19/`.
- 각 폴더의 `experiment.json`(조건/hash), `inputs.npz`(선택된 입력),
  `results.json` / `comparison.csv`(실패와 통계), 방법별 `*.npz`(q/FK/오차),
  `comparison.png`(그래프). NPZ는 **진단용**이며 바로 재생하는 12-DoF reference가 아니다.
- [문제 구간 비교 그래프](../outputs/diagnostics/singularity_methods_20260911/episode11_run/comparison.png).

## 2026-09-15: Analytic all-branch IK와 SQP 비교

**실행 완료: solver만 바꿔 원래 시간·손목 pose·작업대를 모두 유지하는 엄격한
해결은 확인되지 않았다.** 해석해 전체 탐색은 기존 수치 IK의 급변을 재현했고,
SQP는 pose 오차를 허용하면 급변을 완화했지만 특이점 자체를 회피하지는 않았다.
기본 controller, 정책, gains, USD, reference는 변경하지 않았다.

### 입력과 검사 범위

- 위와 동일한 **과거 5k 정책의 실제 IK 입력**: 배치 11, 추가 배치 0/5/19.
  현재 10k 정책의 새 rollout이나 nominal reference 검증이 아니다.
- 120 Hz 원래 명령 시각. 배치 11/5/19는 각각 reset+148명령,
  배치 0은 reset+152명령. 총 596개 명령, reset 포함 600개 pose.
- yaw/캔 위치/손목 목표/시간축 변경 없음. hand command는 입력에 보존하되
  이 오프라인 계산에서는 실행하지 않음. 기존 mount, world FK, Jacobian,
  joint ordering 및 runtime에서 기록한 10 rad/s 제한 재사용.
- **실제 Isaac physics / actuator / 파지 평가는 실행하지 않았다.** 아래 오차는
  `IK 입력 ↔ FK(q)`이고 속도·가속도는 관절 좌표의 차분이다. 실제 속도나 토크가 아니다.
- 작업대 검사는 기존 팔 중심선–AABB 검사와 선택 경로의 0.02 rad 이하 간격 검사.
  mesh/self/hand 충돌, 다른 시작 자세로 접근하는 경로는 미검증.

### 1. 해석적 IK와 전체 branch 탐색

`analytic_branch_ik.py`는 모델의 ZYY–ZYZ 축과 offset 구조를 확인한 뒤,
shoulder 2 × elbow 2 × wrist 2의 해석해를 구한다. 단순 random/multi-start를
“모든 branch”라고 부르는 구현이 아니다.

먼저 기존 mount를 역변환해 link6 원점(손목 교차점)을 구한다:

$$
T_{world,link6}=T_{world,Revo2base}T_{link6,Revo2base}^{-1}.
$$

RB3 base의 shoulder 기준 wrist-center 좌표를 $(x,y,z)$, 모델의 측면 offset을
$d=-0.00645$ m, 길이를 $L_1=0.286$, $L_2=0.344$ m라 하면,
$\rho=\sqrt{x^2+y^2}$, $\theta=\operatorname{atan2}(y,x)$에 대해

$$
q_1\in\{\theta-\arcsin(d/\rho),\ \theta-\pi+\arcsin(d/\rho)\},
$$

$$
r=x\cos q_1+y\sin q_1,\quad
q_3=\pm\arccos\frac{r^2+z^2-L_1^2-L_2^2}{2L_1L_2},\quad
q_2=\operatorname{atan2}(r,z)-\operatorname{atan2}(L_2\sin q_3,L_1+L_2\cos q_3).
$$

남은 회전의 ZYZ 분해에서 wrist 2개 해를 얻는다. 실제 관절 제한 안에 드는
모든 $2\pi$ 표현도 열거하고, **각 결과를 기존 FK로 검증**한다. 이 입력 600개
모두 8개 기하 branch / 256개 bounded-coordinate 표현이 나왔다. 시간 차분에는
angle wrapping을 사용하지 않는다.

정확히 wrist2=0 또는 π이면 wrist1/wrist3 해가 연속적으로 무한히 존재한다.
그 경우 `singular_families`와 `exhaustive_isolated=False`를 기록하고 전체 branch
그래프 비교를 중단한다. 이번 입력에는 이 경우가 없었으며, near-singular 해를
반올림하여 없애지 않았다.

기존 `minimax_path`로 전체 프레임의 후보 그래프에서 최대 관절 step을 최소화하고,
동률이면 step 제곱합을 최소화한다. **미래 목표를 사용하는 오프라인 선택**이며
causal closed-loop 정책에 그대로 끼울 수 있는 online solver가 아니다.
같은 reset, 자유 reset, 작업대 검사를 생략한 진단 후보를 구분했다.

### 2. SQP / constrained nonlinear optimization

SciPy **1.17.0**의 `minimize(method="SLSQP")`와 기존 pose Jacobian을 사용했다.
각 명령에서 이전 q/v를 사용하며, pose는 원래 목표에 대한 ball constraint다:

$$
\min_q \frac12\left\|\frac{q-q_{prev}}{v_{max}\Delta t}\right\|^2
+\frac12\left\|\frac{q-q_{prev}-v_{prev}\Delta t}{v_{max}\Delta t}\right\|^2,
$$

$$
q_{min}\le q\le q_{max},\quad |q-q_{prev}|\le v_{max}\Delta t,\quad
|q-q_{prev}-v_{prev}\Delta t|\le 250\Delta t^2,
$$

$$
\|p_{FK}(q)-p_{target}\|\le\epsilon_p,\qquad
\|\log(R_{target}^{T}R_{FK}(q))^\vee\|\le\epsilon_R.
$$

250 rad/s²는 기존 진단과 맞춘 **명령 생성용 시험 제약**, 제조사 사양이나
actuator 변경이 아니다. 엄격한 경우 `(0.1 mm, 0.001 rad)`, 완화한 경우
`(5 mm, 0.05 rad)`를 비교했다. 마지막 variant는 축 분리각 5° 이상 조건도 추가했다.
반올림 여유를 위해 solver 내부 pose 반경은 명시한 값의 0.999배이고,
판정·보고는 원래 기준을 사용한다. 모든 실행 maxiter=150, ftol=1e-9.

`optimizer_success`, 제약을 실제 만족하는 `feasible`, 둘 다 만족하는 `accepted`를
별도 기록했다. **실패 후에도 box 안의 최종 iterate를 다음 진단 seed로 이어서
실패 구간을 기록했다. 이것은 실제 actuator에 보낼 허용 명령이나 fallback이 아니다.**
실패 뒤의 수치는 성공한 replay 결과로 해석하면 안 된다.

### 배치 11 결과

| 방법 | max step [rad] | max 요구 속도 [rad/s] | max 요구 가속도 [rad/s²] | max FK 오차 [mm / deg] | 결과 |
|---|---:|---:|---:|---:|---|
| 기존 raw IK | .798175 | 95.781 | 5789.39 | 수치 오차 | 속도 초과 |
| 해석해 전체 branch, 같은 reset | .798175 | 95.781 | 5789.39 | 수치 오차 | 동일 급변 재현 |
| 해석해 전체 branch, 다른 reset, 작업대 검사 | .128090 | 15.371 | 514.92 | 수치 오차 | step 83.95% 감소, 여전히 속도 초과 |
| 해석해 전체 branch, 작업대 검사 생략 | .018901 | 2.268 | 110.60 | 수치 오차 | 작업대 관통 → 제외 |
| SQP 엄격, 같은 reset | .083333 | 10.000 | 250.00 | 7.785 / 1.727 | 제약 실패 iterate 포함; 미해결 |
| **SQP 완화, 같은 reset** | **.019973** | **2.397** | **73.33** | **4.995 / 2.862** | **148/148 solver+제약 통과, strict 불충족** |
| SQP 완화+축 분리 5°, 같은 reset | .083333 | 10.000 | 250.00 | 18.368 / 4.161 | 125/148 accepted; 나머지 실패 |
| SQP 엄격, 해석해가 선택한 reset | .083333 | 10.000 | 250.00 | 3.967 / .281 | 112/148 accepted; 미해결 |
| SQP 완화, 해석해가 선택한 reset | .083333 | 10.000 | 250.00 | 6.175 / 2.951 | 143/148 accepted; 미해결 |

해석해의 같은-reset 급변은 **명령 1.033333초**, 저장된 post-state 시각
1.041667초에 재현됐다. 자유 reset의 좌표 차이는 `results.json`에 저장했으며
기존과 동일 초기 관절 배치 실험으로 취급하지 않는다.

완화형 SQP의 wrist1/wrist3 max 요구 속도는 각각 **1.019 / 1.148 rad/s**,
max 요구 가속도는 **34.91 / 63.49 rad/s²**였다. 그러나 위치 오차 mean/P95/max는
**4.887 / 4.995 / 4.995 mm**: optimizer가 대부분 구간에서 허용 오차를 사용한다.
축 최소 분리각도 **0.784° → 0.064°**, weighted Jacobian σmin은
**.00618 → .000474**로 더 작아졌다. 즉 **작은 pose 오차로 다른 연속 관절 경로를
지나 급변을 완화한 것**이지 특이점에서 멀어진 것은 아니다.

엄격 SQP는 같은 reset에서 115/148 optimizer+제약 통과, 120/148 제약 만족이다.
최초 실제 pose 제약 위반은 명령 **1.000초 / frame 121**에서 .310 mm / .001089 rad.
그보다 앞의 frame 41은 제약을 만족하지만 line-search 수렴 실패로 기록됐다.
수치 수렴 실패와 기하학적 제약 실패를 혼동하지 않는다. 단일 시작점의 local SQP
실패만으로 허용 오차 범위 안에 해가 전혀 없다고 증명한 것은 아니다.

### 추가 배치 검증

| 배치 | 기존 max 속도 → 해석해 자유 reset+작업대 [rad/s] | 엄격 SQP 같은 reset: 제약 만족 / 전체 | 완화 SQP 같은 reset: accepted / 전체 | 완화 max 오차 [mm / deg] |
|---|---:|---:|---:|---:|
| 0 | 1.829 → 1.829 | 152/152 | 152/152 | 4.995 / 2.862 |
| 5 | 1.946 → 1.863 | 148/148 | 148/148 | 4.995 / 2.862 |
| 19 | 10.965 → 10.965 | 143/148 | 136/148 | 6.835 / 3.053 |

0/5의 엄격 SQP는 pose/속도/가속도 조건은 전부 만족했지만 각각 7건의
optimizer line-search 실패가 있어 완전한 solver 통과로 세지 않았다.
배치 19에서는 완화해도 실제 제약 위반이 남았다. 4개 궤적의 FK 결과이며
파지 성공률이나 일반적 해결 확률로 보고하지 않는다.

### 결론, 처리량과 한계

- **해석해로 바꾸면 계산은 빨라지지만 나쁜 기구학적 조건이 사라지지는 않는다.**
  배치11에서 전 branch 생성+기존 FK 검증 평균 .669 ms, P95 .923 ms.
  전체 경로 DP/작업대 검사 시간은 별도이고 online 120 Hz 보장을 뜻하지 않는다.
- 작업대 검사를 통과한 열거 그래프에서 exact-pose 최소 최대 속도가 15.371 rad/s다.
  따라서 이 그래프에서는 **원래 pose/시간을 정확히 유지하며 10 rad/s 이하로 만드는
  경로를 얻지 못한다**. 전체 물리 충돌이나 tolerance-aware global 최적화의 불가능
  증명은 아니다.
- 완화 SQP는 문제 배치의 급변을 97.50% 줄였지만 5 mm / 2.86° 오차를 사용하며,
  다른 배치에서는 실패했다. 실제 캔 파지 개선 여부는 미검증이다.
- 배치11 SQP solve P95: 엄격 23.86 ms, 완화 2.60 ms; max 111.90 / 2.98 ms.
  실행 환경에 따른 CPU 시간이며 hard real-time 보장은 없다.
- 다음 비교 후보는 앞 절의 **접근 yaw+초기 branch 선택**이다. 이번에는 yaw나
  정책을 변경하지 않았다. 두 새 solver 모두 진단 전용이며 기본 경로에 미연결이다.

### 재현 / 산출물 / 검증

```bash
bash scripts/compare_analytic_sqp.sh \
  --source outputs/diagnostics/ik120_improvement_20260907/old20_baseline \
  --episode 11 --out outputs/diagnostics/analytic_sqp_repeat/episode11
# --episode 0 / 5 / 19와 각각 다른 --out으로 추가 배치 재현.
./scripts/run_tests.sh
```

기존 output 디렉터리가 있으면 실행을 거절하므로 과거 로그를 덮어쓰지 않는다.
실행 결과: `outputs/diagnostics/analytic_sqp_20260915/episode{11,0,5,19}/`.

- `experiment.json`, `inputs.npz`: 고정된 설정, 실제 입력/시각, checkpoint/model/source hash.
- `all_branches.npz`, `branch_enumeration.json`: 전체 해, branch ID, frame offset, 연속해 예외.
- `results.json`, `<method>.npz`, `<sqp_method>_solver.json`: 경로, FK/속도/가속도,
  제약·수렴 실패 frame, 변경한 reset 좌표.
- [문제 배치 비교 그래프](../outputs/diagnostics/analytic_sqp_20260915/episode11/comparison.png).
- 새 코드: `tools/rb3_revo2_ik/analytic_branch_ik.py`, `sqp_pose_ik.py`,
  `tools/arm_diagnostics/compare_analytic_sqp.py`, root launcher.
- `tests/test_analytic_sqp_ik.py`: 12개 신규 검증. random FK/해석해 왕복,
  world/mount 회전, 모든 bounded 표현, exact/near singularity 구분,
  잘못된 모델/입력 거절, SQP 제약 및 실패 판정. 기존 166개 보존.
- 변경 전 **166개**, 변경 후 **178개 테스트 통과**; CLI help, shell syntax,
  `git diff --check` 통과. Isaac 실행이나 파지 재평가를 통과했다고 주장하지 않는다.

알고리즘 API 근거: [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).
- 실제 계산 시간: 배치11 17.2초, 별도 배치 각각 약 7–8초(CPU).
- 검증: 변경 전 158개 → 변경 후 **166개 unit regression 통과**. 새 테스트는
  30/120 Hz timestamp 구분, 누락 frame, 이름별 joint mapping, quaternion,
  raw angle jump, 실패/pose 오차 보존, reset 가속도 이력을 검사한다.
  네 실험의 입력 pose/손 명령은 원본과 배열 단위로 일치하며 model/config 및
  metadata/input hash도 확인했다. CLI help, shell syntax, `git diff --check` 통과.
