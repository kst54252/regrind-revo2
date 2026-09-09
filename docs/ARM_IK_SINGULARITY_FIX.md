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
