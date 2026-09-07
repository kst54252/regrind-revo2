# 120 Hz mounted-policy improvement — completed, candidate not promoted

Historical experiment report: rejected wrist3 gain tables/config were retired
during [layout cleanup](cleanup-plan.md#readable-layout-cleanup). Their exact
contents remain in Git commit `531b9c8`. Candidate commands below require that
historical revision; they are not supported commands in the current checkout.
Current bounded-IK execution is documented in [ARM_IK_SINGULARITY_FIX](ARM_IK_SINGULARITY_FIX.md).

**결론:** 120 Hz를 유지한 3개 gain 후보와 실제 40개 초기 배치 비교를
완료했다. wrist3 오차 감소는 확인했지만 손목 위치 정밀도는 개선되지 않았고,
held-out 16번에서 파지 유지가 악화되어 후보를 기본 실행으로 채택하지 않았다.
지원 설정은 기존 `rb3_transfer_recovery_candidate.json` + fast IK 그대로다.
마지막 구간의 IK 관절 속도 요구/제한도 확인했다. 새 알고리즘이나 네 번째
튜닝 후보로 범위를 확장하지 않았다.

## Scope and frozen baseline

Baseline is **not** the original zero-velocity arm controller: it is
`scripts/play_arm_candidate.sh`: simple mounted bridge, warm-first existing IK at
120 Hz, c3 gains, physics-rate wrist response tau=0.1 s and path velocity
targets. Policy is 30 Hz, physics is 120 Hz. The 30 Hz IK option remains off.
Normal launch/training defaults are not promoted or modified.

- Checkpoint: `logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt`.
- Reference: `outputs/isaac/dexycb/20200709_143747_left/rb3_revo2_reference_stable.h5`.
- Baseline config: `config/experiments/rb3_transfer_recovery_candidate.json`.
- Single diagnostic root: `outputs/diagnostics/ik120_improvement_20260907/`.
- Existing dirty files and patch preserved in its `preserved_before/` directory.
- Tuning: first 3 original placements in `arm_policy_velocity_zero20_20260907.jsonl`.
- Final comparison: original20 and distinct `arm_transfer_recovery/heldout_initial_states_v2.jsonl`;
  select/freeze a candidate before running the latter.

Preserve checkpoint, normalization, action/observation semantics, reference,
phase/timesteps, hand commands, native dynamics/drive type, limits and all
collisions. Only explicitly selected experimental arm changes are permitted.
No verified implicit drive-only effort API: saturation remains **UNKNOWN**.

## Criteria fixed before candidate selection

Existing task success/termination is authoritative. Separately report the
existing diagnostic lift proxy: final 0.2 s minimum lift >=0.1 m and >=80%
can–robot contact samples (>0.01 N). This proxy is not the task definition.
Reject a candidate that loses paired grasps, introduces nonfinite/position-limit
violations or worsens command/actual acceleration spikes. Inspect contact
onset and terminal state before autoreset. No post-hoc time shifting.

A: IK input versus FK(raw solve); B: FK(accepted IK) versus FK(final command);
C: FK(final command) versus FK(actual joints); D: FK(actual) versus runtime
wrist; E: raw policy equilibrium versus runtime wrist. Distinguish a failed
raw solve from the previous accepted target held on failure.

## Progress / resume point

- Complete: unchanged baseline, native instrumentation noninterference,
  policy-free tests, three-candidate screening, frozen old20 + held-out20 pairs,
  floating3 and reduced-logging timing controls, final smoke and regressions.
- No running continuation or further tuning. New work requires a new experiment.

### Diagnosis before tuning

3/3 live baseline grasps passed. Additional instrumentation reproduced all 448
physics samples bit-for-bit (commands/actions, actual joints/velocities,
wrist and object). A <=0.0001 mm, B <=3.4e-7 mm; no mount/FK changes indicated.
Before contact, C max=0.74–1.03 mm; after contact max=2.92–4.70 mm.
Wrist3 error grows from <=0.00167 rad before contact to 0.0225–0.0248 rad
after contact. Single-joint, analytical no-can test passes: static max
0.653 mm / 0.110 degrees; motion max 0.655 mm / 0.112 degrees, no reported
contact or limit violations. This does not prove adequate contact response.

Three hypotheses at most:
1. Local wrist3 contact-response contributes to the residual C peaks (supported
   by contact-aligned errors; causality not yet established).
2. Raw equilibrium E includes deliberate 0.1 s response shaping, not IK failure
   (verified distinct targets). Leave tau and policy timing unchanged.
3. Unverified drive saturation/inertial or contact effects may constrain response
   (unresolved; do not label approximate PD effort a torque measurement).

Bounded candidates in `config/experiments/rb3_ik120_gain_candidates.json` alter
**only wrist3**: c1 Kp/Kd=500/30, c2=500/45, c3=350/30 versus 250/20.
All other arm/hand gains, native limits and dynamics remain fixed. No fourth
candidate or automatic gain search. Select on original placements 0–2 only.

`baseline_no_can` live-policy trial terminated after four timesteps per placement
because the relocated can triggered existing deviation conditions. It is NOT
a valid no-contact tracking comparison; preserved as an unsuccessful diagnostic.
Use the policy-free single-joint/coordinated/AB runs instead; do not change
termination conditions to make this live-policy trial run longer.

### Frozen selection before final paired evaluation

All three candidates retained 3/3 task and lift-proxy successes, but none met
all improvement criteria: C mean 0.838 mm became 0.845/0.853/0.855 mm.
c1 reduced wrist3 mean 0.003275→0.002541 rad and max 0.02479→0.02043 rad,
but increased commanded peak acceleration and some other joints' acceleration.
**It is only the best local wrist3 candidate, not an accepted replacement.**
Freeze c1 in `config/experiments/rb3_ik120_wrist3_candidate.json` for final
paired evaluation; no additional candidates, retuning or held-out selection.
Baseline remains recommended unless the full comparison supports a safe benefit.

## 실제 비교 결과 / 채택 판단

두 bank는 각각 20개 서로 다른 위치이며 bank 간 최소 거리는 8.13 mm다.
동일 checkpoint/reference, 실제 초기 arm/hand q,dq, wrist, object pose/velocity,
phase, reset action/response buffers와 첫 actor observation/action까지 일치했다.
정책은 이후 각 실행의 실제 관측으로 추론했다. 기록 행동 재생 결과가 아니다.

| 지표 | 기존20 baseline → 후보 | 별도20 baseline → 후보 |
|---|---:|---:|
| 기존 task success | 20/20 → 20/20 | 20/20 → 20/20 |
| 마지막 lift/contact 보조 기준 | 20/20 → 20/20 | **20/20 → 19/20** |
| C 위치 mean [mm] | 0.936 → 0.943 | 0.868 → 0.875 |
| C 위치 P95 [mm] | 2.806 → 2.845 | 2.607 → 2.677 |
| C 위치 max [mm] | 5.852 → 5.817 | 5.998 → 6.034 |
| C 회전 mean [rad] | .004219 → .003665 | .004481 → .003829 |
| C 회전 max [rad] | .02812 → .02449 | .02963 → .02823 |
| wrist3 관절오차 mean [rad] | .003124 → .002404 | .003496 → .002645 |
| wrist3 관절오차 max [rad] | .02573 → .02176 | .03039 → .02411 |
| 실제 팔 가속도 max [rad/s²] | 636.68 → 539.97 | **729.27 → 743.57** |
| 명령 경로 가속도 max [rad/s²] | 640.05 → 532.10 | **726.23 → 782.81** |
| B 위치 max [mm] | 16.845 → 18.064 | 10.784 → 14.584 |

Task success→failure / failure→success 배치는 모두 없다. 그러나 held-out
**16번**에서 보조 기준 성공→실패가 발생했다: 마지막 0.2 s 접촉 비율
100%→50%, 최종 lift 0.235→0.168 m. 해당 배치의 C mean은
0.914→0.886 mm로 줄었는데 파지는 나빠졌다. 기존 종료 시점의 상태를 사용했고
autoreset 후 상태나 녹화 종료 후 추가 명령을 평가에 섞지 않았다.
따라서 `rb3_ik120_wrist3_candidate.json`은 **reproduction-only / rejected**다.
40개 결과를 일반적인 성공률 보장으로 해석하지 않는다.

## 추가로 확인된 제한: 저속 IK 성공과 빠른 실행 가능성은 다름

3개 tuning 배치에서는 B가 수치 오차 수준이었지만 전체 평가에서 다른
후반 구간이 드러났다. 이를 보고 후보를 다시 튜닝하지 않았다.

- 기존20 배치11, **1.041667 s**: IK 입력 위치 변화 2.317 mm,
  회전 변화 .002692 rad에 wrist1/wrist3 IK 변화가 각각
  **-.79817 / +.79750 rad**였다. 한 physics step의 요구 속도는 약
  **95.8 rad/s**, 유지한 속도 제한은 10 rad/s다.
- 이때 wrist2=.015789 rad. 기존 IK Jacobian(위치 weight=10, 회전=1)의
  최소/최대 singular value는 .00618/6.068이다. 스케일 의존 수치이며
  제조사 기준이 아니다. 작은 pose 변화가 큰 wrist 관절 변화로 증폭되는
  특이점 근처의 민감성과 일치한다. 임의의 angle wrapping은 하지 않았다.
- 배치11 **1.141667 s**에는 IK 위치 오차는 수치 수준이어도
  B=16.845 mm / .11936 rad. wrist1/wrist3 명령 속도는 -10/+10 rad/s로
  제한되어 이전 목표를 따라잡는 중이다. gain을 높여도 이 요구 속도 자체가
  없어지지는 않는다. 전체 old20의 명령 slew 제한은 81 timestep이다.
- 별도16번도 IK 최대 step .24945 rad→후보 .34373 rad,
  B 최대 10.784→14.584 mm. 접촉 손실과 동반되지만 이것만으로 유일한
  파지 실패 원인이라고 단정하지 않는다.

A(raw float64 solver output)는 수치 수준, accepted float32 IK goal의 FK
오차도 <0.1 µm다. D 최대 <0.7 µm / 1.8e-6 rad이며 IK failure는 0이다.
이는 **장착 frame 불일치나 IK 수렴 실패를 지지하지 않는다**. 반대로
모든 IK pose를 현재 속도 제한 안에서 시간 맞춰 실행할 수 있다는 뜻도 아니다.
E mean은 약 25.6 mm: 기존 .1 s wrist response를 포함한 raw equilibrium
오차이므로 C와 혼동하면 안 된다.

다음 단일 비교 실험 후보는 **120 Hz와 나머지 설정을 유지하면서 특이점 근처
IK 해의 속도 실현 가능성을 처리하는 opt-in 경로**다. 단순 gain 추가 증대나
목표 도착까지 정책 phase 정지는 이번 결과로 정당화되지 않는다. 새 방식은
원래 pose 정확도/phase/파지에 미치는 영향을 별도로 검증해야 한다.

## 정책 없는 검증 / 정확히 바꾼 설정

| 고정 시험 | baseline → 후보 |
|---|---:|
| 6개 단일 관절 .05 rad out/back, quintic 1 s | baseline max .655 mm, .1124°, contact 0 |
| A/B/C 저속 coordinated motion P95 | .74325 → .74326 mm |
| A/B/C static final 1 s max | .74463 → .74461 mm |
| 별도 D/A motion P95 | .772689 → .772678 mm |
| 별도 D/A static max | .773039 → .773033 mm |
| 별도 D/A motion max 회전 | .10726° → .09025° |
| 기존 reference 38 pose → IK, 무정책/캔 이동 | baseline C mean .575 / max .890 mm |

저속/정지 engineering 기준(1 mm, .5°)과 안정성 검사는 모두 통과했다.
실제 can contact가 있는 원래 속도의 policy 실행에 대한 보장은 아니다.
캔만 초기화 때 X+10 m 이동했고 로봇/책상/self collision은 유지했다.
단일 관절 시험은 analytical max .09375 rad/s, .28868 rad/s²이며 ZOH 샘플을
반복해 느린 시험으로 가장하지 않았다. 초기 1회 외 state overwrite 없음.
정지창, 6초 quintic, 3초 hold 및 별도 자세는 기존 고정 benchmark를 재사용했다.

| ARM joint | Kp baseline→c1 [N·m/rad] | Kd baseline→c1 [N·m·s/rad] | effort / speed limit |
|---|---:|---:|---:|
| base | 700→700 | 35→35 | 10 N·m / 10 rad/s |
| shoulder | 20000→20000 | 260→260 | 100 / 10 |
| elbow | 12000→12000 | 140→140 | 100 / 10 |
| wrist1 | 900→900 | 45→45 | 100 / 10 |
| wrist2 | 2400→2400 | 70→70 | 100 / 10 |
| wrist3 | **250→500** | **20→30** | 10 / 10 |

Native runtime verified `ImplicitActuator`, force drive type=1 on all ARM joints.
Isaac Sim6.0.1.0 / isaaclab distribution13.3.0 / isaaclab_physx3.1.1,
PhysX CUDA backend. Config table is applied only through diagnostic evaluator
overrides, not task/asset defaults. Hand gains/coupling are unchanged.
Explicit native arm feedforward is zero: no added gravity compensation.
Low-speed pairs verified native masses/inertias, gravity flags, drive type,
effort/speed/position limits and simulator config identical.

All live ARM position violations=0 and no actual ARM speed exceedance above
1e-4 rad/s numerical allowance; near-limit intervals are recorded separately.
**Not all hand limit checks pass**: existing index leader/follower soft-limit
excursions remain (held-out baseline max .001849 rad, candidate .001830 rad).
These were not fixed or hidden in an arm-only task. No NaN/Inf or IK failures.
There is no verified solver drive torque; saturation remains UNKNOWN. Offline
Kp(qcmd−qpre), Kd(vcmd−vpre) are labelled approximate force-PD terms, never
compared as measured implicit solver effort. Native explicit effort is separately
captured before physics. [Installed API provenance](ARM_ACTUATOR_DIAGNOSIS.md).

## 실행 경로 / timing / evidence

- `evaluate_mounted_interface.py`: `snapshot`/`verify_initial` → frozen adapter
  in live loop → `wrapper.step` → action manager → native scene write → physics
  → wrapped scene `update`; termination snapshot occurs before reset.
- Existing `actions.py:SE3ImpedanceActionTerm.process_actions` remains the decoder.
  `simple_mounted_interface.py:process_actions/apply_actions/solve` retain the
  .1 s response, 120 Hz IK, bounded q command and final-q backward-difference
  velocity. `reset_from_reference` clears response/command history.
- `recovery_probe.py:RecoveryProbe` checks joint-name-mapped staged targets
  against each native q/v/explicit-effort setter; exactly one call, no overwrite.
- PhysX actual wrist getters remain
  `rb3_revo2_commands.py:current_hand_wrist_pos/current_hand_wrist_quat`.
  World/env-local origin=[0,0,0], mounted Revo2 base, quaternion XYZW.
- `physics.jsonl`: command at k·dt, actual after physics at (k+1)·dt,
  raw solve, accepted q, native q/v/effort, actual q/dq/wrist/object, A–E,
  contact and termination. Approximate-PD summaries reuse its pre-step state.
- `metadata.json`, `policy.json`, `realtime.json`: source hashes, native settings,
  actual resets, policy observation/action, frozen check and timing.
- `ik120_summary.json`: per-joint/per-episode/region mean/P95/max, peaks/times,
  limit checks, large IK steps/Jacobian sensitivity; `ik120_ep16.png` shows the
  held-out proxy regression. Root `paired_comparison.png`/`ik120_comparison.json`
  summarize the four paired runs.

Fresh floating3 baseline passed task and lift proxy 3/3; common wrist/hand/object
initial states matched arm within 2 µm. Floating E mean=29.63 mm despite successful
grasp, reinforcing that its equilibrium is not an achieved motion target.
Floating/root inertia differences remain; no claim of identical embodiments.

Reduced-logging headless timing: baseline/candidate sim/wall **.601/.598×**,
IK mean 1.676/1.677 ms, policy inference .659/.650 ms,
physics step 8.879/8.969 ms, scene update .158/.162 ms.
This is not GUI timing or a 1× guarantee. Full trace is slower and includes
offline FK/JSON overhead inside scene-update measurements. IK is nested in
root-apply: do not add these columns twice. Timers measure Python-call wall
time, not isolated CUDA kernel time; no extra GPU synchronization was inserted.
No dt changes, skipped steps or
post-hoc trajectory time shifts. Final smoke verified 152 IK solves for
152 physics steps, `ik_update_dt=1/120`; policy stays 1/30.

## Historical reproduction (retired candidate requires revision 531b9c8)

From repository root; choose a **new** output directory (existing logs protected):

```bash
# Supported 120 Hz GUI baseline — NOT the rejected wrist3 candidate.
bash scripts/play_arm_candidate.sh outputs/diagnostics/ik120_repeat_gui

CKPT=logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt
DIAG_OUT=outputs/diagnostics/ik120_repeat
for bank in old20 held20; do
  STATES=outputs/diagnostics/arm_policy_velocity_zero20_20260907.jsonl
  if [ "$bank" = held20 ]; then
    STATES=outputs/diagnostics/arm_transfer_recovery/heldout_initial_states_v2.jsonl
  fi
  for variant in baseline candidate; do
    CFG=config/experiments/rb3_transfer_recovery_candidate.json
    if [ "$variant" = candidate ]; then
      CFG=config/experiments/rb3_ik120_wrist3_candidate.json
    fi
    bash scripts/evaluate_mounted_interface.sh --mode simple --episodes 20 \
      --headless --fast-ik --recovery-capture --states "$STATES" \
      --checkpoint "$CKPT" --transfer-config "$CFG" \
      --output "$DIAG_OUT/${bank}_${variant}" || break 2
  done
done
/home/wanjunkim/IsaacLab/.venv/bin/python -m tools.arm_diagnostics.analyze_ik_tracking \
  "$DIAG_OUT" old20_baseline old20_candidate held20_baseline held20_candidate \
  --paired --plot-episode 16

bash scripts/benchmark_arm_precision.sh --candidate c3 --headless \
  --move-can-away --single-joints --output "$DIAG_OUT/single_baseline"
bash scripts/benchmark_arm_precision.sh --candidate c1 \
  --candidates config/experiments/rb3_ik120_gain_candidates.json \
  --headless --move-can-away --held-out --output "$DIAG_OUT/precision_held_candidate"
# Omit --single-joints/--held-out for the existing A/B/C coordinated benchmark.
bash scripts/evaluate_mounted_interface.sh --mode simple --stage ab --headless \
  --fast-ik --recovery-capture --arm-gains-key c3 --arm-response-physics \
  --response-tau .1 --arm-velocity-path --checkpoint "$CKPT" \
  --output "$DIAG_OUT/reference_ab"
./scripts/run_tests.sh
```

Changed: diagnostic evaluator, existing precision trajectory/benchmark, new
offline analyzer, two opt-in gain configs, precision/analysis tests and this
report/index link. No current-task edit to action decoder, IK solver,
`SimpleMountedWrist`, asset, normal training/play config or checkpoint.
The pre-existing 30 Hz option and other dirty work remain intact.

Regression baseline 102 tests → final **106 passed**, shell syntax and diff check.
Final baseline and candidate one-placement smoke both reproduced their earlier
152-sample joint/velocity/wrist/object/action/command traces **bit-for-bit**.
No commit/push or previous log overwrite. To keep/restore supported behavior,
use `play_arm_candidate.sh` without an extra transfer-config override; no source
revert, asset edit or checkpoint rollback is needed.

See [execution index](../scripts/README.md), [transfer recovery](ARM_TRANSFER_RECOVERY.md),
[actuator provenance](ARM_ACTUATOR_DIAGNOSIS.md), and
[real-time execution](ARM_REALTIME_EXECUTION.md). Historical success is not a
substitute for current comparison results.
