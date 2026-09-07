# 팔 속도 목표 비교: zero vs recorded v_path

2026-09-07 실제 Isaac GPU 실행. [이전 actuator 진단](ARM_ACTUATOR_DIAGNOSIS.md)의
후속 단일 변수 실험이다. 추가 튜닝이나 정상 rollout 동작 변경은 하지 않았다.

## 결론

캔 접촉 OFF에서 팔 속도 목표만 기록된 `v_path`로 바꾸면 손목 최대 위치 오차는
41.76%, 최대 회전 오차는 75.80% 감소했다. 특히 끝부분 wrist1/wrist3의 큰
관절 추종 지연이 거의 사라졌다. **움직이는 위치 목표에 속도 목표 0을 보내는
것이 이 실행의 추종 지연에 기여한다는 가설을 지지한다.** 모든 위치 오차나
물리 문제가 해결된 것은 아니며, 캔 접촉 ON에서의 파지 성공은 이번에 검증하지 않았다.

## 비교의 동일성 및 시간 기준

- 기존 episode 15의 108 physics commands, dt=1/120 s, 총 0.9 s.
- 초기 PhysX 이력을 보존하기 위한 기존 15-episode prefix를 양쪽 모두 동일하게
  실행했다. 측정 대상 108 step에서는 policy/IK를 호출하지 않았다.
- fresh zero 실행은 이전 OFF baseline의 전체 관절 위치/속도, wrist pose,
  object state와 **모든 sample에서 bit-for-bit 일치**했다.
- 위치 목표 17개, 손가락 속도 목표, feedforward effort, simulator에 실제 전달된
  위치 목표, 초기 상태, physics step, 명령/측정 시간, gains/limits 및 기록된
  물리 설정의 동일성을 분석 코드가 assert했다. 정책/IK/충돌/질량/관성 설정은 수정하지 않았다.
- OFF는 기존 방식 그대로 캔을 world X로 +10 m 옮기는 처리이다. 책상 접촉과
  self-collision 설정(false), 중력은 보존했다.
- 수정은 **이름으로 매핑한 팔 6개 속도 목표**뿐이다. 실제 native velocity setter
  전달값이 이전 OFF 로그의 같은 step `v_path`와 정확히 일치함을 검증했다.
- `v_path[k]=(q_cmd[k]-q_cmd[k-1])/physics_dt`. 첫 predecessor는 기록된
  `previous_accepted_q`; angle wrapping, 재보간, 필터, 스케일, 추가 clipping 없음.
  command 시각은 `(k-1)*dt`, state 시각은 `k*dt`이다.
- 손목 오차는 검증된 기존 FK로 계산한 **FK(q_cmd) vs PhysX runtime Revo2 base**
  (stage C), 관절 오차는 `q_cmd-q_actual`이다. 별도 reference-only IK 검증값이 아니다.

## 결과

전체 108 sample의 절대 오차 통계. 위치 단위 mm, 회전/관절 단위 rad.

| 지표 | 속도 목표 0: mean / p95 / max | v_path: mean / p95 / max | mean / max 감소 |
|---|---:|---:|---:|
| 손목 위치 [mm] | 16.554 / 45.147 / 51.234 | 12.916 / 25.918 / 29.839 | 21.98% / 41.76% |
| 손목 회전 [rad] | .134093 / .217411 / .328776 | .032795 / .068305 / .079557 | 75.54% / 75.80% |
| wrist1 [rad] | .071806 / .298089 / .411731 | .002545 / .004064 / .004554 | 96.46% / 98.89% |
| wrist3 [rad] | .149552 / .485191 / .722778 | .004372 / .012543 / .014654 | 97.08% / 97.97% |

첫 4 physics step을 수치상 startup으로 분리한 작업 구간 mean도 위치
16.960 → 13.299 mm, 회전 .138532 → .033837 rad로 개선됐다.
이 구분은 실제 접촉/파지 이벤트의 판정이 아니다.

| 원래 문제 시각 | 위치 [mm] zero → v_path | 회전 [rad] zero → v_path | wrist1 절대 오차 [rad] | wrist3 절대 오차 [rad] |
|---|---:|---:|---:|---:|
| .575 s | 50.781 → 25.159 | .165536 → .067696 | .016598 → .003462 | .072155 → .005260 |
| .900 s | 17.744 → 12.987 | .328776 → .033366 | .411731 → .003737 | .722778 → .003724 |

위치 최대 오차 발생 시각은 .566667 → .550 s, 회전은 .900 → .550 s.
.900 s에서 wrist1 실제 속도는 6.184 → 8.654 rad/s, wrist3는 -3.634 →
-8.819 rad/s로, 같은 step의 경로 속도 +8.627 / -8.826 rad/s에 가까워졌다.
그래프: [시간 비교](../outputs/diagnostics/arm_velocity_analysis_20260907/velocity_target_comparison.png).

## 진동·새 문제와 해석의 한계

- 발산, NaN/Inf, 관절 위치 limit 위반은 관측하지 않았다. 모든 팔 실제 속도는
  10 rad/s limit의 95% 미만이었다. 최소 위치 margin은 .22057 → .21349 rad.
- 그러나 **모든 시점의 위치 오차가 줄지는 않는다**. 43/108 sample에서 증가했고,
  가장 큰 악화는 .391667 s에서 4.965 → 10.589 mm (+5.625 mm)였다.
- wrist3에는 .316667 s 부근 짧은 속도 출렁임이 남았다. 당시 목표 +.3703 rad/s에
  실제 속도가 -1.0484, 다음 step +.9963 rad/s였고, .333333 s 관절 오차 .014654 rad였다.
  지속적 진동/발산으로 단정하지 않으며, 기존 zero에도 .333–.342 s 속도 출렁임이 있었다.
- 실제 속도 부호 전환 횟수(절댓값 .05 rad/s 이하 제외)는 wrist1 3 → 5,
  wrist3 3 → 9였다. 움직이는 경로의 방향 전환과 작은 과도응답이 섞인 수치이므로
  이것만으로 불안정성을 판정하지 않는다. 관절 오차 total variation은 각각
  .6517 → .0164 rad, 1.1768 → .0336 rad로 오히려 감소했다.
- wrist1 최대 실제 가속도는 51.72 → 276.13 rad/s², wrist3는 285.81 →
  287.33 rad/s². 가속도는 인접 physics sample의 실제 속도 차분이며,
  더 빠르게 경로를 따라가면서 wrist1 가속도도 증가했다. 가속도 limit 판정은 아니다.
- wrist3 **근사 effort clipping** sample은 3 → 7로 증가했다. 새 시각은
  .141667, .241667, .325, .775, .808333, .841667, .875 s.
  이는 implicit actuator의 보조 계산값일 뿐 solver drive torque 측정이 아니다.
  실제 torque saturation은 여전히 **UNKNOWN**이다.
- 이 실험은 짧은 0.9 s 움직이는 경로이며 종료 후 정지/수렴 구간을 추가하지 않았다.
  장기 안정성, 힘 여유, 접촉 ON 파지 성능은 결론 내릴 수 없다.

## 재현 명령과 산출물

프로젝트 루트에서 실행. 아래 `_repeat` 파일은 새 출력 경로이며, 이미 존재하면
덮어쓰지 않고 실패하므로 다시 실행할 때 새로운 이름을 사용한다.

```bash
common=(
  --sequence 20200709_143747_left
  --checkpoint logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/model_4999.pt
  --num_envs 1 --headless --random-placement
  --arm-contact-replay outputs/diagnostics/arm_contact_source_20260907.jsonl
  --arm-contact-episode 15 --arm-contact-condition absent
  --arm-actuator-diagnostic
)
./scripts/rl.sh play-arm "${common[@]}" \
  --arm-contact-output outputs/diagnostics/arm_velocity_zero_repeat.jsonl
./scripts/rl.sh play-arm "${common[@]}" \
  --arm-contact-output outputs/diagnostics/arm_velocity_path_repeat.jsonl \
  --arm-velocity-path-trace outputs/diagnostics/arm_actuator_absent_20260907.jsonl
bash scripts/analyze_arm_velocity.sh \
  outputs/diagnostics/arm_velocity_zero_repeat.jsonl \
  outputs/diagnostics/arm_velocity_path_repeat.jsonl \
  --output-dir outputs/diagnostics/arm_velocity_analysis_repeat
```

실제 완료된 실행은 위 명령의 `_repeat`를 `_20260907`로 바꾼 것이다.
로그는 `outputs/diagnostics/arm_velocity_{zero,path}_20260907.jsonl`,
분석은 `outputs/diagnostics/arm_velocity_analysis_20260907/summary.json`과 PNG.
console 로그는 `/tmp/arm_velocity_{zero,path,analysis}_20260907.log`.

## 변경 범위와 검증

- `regrind/scripts/rsl_rl/play.py`: 기본 None인 진단 전용
  `--arm-velocity-path-trace`; OFF recorded replay + actuator diagnostic에서만 허용.
- `tools/rb3_revo2_ik/replay_arm_contact.py`: 팔 이름 매핑 후 동일 step 속도만 교체.
- `tools/rb3_revo2_ik/analyze_arm_velocity.py`: trace 일치 검증, 분석/그래프.
- `scripts/analyze_arm_velocity.sh`: 유지관리용 분석 진입점.
- `tests/test_arm_velocity_analysis.py`: timestep, 이름, finger command,
  미분값, baseline 구분, deadband 검사 7개.

Isaac 양쪽 실행 완료, 비교 assertion 통과, 기존 포함 **54/54 테스트 통과**,
Python compile 및 `git diff --check` 통과. 정상 training/evaluation의 기본
속도 목표 동작은 변경하지 않았고, 추가 실험이나 튜닝은 수행하지 않았다.

후속으로 승인된 [접촉 ON 및 live policy 20개 배치 비교](ARM_VELOCITY_CONTACT_POLICY_VALIDATION.md)가
완료됐다. 추종은 개선됐지만 해당 정책의 파지 결과는 악화되어 기본 zero를 유지한다.
그 후속 구현에서 fixed replay 옵션은 같은 조건의 ON 로그도 허용하도록 확장됐다.
