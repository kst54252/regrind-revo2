# RB3-730 + Revo2 제어 과정: 초기 RL부터 현재 실행까지

작성일: 2026-09-10. 확인한 코드 기준: `e452939`.

이 문서는 프로젝트에서 **팔을 어떻게 움직였고, 왜 floating-hand와 달랐으며,
어떤 진단·수정을 거쳐 현재 방식에 도달했는지** 설명하는 한국어 해설이다.
새 실험 보고서가 아니라, 현재 코드와 기존 실행 보고서를 연결한 정리다.
이번 문서 작성에서 정책·제어기·물리 설정을 변경하거나 실험을 재실행하지 않았다.

과거 보고서의 “기본값 유지”, “opt-in”은 해당 실험 당시의 상태다.
**현재 기본값은 아래 1절과 12절**을 기준으로 읽는다. 날짜가 비슷한 실험도
체크포인트, 제어기, 접촉 설정, 오차 정의가 다르면 같은 비교가 아니다.

## 읽는 순서

- 전체 구조를 먼저 이해하려면: 1–6절.
- 오차를 줄이기 위해 무엇을 했는지: 7–10절.
- 지금 무엇이 기본이고 어떤 한계가 남았는지: 11–14절.
- 코드와 상세 근거를 찾으려면: 15절.

## 1. 먼저 결론: 현재 팔은 무엇으로 제어하는가?

현재 기본 실행은 **floating-hand에서 학습한 정책 + 손목 목표를 팔 관절 목표로
바꾸는 IK + 실제 관절 actuator**의 조합이다.

```text
실제 mounted 손목·손가락·캔 상태
        ↓
학습 당시 형식의 observation + reference + phase
        ↓
기존 floating RL 정책, 30 Hz
        ↓
손목 위치/회전 residual 6개 + Revo2 손가락 residual 6개
        ├─ 손목: reference 결합 → 응답 필터 → IK → 팔 관절 목표
        └─ 손가락: reference 결합 → 6 leader 목표 → 5 mimic 목표
                                      ↓
                          팔·손 actuator와 PhysX, 120 Hz
                                      ↓
                              실제 상태를 다시 관측
```

현재 `scripts/rl.sh play-arm`의 정책은 **원본 10,000회 floating 모델**이다.
팔 제어에는 기존 영상에서 검증한 `c3` gains, 0.1초 응답 필터,
120 Hz IK, 최종 관절 목표의 차분 속도를 사용한다. 손끝에는 고무를 근사한 접촉 설정을 적용한다.

팔을 연결한 환경에서 추가로 한 시간 파인튜닝도 실행했다. 그러나 동일 배치 40개에서
파지 결과는 그대로이고 일부 추종오차가 커져, **그 모델로 기본 정책을 교체하지 않았다.**

“기본 제어기 선택”과 “기본 정책 체크포인트 선택”은 서로 다른 결정이다.

## 2. 이름이 비슷하지만 다른 네 가지 구조

| 구조 | RL이 내보내는 값 | 팔 관절각을 결정하는 방법 | 현재 의미 |
|---|---|---|---|
| 초기 결합형 RL | RB3 관절 residual 6 + Revo2 관절 residual 6 | 미리 만든 팔 reference 관절각에 RL 보정값을 더함 | `--legacy-arm-rl`로 보존한 별도 구조 |
| Reference/rollout의 오프라인 IK | RL 없음, 또는 이미 저장된 RL rollout | 저장된 손목 pose 전체를 사전에 IK 변환 | 궤적 변환·재생용 |
| Floating 정책의 온라인 팔 실행 | 손목 SE(3) residual 6 + Revo2 residual 6 | 실행 중 손목 목표를 매번 IK로 변환 | 현재 기본 실행 |
| Mounted 환경 파인튜닝 | 위 floating 정책과 같은 12차원 | 동일한 온라인 IK/actuator 경로 | 팔 환경의 실제 응답으로 같은 정책을 추가 학습 |

모두 “12차원”이라고 부를 수 있지만 **앞의 6개 값의 단위와 의미가 다르다.**
초기 결합형의 앞 6개는 관절각 보정이고, floating 방식의 앞 6개는
손목 위치 3개와 회전 3개의 보정이다. 서로 체크포인트를 그대로 바꿔 끼우는 관계가 아니다.

또한 “legacy arm”이라는 실험 모드는 주로 예전 `RB3WristIKAction` 실행을 뜻한다.
그것이 곧 `--legacy-arm-rl`의 **관절 residual 정책**을 뜻하는 것은 아니다.

## 3. RL에 넣기 전: 사람 동작을 로봇 reference로 만드는 과정

### 3.1 데이터와 retargeting

DexYCB의 두 번째 카메라 시점 데이터를 읽고, 왼손 동작은 오른손 convention으로
전처리한다. 현재 대표 sequence는 `20200709_143747_left`이다.

사람의 MANO 21점과 물체 표면점의 관계를 REGRIND의 Interaction Mesh/Laplacian
retargeting에 사용한다. 로봇 손목 pose와 Revo2의 **6개 독립 관절**을 조정하면
FK를 통해 로봇의 semantic keypoint 21개가 나온다.

```text
사람 손 21점 + 물체 표면점
    → 손목 pose와 손 관절을 최적화
    → wrist_pos, wrist_quat, revo2_joints
    → FK로 Revo2 semantic keypoints 21개 계산
```

21점은 서로 독립적으로 움직이는 제어변수가 아니다. 손의 기구학과 mimic/coupling을
따르는 결과다. 이후 RL의 fingertip 관측에는 semantic 21점 전체가 아니라
**실제 5개 손끝 링크**를 사용한다.

### 3.2 좌표 정렬과 reference 생성

Camera 좌표계에서 Isaac workcell 좌표계로 옮길 때 손목·물체·MANO에 같은 rigid
transform을 적용한다. 캔 mesh를 기준으로 바닥을 정렬하며 현재 workcell의
테이블 윗면은 world `Z=0`이다. 테이블 “높이 72 cm”와 “world Z=0”은 모순이 아니다.
테이블 형상의 높이와 좌표계 원점은 다른 개념이다.

초기의 위아래 반전, 물체가 Z 대신 X로 움직이는 문제는 이 좌표 convention과
물체 자세의 문제 영역이었다. 이후 확인된 **actuator 추종 지연**과는 구분해야 한다.
좌표를 다시 뒤집거나 mount 오프셋을 추가하는 것으로 모든 후속 오차가 해결되지는 않는다.

오프라인 IK를 거치면 다음 reference를 저장한다.

```text
rb3_joints       (T, 6)
revo2_joints     (T, 6)
reference_joints (T, 12) = [RB3 6개, Revo2 6개]
wrist/object pose와 사람 손 좌표 등 부가 정보
```

원본 데이터 quaternion은 메타데이터의 convention을 따르고,
현재 온라인 IK/PhysX 제어 경로의 quaternion은 **xyzw**다. 모든 파일을 무조건 같은
순서로 읽지 않는다. 전처리·좌표계 상세는 [DATA_PIPELINE.md](DATA_PIPELINE.md) 참조.

## 4. 맨 처음: 팔과 손을 함께 넣은 RL

초기 결합형 환경에서는 미리 계산한 12-DoF reference를 기준으로
RL이 팔 6관절과 손 6관절에 각각 보정을 더하는 구조를 만들었다.

```text
q_arm_target  = q_arm_reference  + arm_scale  × clip(a_arm)
q_hand_target = q_hand_reference + hand_scale × clip(a_hand)
```

남아 있는 결합형 config의 residual scale은 arm `0.05 rad`, hand `0.15 rad`다.
이는 해당 구조의 설정이며 현재 floating 손목 residual scale과 다르다.
결과를 joint limit 안으로 제한한 뒤 actuator에 전달하고,
손의 mimic 관절 목표는 leader에서 파생한다.

이 구조에서 IK는 주로 **reference 준비 단계**에 있다. 온라인 정책의 앞 6개가
손목 pose가 아니라 팔 관절 보정이므로, floating 정책처럼 매 행동의 손목 residual을
IK에 넣는 구조와 구별된다.

PPO, 관측·보상·RSI·randomization·curriculum 연결은 단계적으로 구현했다.
그러나 현재 이 초기 결합형 경로를 주력으로 사용하지 않으며,
최근 floating/arm 비교 성적을 이 초기 정책의 성적으로 해석해서는 안 된다.
현재 코드가 남아 있다는 사실과 최근 학습 품질이 검증됐다는 사실도 다르다.

구현: `config/rb3_revo2/rb3_revo2_tuna_env_cfg.py::ActionsCfg`,
`mdp/rb3_revo2_actions.py::RB3Revo2ResidualJointPositionAction`.
여기서 `config/`, `mdp/`는 15절의 task package 기준 경로다.

## 5. Floating-hand RL로 바꾸면 무엇이 달라지는가?

### 5.1 정책이 내보내는 것은 손목 pose 보정과 손가락 보정

Floating 환경에는 RB3 팔이 없고 Revo2가 공간에서 움직인다.
정책 action은 다음 12개 값이다.

```text
a[0:3]   : 손목 위치 residual
a[3:6]   : 손목 회전벡터 residual
a[6:12]  : Revo2 독립 관절 6개의 residual
```

30 Hz control 기준으로 원래 decoder가 하는 처리는 다음과 같다.

```text
raw action → [-1, 1] clipping → scale → reference와 한 번 결합

p_target = p_reference + (1/30 m) × a_position
R_target = Exp((3.2/30 rad) × a_rotation) × R_reference
q_hand_target = q_hand_reference + (3.2/30 rad) × a_hand
```

회전은 Euler 세 값을 단순히 더하는 것이 아니라 회전벡터를 회전으로 바꿔
reference에 왼쪽에서 곱한다. 손 관절은 이어서 위치 한계를 적용한다.
Residual을 두 번 더하지 않고, 팔에 연결해도 앞 6개를 RB3 관절각으로 치환하지 않는다.

### 5.2 “6D pose를 줬다”는 말에서 가장 중요했던 오해

Floating 제어기는 손을 `p_target, R_target`으로 순간이동시키지 않는다.
목표 pose와 실제 pose 차이를 이용해 **Cartesian 힘과 토크**를 만드는 PD 제어다.
개념적으로는 다음과 같다.

```text
F ≈ Kp_pos × (p_target - p_actual) - Kd_pos × v_actual
τ ≈ Kp_rot × rotation_error       - Kd_rot × ω_actual
```

진단에 사용한 floating 설정은 위치 Kp/Kd `300/30`, 회전 `3/0.3`이었다.
손 로봇의 중력 비활성 설정도 mounted 팔과 다른 조건이었다.
캔의 중력까지 없다는 뜻은 아니다.

따라서 정책이 학습한 것은 **이 물리 응답을 거쳐 잘 잡히는 목표 평형 자세**다.
평형 자세와 실제 손목 사이에 지연과 오차가 있는 상태에서도 파지가 성공할 수 있다.

```text
정책의 손목 목표 = 제어기를 끌어가는 평형점
정책의 손목 목표 ≠ floating 손목이 그 시각에 실제로 있었던 위치
```

이 목표를 팔 IK로 정확하게 실현하는 것과 floating의 실제 동작을 똑같이 재현하는 것은
다른 문제다. [응답 차이 진단](MOUNTED_RESPONSE_COMPARISON.md)이 이 점을 실제 로그로 확인했다.

### 5.3 실제 팔 상태를 어떻게 RL에 넣는가?

현재는 별도 floating simulation을 같이 돌려 가짜 관측을 만드는 방식이 아니다.
Mounted Revo2의 **실제 손목·손가락·캔 상태**를 기존 observation 함수에 넣는다.

Actor 67차원 구성:

| 항목 | 차원 |
|---|---:|
| 실제 물체 위치 / 6D 회전 표현 | 3 + 6 |
| 실제 손목 위치 / 회전의 2-frame history | 6 + 12 |
| 6 leader 관절 위치의 2-frame history | 12 |
| 직전 action | 12 |
| reference phase | 1 |
| reference 손목 위치·회전 | 9 |
| reference 손 관절 | 6 |

Critic 94차원은 여기에 물체 선·각속도 6, 실제 손끝 위치 15, leader 속도 6을 더한다.
위치에서는 env origin과 배치 offset을 기존 규칙대로 처리한다.
평가할 때 모델 가중치와 observation normalizer는 동결한다.

같은 초기 상태에서는 같은 관측·행동을 만드는지 확인했지만,
진행 중 팔과 floating의 물리 응답이 달라지면 관측이 달라지고 다음 행동도 달라진다.
이는 closed-loop 정책의 정상적인 성질이다.

## 6. “그냥 IK로 한다”는 방법과 실제 팔 제어

### 6.1 IK의 역할과 하지 않는 일

FK는 팔 관절각으로 손목 pose를 계산한다. IK는 그 반대다.

```text
FK(q_arm) → mounted Revo2 base pose
IK(desired Revo2 base pose) → q_arm_goal
```

기존 IK는 `RB3730Kinematics.inverse()`의 SciPy bounded nonlinear least-squares다.
위치 오차에 weight를 주고 회전 오차와 함께 줄이며 모델의 관절 위치 한계를 적용한다.
이전 해, 현재/neutral 자세 등의 seed를 시도하고 통과한 해 중 이전 해에 가까운 것을 선호한다.
해당 strict 기준은 기본적으로 위치 `0.1 mm`, 회전 `0.001 rad`다.

IK가 통과했다는 뜻은 **그 관절각을 실제로 만들면 목표 pose가 나온다**는 것이다.
그 관절각까지 8.33 ms 안에 움직일 수 있다는 뜻은 아니다.
기본 기하학적 IK 자체는 질량·접촉·모터 응답을 시뮬레이션하는 제어기가 아니다.

### 6.2 마운트 거리는 고려했는가?

고려했다. 현재 모델에서 중요한 끝점은 `right_hand_base_link`다.
단순한 RB3 `link6`, 원래 TCP, USD의 상위 컨테이너 prim과 혼동하면 안 된다.

`L = link6`, `B = Revo2 base`, `W = world`라 하면:

```text
T_W_B(q) = T_W_L(q) × T_L_B
T_L_B = translation(0, 0, 0.141304972 m), relative rotation = identity
```

이는 현재 모델의 link6 기준 offset이다. 초기 대화의 “34 cm 정도” 추정값이나
STL 외형 치수 자체를 그대로 쓰는 것이 아니다. 모델 설명상 flange 100 mm와
adapter 약 41.305 mm가 반영된 값이다.

Bare link6를 endpoint로 쓰는 다른 IK라면:

```text
T_W_L_target = T_W_B_target × inverse(T_L_B)
```

하지만 **현재 FK/IK는 이미 B까지 포함**한다. 현재 solver에 손목 target을 넣기 전에
위 역변환을 다시 적용하면 mount 보정이 중복된다.
컨테이너 prim의 authoring pose도 움직이는 palm의 실제 pose 대신 쓸 수 없다.

X=0.50의 reference 38 pose 검증에서 최대 오차는 `4.65e-7 m / 1.31e-6 rad`였다.
기록된 실제 관절각으로 복원한 base와 기록된 실제 base의 차이도 `6.15e-8 m`였다.
그런데 같은 실행의 target→actual에는 `50.43 mm / 0.40107 rad`가 있었다.
즉 이 검증에서는 **mount/FK는 맞고 실행 추종에는 오차가 있었다.**

근거: [WRIST_FRAME_DIAGNOSIS.md](WRIST_FRAME_DIAGNOSIS.md).
이 수치는 reference 정적 검증과 rollout 추종 검증을 구분한 결과다.

### 6.3 오프라인 IK와 온라인 IK

**오프라인:** retargeting 결과나 저장된 floating rollout을 미리 IK로 변환한다.
실행 시 저장된 관절 궤적을 재생한다. 새 캔 상태에 따라 매번 정책이 반응하는 것과 다르다.

**온라인:** 실제 mounted 상태로 정책을 추론하고, 새 손목 target을 실행 중 IK에 넣는다.
현재 `play-arm`은 이 방법이다. 저장된 floating action을 틀어주는 방식이 아니다.

또한 관절 상태를 직접 쓰는 kinematic replay는 시각적 FK 검증에 적합하지만,
실제 joint drive가 같은 동작을 수행하거나 캔을 잡는다는 증거는 아니다.
현재 정책 평가는 초기 reset 외에는 관절/root 상태를 직접 덮어쓰지 않는다.

### 6.4 IK 뒤에는 물리적 제어가 남아 있다

```text
손목 target → IK의 q_ik → 보간/제한 뒤 q_cmd
                              ↓
                       actuator 목표 전달
                              ↓
                    물리 적분, 중력, 관성, 접촉
                              ↓
                    q_actual → 실제 손목 pose
```

팔은 implicit PhysX force drive를 사용한다. 설명용 근사식은 다음과 같다.

```text
τ_PD ≈ Kp × (q_cmd - q_actual) + Kd × (dq_cmd - dq_actual) + τ_ff
```

실제 implicit solver 내부 결과가 위 식의 즉시 계산값과 같다고 가정하지 않는다.
특히 `set_joint_position_target`은 “원하는 위치를 전달”하는 API이지
“관절을 그 위치로 순간이동”하는 API가 아니다.

## 7. 첫 원인 분리: 명령을 잘못 보냈나, 팔이 못 따라갔나?

### 7.1 오차를 A/B/C로 분해

실패 episode의 정책→IK→후처리→native setter→physics→실제 상태 읽기를 계측했다.
관절은 이름으로 대응시키고, pose는 같은 Revo2 base 기준으로 맞췄다.

| 구분 | 비교 | 무엇을 확인하는가? |
|---|---|---|
| A | 실제 IK 입력 ↔ FK(q_ik) | 기하학적 IK 해의 정확도 |
| B | FK(q_ik) ↔ FK(q_cmd) | 보간·필터·위치/속도 제한이 목표를 바꾼 양 |
| C | FK(q_cmd) ↔ runtime base | 실제 관절 drive와 물리의 추종오차 |

실제 pose는 검증된 PhysX runtime body 데이터에서 읽었다.
USD `XformCache`만 보면 physics pose 업데이트 설정 때문에 오래된 authoring pose를
읽을 수 있다. USD transform이 유용한 정적 장착 검증과 runtime 측정을 구분했다.

기존 5k 정책의 실패 episode 15, 108 physics step에서:

- A 위치 최대: `5.79e-15 m`, 수치 오차 수준.
- B 위치 최대: `19.671 mm`; 당시 4-substep 보간의 중간 구간 포함.
- C 위치 평균/최대: `16.056 / 44.602 mm`.
- C 회전 최대: `0.340548 rad`, 약 19.51°.
- 보간이 끝나는 4번째 substep에서도 C 오차는 남았다.
- 최종 `q_cmd`와 실제 native position setter 인자는 일치했고 덮어쓰기는 관측되지 않았다.

따라서 그 실행에서는 주로 **C 단계**가 문제였다. 그렇다고 이후 모든 실행의 문제도
C라고 단정할 수는 없다. 뒤의 특이점 시험에서는 B도 크게 증가했다.

근거: [ARM_EXECUTION_DIAGNOSIS.md](ARM_EXECUTION_DIAGNOSIS.md).

### 7.2 캔을 놓쳐서만 오차가 생긴 것인가?

실패 실행의 동일한 108개 팔·손 명령을 기록한 뒤 policy와 IK를 호출하지 않고 재생했다.
캔 접촉 ON은 원래 실패 궤적을 정확히 재현했다. OFF에서는 기존 방식대로 캔을
멀리 옮기되 테이블·자기충돌 설정 등 나머지 조건을 보존했다.

| C 위치 오차 | 캔 ON | 캔 OFF |
|---|---:|---:|
| 평균 | 16.056 mm | 16.554 mm |
| 최대 | 44.602 mm | 51.234 mm |

캔이 없어도 큰 오차가 남았다. 따라서 “캔을 놓쳐서 RL 행동이 커진 것”만으로
설명할 수 없었다. 이 재생 중에는 RL 자체를 호출하지 않았다.
반대로 접촉의 영향이 전혀 없다는 결론도 아니다. ON/OFF 실제 움직임은 달랐다.

근거: [ARM_CONTACT_COMPARISON.md](ARM_CONTACT_COMPARISON.md).

### 7.3 토크가 부족했는가?

설치된 actuator 및 effort API의 구현을 확인했다.
로그의 `computed/applied effort` 중 일부는 **근사 PD 계산값이나 그 clipping 결과**였고,
native effort setter 값은 **명시적으로 제출한 feedforward**였다.
Joint reaction은 구동 토크만 분리한 측정이 아니었다.

검증한 의미의 **implicit solver drive-only torque를 얻지 못했으므로 포화는 UNKNOWN**이다.
“applied torque가 한계에 닿았으니 실제 모터 토크 부족”이라고 결론 내리지 않았다.
이 실행에서 제출된 arm feedforward는 0이었으며, 그것은 drive 토크도 0이라는 뜻이 아니다.

초기 ON/OFF 재생에서는 실제 속도와 위치명령 차분 속도가 10 rad/s 제한의
95% 근처에 머문 시간이 0%였다. 따라서 그 자료에는 sampled velocity clipping 증거도 없었다.
이는 나중의 특이점 구간에서 속도 한계가 문제 된 사실과 모순되지 않는다.

근거: [ARM_ACTUATOR_DIAGNOSIS.md](ARM_ACTUATOR_DIAGNOSIS.md).

## 8. 추종오차 개선 실험과 결과

### 8.1 움직이는 위치 target에 속도 target은 0이던 문제

기존 arm baseline은 위치 목표를 보간해 움직이면서 속도 목표는 0으로 전달했다.
이 경우 근사 PD에서 D항은 `-Kd × actual_velocity`가 되어 움직임에 반대 방향으로 작용한다.
이 자체가 항상 잘못된 제어라는 뜻은 아니지만, 당시 P/D 항과 지연의 관계를 시험할 근거였다.

속도 목표만 다음 값으로 바꾸는 비교를 실행했다.

```text
v_path[k] = (q_cmd[k] - q_cmd[k-1]) / physics_dt
```

캔 OFF 동일 명령 재생에서 C 최대 위치 오차는 `51.234→29.839 mm`,
wrist3 관절 최대 오차는 `.722778→.014654 rad`로 감소했다.
이는 **속도 목표 0이 그 실행의 지연에 기여했다는 가설을 지지**했다.

그러나 실제 정책을 각자의 관측으로 평가하자 같은 20배치의 성공이 **19/20→7/20**으로 줄었다.
wrist1/wrist3 최대 실제 가속도는 약 `365.6→1920.3`, `277.2→2255.7 rad/s²`로 증가했다.
초기 live 실험은 추가 제한 없이 차분을 전달했으므로 속도 목표가 10 rad/s를 넘는 구간도 있었다.

즉 **지연 감소와 파지 성능 악화가 동시에 일어났다.** 현재의 bounded 최종 q 차분,
c3 gains, physics-rate 응답 필터 조합과 이 “속도 목표만 바꾼” 시험을 같은 조건으로 보면 안 된다.

근거: [OFF 비교](ARM_VELOCITY_TARGET_COMPARISON.md),
[접촉 ON·정책 평가](ARM_VELOCITY_CONTACT_POLICY_VALIDATION.md).

### 8.2 정적·저속 시험에서 gains 조정

다음에는 정책/파지가 아니라 충돌 없는 정지 자세와 느린 움직임을 고정한 benchmark를 만들었다.
목표 자세, 초기 상태, 이동 시간, 측정 구간을 먼저 고정하고 gains를 비교했다.
동일 quintic 궤적에서 위치와 해석적 속도를 만들었으며 시험 명령 범위는
최대 `0.5 rad/s`, `1 rad/s²` 이하였다. 이는 제조사 사양이 아니라 시험 조건이다.

기준 포함 4개 gains 설정을 실행했고 `c3`를 선택했다.

| 시험 | 기존 | c3 |
|---|---:|---:|
| 선택 자세의 정지 최대 위치 오차 | 21.593 mm | 0.745 mm |
| 별도 검증 자세의 정지 최대 위치 오차 | 22.329 mm | 0.773 mm |
| 별도 검증 동작의 이동 P95 위치 오차 | 22.322 mm | 0.773 mm |

정지 1 mm, 느린 이동 P95 2 mm와 회전 기준을 통과했다.
중력·질량·관성·effort/velocity limit을 바꾸거나 매 step 관절 상태를 덮어쓴 결과가 아니다.
이 실험에서는 추가 중력보상도 넣지 않았다.

이어 같은 gains로 저고도 저속 명령을 비교해, 지연하지 않은 원본 목표 대비 이동 P95를
`1.297→1.059 mm`로 줄였다. 다만 최대는 `1.088 mm`였고,
임의의 온라인 정책이 아닌 **미리 알고 있는 부드러운 궤적**의 시험이었다.

따라서 “시뮬레이션에서 1 mm가 불가능하다”는 말도 틀리지만,
“이 gains면 빠른 접촉 파지도 언제나 1 mm”라는 말도 틀리다.

근거: [정밀 benchmark](ARM_PRECISION_BENCHMARK.md),
[저고도 명령 비교](ARM_LOW_HEIGHT_COMMAND_COMPARISON.md).

### 8.3 단순한 새 mounted 인터페이스와 응답 필터 단독 시험

수정이 누적된 실행과 분리해 `SimpleMountedWrist` 경로를 만들었다.
새 관측 정의를 만들지 않고 기존 ObservationManager, frozen policy,
reference/action decoder, FK/IK, hand mimic을 재사용했다.

초기 5k 정책의 같은 20배치 비교는 다음과 같았다.

| 방식 | 기존 task 성공 |
|---|---:|
| Floating | 20/20 |
| 기존 online arm | 19/20 |
| 초기 simple arm | 17/20 |
| 초기 simple arm + 30 Hz에서 0.1초 필터만 추가 | 5/20 |

Floating은 평균 목표→actual 손목 오차가 약 29.85 mm여도 잘 잡았고,
simple arm은 그 오차가 18.90 mm로 더 작아도 덜 잡았다.
한 배치에서 arm의 최초 캔 접촉은 0.10초, floating은 약 0.392초였다.
팔의 손이 먼저 캔을 건드리고, 캔이 움직이고, 관측과 정책 행동이 달라지는 순서가 관측됐다.

필터만 넣는 시도도 실패했다. 손목이 늦어지는 동안 손가락 닫힘은 원래 phase로 진행해
협응이 달라질 수 있다는 가설을 남겼다. 단일 원인으로 확정한 것은 아니다.

이때의 “필터만 추가해서 5/20”과 이후 “c3 + 120 Hz 응답 + bounded 속도 목표로 40/40”은
다른 실험이다. 같은 `tau=0.1`이라는 숫자만으로 서로 모순이라고 보면 안 된다.

근거: [최소 인터페이스](MINIMAL_MOUNTED_INTERFACE.md),
[응답 비교](MOUNTED_RESPONSE_COMPARISON.md).

### 8.4 성공한 floating의 실제 움직임을 복사하면 되는가?

별도 진단에서 성공한 floating의 **실제 wrist/손가락 각도**를 저장해 팔로 재생했다.
정책은 호출하지 않았고 캔은 dynamic body로 유지했다. 처음 결과는 0/20이었다.

여기에는 두 구분이 중요하다.

1. 실제 손가락 각도는 원래 actuator에 보낸 손가락 목표가 아니다.
   이미 부하와 지연을 겪은 actual angle을 새 목표로 쓰면 다른 drive 동작이 된다.
2. 실제 wrist 경로를 목표로 주더라도 팔 actuator가 못 따라가면 원래 물리 상태는 재현되지 않는다.

후속 R1에서 손목 actual 경로에 **원래 손가락 명령**을 결합해도 0/20이었다.
R2에서 c3와 path velocity 응답을 적용하자 기존 task 지표 20/20,
지속 lift/contact 보조 지표 19/20, wrist 평균 약 0.911 mm가 됐다.
이 결과는 제어 응답 개선의 증거지만 **기록 재생 결과이지 live RL 평가가 아니다.**

근거: [actual motion 재생](FLOATING_ACTUAL_MOTION_REPLAY.md),
[복구 실험](ARM_TRANSFER_RECOVERY.md).

### 8.5 실제 정책에서 효과가 확인된 조합

실제 mounted observation으로 정책을 매번 추론하면서 다음 조합을 시험했다.

- 정적·저속 benchmark의 `c3` gains.
- 0.1초 first-order wrist 응답을 policy 경계가 아니라 physics 120 Hz에서 갱신.
- 최종 q_cmd를 위치·속도 한계 안으로 제한.
- 그 **최종 q_cmd**의 physics-dt 차분을 속도 목표로 전달.
- 손가락 명령과 reference phase는 원래 경로 유지.

5k 정책의 old20은 `19→20`, 별도 heldout20은 `17→20`의 task 성공이었다.
새 조합은 두 bank 모두 lift/contact 보조 기준까지 통과했다.
FK(최종 q_cmd)→actual 최대 위치 오차도 약 `56→6 mm`로 줄었다.

반면 raw 정책 평형점→actual 평균 오차는 약 `22.5→25.6 mm`로 오히려 커졌다.
이는 응답 필터가 만드는 차이를 숨기지 않은 수치다.
실제 팔 가속도 최고값은 높아졌으므로 이 조합을 실물 안전 제어기로 검증한 것은 아니다.

이 조합이 이후 “잘 잡히던 영상의 팔 실행 경로”이며 현재 기본 제어기의 바탕이다.
근거: [ARM_TRANSFER_RECOVERY.md](ARM_TRANSFER_RECOVERY.md).

## 9. 실행이 느렸던 문제와 30 Hz / 120 Hz

서로 다른 세 시간을 구분해야 한다.

| 시간 | 현재 기본값/의미 |
|---|---|
| 정책·reference 갱신 | 30 Hz, 약 33.33 ms 간격의 simulation time |
| IK·응답 필터·physics | 120 Hz, 약 8.33 ms simulation step |
| 화면과 실제 경과시간 | 계산·렌더링 성능에 따라 느려질 수 있음 |

30 Hz로 목표를 보낸다고 “다음 1/30초까지 무조건 도착”하는 것은 아니다.
위치명령 주기와 실제 도달 능력은 별개다. 목표에 도착할 때까지 phase를 멈추면
학습 당시의 손목·손가락·물체 reference 시간 관계를 바꾸게 된다.
현재는 그렇게 기다리며 진행하는 방식이 아니다.

당시 비싼 multi-seed IK 때문에 느린 부분에 대해 다음을 적용했다.

- 먼저 이전 해에서 한 번 풀고, 충분히 작고 정확한 해면 채택하는 warm-first 경로.
- 같은 FK 모델의 analytic Jacobian과 solve 내부 cache.
- 실패/큰 변화에는 원래 multi-seed solver로 fallback.
- 시각화 모드에서는 비싼 진단 기록 일부를 생략하되 physics step 자체는 유지.

기록된 headless 비교에서는 원래 IK 약 35.9 ms에서 fast IK 약 1.9 ms로 줄고,
전체 속도는 약 0.172×에서 0.632×로 개선됐다. GUI는 당시 약 0.455–0.485×였다.
이는 해당 실행 기록이며 **현재 모든 PC에서의 속도 보장**은 아니다.
Fallback, PhysX, 렌더링 때문에 8.33 ms deadline을 항상 만족한다는 증거도 없다.

IK만 30 Hz로 낮춰 첫 substep에서 계산하고 나머지에 목표를 유지하는 옵션도 시험했다.
실제 30/120 스케줄은 확인했지만, 현재 기본값은 사용자 요청에 따라 **120 Hz IK**다.
별도 30 Hz 옵션이 모든 파지 배치에서 동일 품질을 보장한다고 기록하지 않는다.

영상의 0.25×/0.5× 편집과 마지막 frame hold는 또 다른 문제다.
영상 편집 속도는 policy dt, IK 주기 또는 simulator 물리 설정 변경이 아니다.

근거: [ARM_REALTIME_EXECUTION.md](ARM_REALTIME_EXECUTION.md).

## 10. 작은 손 움직임에 팔이 0.8 rad 튀던 이유

### 10.1 관절 한계 끝이 아니라 손목 특이점 근처였다

한 배치의 1.041667초에서 손목 목표는 **2.317 mm / 0.002692 rad**만 변했는데,
IK의 wrist1/wrist3는 **-0.79817 / +0.79750 rad** 변했다.
8.33 ms 동안 실현하려면 약 95.8 rad/s인데 당시 팔 제한은 10 rad/s였다.

이때 wrist2는 약 0.9°여서 wrist1과 wrist3 축이 거의 겹쳤다.
두 축의 반대 회전이 손 전체에서는 거의 상쇄되므로, 손 움직임은 작아 보이면서
관절각은 크게 변할 수 있다. Jacobian 분석도 이 민감도를 뒷받침했다.

가장 가까운 위치 한계까지는 약 0.275 rad 남아 있었고,
모든 관절 범위를 0.1 rad씩 줄여도 같은 급변이 재현됐다.
따라서 **관절 끝 제한을 조금 줄이는 것만으로 해결되는 문제가 아니었다.**

### 10.2 Gains를 더 올리는 시험은 채택하지 않았다

wrist3 Kp/Kd 후보를 비교해 관절 추종오차 일부는 줄였다.
그러나 별도 20배치 중 한 배치의 마지막 lift/contact 유지가 실패해
그 후보는 교체용으로 채택하지 않았다. 수학적 IK가 요구하는 큰 속도 자체는
gains를 올려도 없어지지 않았다.

근거: [ARM_IK120_IMPROVEMENT.md](ARM_IK120_IMPROVEMENT.md).

### 10.3 별도 속도·명령 가속도 제약 IK 후보

후속 옵션에서는 unrestricted IK 결과를 사후에 자르기만 하는 대신,
이전 **실제 전달 q_cmd** 주변의 한 timestep 이동 가능 범위 안에서 IK를 풀었다.
속도 제약만으로는 큰 가속도가 남아 명령 가속도 `250 rad/s²` 제약도 추가했다.
이 값은 명령 생성용 실험값이며 native actuator나 제조사 사양이 아니다.

5k 정책의 두 bank에서 파지·접촉 유지 40/40을 보존하며 실제 최대 가속도를
약 `636.7→289.6`, `729.3→281.6 rad/s²`로 줄였다.
문제 배치의 최대 IK step도 약 `.798→.0833 rad`로 줄었다.

대신 작은 pose 근사를 허용했다. 최대 raw IK 오차가 위치 1.169 mm,
회전 .04849 rad였고 strict 기준을 넘은 timestep도 명시적으로 기록했다.
“정확한 pose를 순간적으로 실현할 수 없다면 실행 가능한 근사 경로를 선택”하는 trade-off다.

**이 후보는 현재 기본 video/rubber 경로에 자동으로 들어가 있지 않다.**
현재 공유 설정은 `velocity_bounded_ik=False`, `ik_acceleration_limit=0`이다.
좋았던 5k 후보의 수치를 현재 10k+고무 실행의 성적으로 그대로 인용하면 안 된다.

근거: [ARM_IK_SINGULARITY_FIX.md](ARM_IK_SINGULARITY_FIX.md).

## 11. 영상과 최근 실행이 달랐던 이유, 고무 설정, 파인튜닝

### 11.1 “같은 정책인데 영상과 다르다”를 재현해서 확인

보존한 과거 영상은 5k policy + `SimpleMountedWrist` + c3 + 120 Hz response/IK + path velocity였다.
최근의 다른 arm 실행은 10k 또는 transfer policy + `RB3WristIKAction` + 기존 낮은 gains + 30 Hz IK였다.
체크포인트뿐 아니라 **팔 제어기 자체가 다른 경로**였다.

영상의 metadata로 checkpoint/controller/state bank를 복원하는 `--match-recording`을 만들고,
실제 정책을 다시 추론해 원래 152개 physics 샘플을 재현했다.
즉 기록된 action을 재생해서 억지로 맞춘 결과는 아니었다.

같은 원본 10k 정책으로 제어기만 기존 영상 조합으로 맞췄을 때:

- 기존 strict arm의 hold: rigid 33/40, compliant 34/40.
- 영상 제어기의 task/hold: rigid 40/40, compliant 40/40.
- 실제 floating wrist와의 평균 차이도 대략 12–13 mm에서 5–6 mm로 줄었다.
- 여전히 완전히 같은 실제 동작은 아니며, 높은 가속도 한계도 남았다.

근거: [영상·현재 경로 비교](ARM_REALTIME_EXECUTION.md#video-versus-current-path-diagnosis--2026-09-09).

### 11.2 Revo2 마지막 마디의 고무 근사

실물 마지막 마디가 고무라는 설명을 반영해 distal link와 고정 touch child의
10개 collision shape에 compliant contact를 적용했다.

- 마찰계수 static/dynamic `0.8/0.8`: 유지.
- 접촉 강성 `10,000 N/m`, 감쇠 `10 N·s/m`: 근사값.
- 원본 USD 저장, mesh/질량/관성 재제작, 손가락 actuator gain 변경: 없음.
- Force-response 근사이지 실제 고무가 변형되는 deformable mesh 모델은 아님.

이 수치는 실물 눌림량·경도를 측정해서 동정한 값이 아니다.
고무 근사 적용이 모든 파지 문제를 해결하거나 실물과 같아졌다는 결론은 내리지 않았다.
현재 arm 기본값에는 적용됐지만 기본 floating train/play 설정까지 바꾼 것은 아니다.

### 11.3 서로 다른 파인튜닝을 구분

팔 환경 파인튜닝은 기존 floating 모델을 초기값으로 로드한 뒤,
**실제 mounted 상태의 관측과 reward로 그 정책 자체를 PPO 업데이트**하는 것이다.
Frozen policy 뒤에 두 번째 residual policy를 추가하거나 팔 관절 action을 추가한 것이 아니다.

이전에 기존 strict controller에서 진행한 100-update 실험들은 조건이 달랐다.
Rigid 실험은 hold 33→39/40, compliant 실험은 34→25/40이었다.
이 결과를 현재 video controller의 추가 학습 결과로 섞지 않는다.

최신 한 시간 실험은 **현재 video controller + 고무 근사**를 학습·평가에서 공유했다.

- 원본 10k 모델에서 새 optimizer로 초기화; 정상 로드 직후 동일 observation의 action 차이 0.
- Actor/critic, distribution, normalizer를 로드하고 학습에서는 업데이트, 평가에서는 동결.
- 기존 PPO 사용, 초기 LR `1e-4`; 기존 adaptive schedule에 따라 실행 중 `1e-5`.
- 1 env, 고정 배치, full gravity, randomization OFF, RSI OFF.
- 3,601초, 2,679 updates, 64,296 transitions, 53,580 optimizer steps.
- 원본 checkpoint hash 유지, NaN/Inf 없음, 최종 모델 `model_2678.pt` 저장.

같은 두 초기 배치 bank에서 실제 초기 arm/hand/object 상태와 controller를 확인하고 비교했다.
각 정책은 자신의 실행에서 생긴 actual observation으로 행동을 계산했다.

| 최신 40배치 비교 | 원본 10k | 한 시간 추가 학습 |
|---|---:|---:|
| 기존 task 성공 | 40/40 | 40/40 |
| 별도 lift/contact 유지 | 40/40 | 40/40 |
| 평균 물체 keypoint 오차 | 3.94 mm | 11.16 mm |
| Raw 정책 손목 목표→actual 평균 위치 오차 | 26.26 mm | 26.46 mm |
| 평균 손가락 target→actual 오차 | .02235 rad | .02880 rad |
| 평균 마지막 캔 상승 | 227.38 mm | 221.52 mm |

기존 task의 `success`는 reference 종료 도달과 관련된 플래그다. 실패 종료 항목도
같이 확인하되, 이 플래그만으로 안정적으로 쥐고 있다고 판단하지 않았다.
추가 lift/contact 지표는 **마지막 0.2초 동안 최소 상승 0.1 m 이상이고,
캔–로봇 접촉력이 0.01 N보다 큰 sample이 그 구간의 80% 이상**인지 확인한다.
이는 학습 reward나 종료조건을 바꾼 것이 아니라 별도 진단 지표다.

성공→실패, 실패→성공 배치는 없었다. 파지를 더 잘하게 됐다는 증거는 없고,
추종 지표는 일부 악화됐으므로 **최종 transfer 모델은 선택형으로 보존**했다.
고정 배치의 짧은 학습이라는 한계는 있지만, 실패 원인을 “학습이 부족해서”라고
단정하거나 더 오래 학습하면 반드시 좋아진다고 결론 내릴 수는 없다.

상세 조건·그래프·100-update 이력: [RL_TASK.md](RL_TASK.md#one-hour-execution--2026-09-09).

## 12. 현재 기본 제어를 한 timestep씩 풀어 쓰면

구현 중심: `regrind/source/regrind/regrind/utils/arm_execution_config.py`,
`mdp/simple_mounted_interface.py`(task package 기준).

### 12.1 Reset

배치와 reference 상태를 선택하고 실제 팔·손·캔 상태를 초기화한다.
그 뒤 IK warm start, 직전 전달 q, 응답 필터, 속도 차분 이력과 action/history를 동기화한다.
이전 episode 끝의 명령이 새 episode 첫 속도 계산에 섞이지 않도록 한다.
현재 video 경로는 단일 env에서 검증됐으며 video transfer는 RSI를 거부한다.
기존 baseline transfer의 선택형 RSI/reset 검사와 혼동하지 않는다.

### 12.2 매 policy step, 1/30초

1. 실제 mounted 손목/손가락/물체 상태로 기존 observation을 만든다.
2. Frozen 10k 정책을 추론한다.
3. 기존 decoder로 residual을 clipping·scaling해 reference와 한 번 결합한다.
4. 다음 네 physics step에 사용할 raw 손목 평형 목표와 손가락 목표를 갱신한다.

### 12.3 매 physics step, 1/120초

손목 위치 응답은 다음과 같다. 회전은 shortest rotation을 이용해 같은 비율로 접근한다.

```text
alpha = 1 - exp(-physics_dt / 0.1)
p_ik[k] = p_ik[k-1] + alpha × (p_policy_target - p_ik[k-1])
```

이것은 명령을 부드럽게 만드는 상태이지, RL에 넣는 “실제 손목 관측”이 아니다.
필터된 pose로 warm-first IK를 풀고, 검증된 해를 목표로 채택한다.
현재 q_cmd의 갱신은 개념적으로 다음과 같다.

```text
q_next = position_clip(q_previous + clip(q_ik - q_previous, ±v_limit × dt))
dq_next = clip((q_next - q_previous) / dt, ±v_limit)

arm position target ← q_next
arm velocity target ← dq_next
hand targets        ← 기존 6 leader + 5 mimic 경로
physics step
actual joint/body/object state 갱신
```

IK 실패 시 이전 유효 목표를 유지하고 실패를 기록한다.
Warm-first의 `.15 rad` 조건은 빠른 해를 채택할지 판단하는 기준이다.
Fallback의 raw IK 변화까지 `.15 rad` 이하로 보장하는 제한은 아니다.
최종 actuator q 이동 제한과도 구분해야 한다.

팔의 현재 runtime 기준 설정은 다음과 같다.

| 관절 | 기존 Kp → 현재 Kp | 기존 Kd → 현재 Kd | effort limit [N·m] |
|---|---:|---:|---:|
| base | 300 → 700 | 20 → 35 | 10 |
| shoulder | 500 → 20000 | 20 → 260 | 100 |
| elbow | 500 → 12000 | 20 → 140 | 100 |
| wrist1 | 300 → 900 | 20 → 45 | 100 |
| wrist2 | 200 → 2400 | 20 → 70 | 100 |
| wrist3 | 50 → 250 | 10 → 20 | 10 |

Force drive의 Kp는 N·m/rad, Kd는 N·m·s/rad다. Arm 속도 제한은 모두 10 rad/s.
손은 Kp=3, Kd=.1, effort=.5 N·m인 기존 설정을 유지한다.
추가 arm 중력보상 feedforward는 없다. **제한 숫자들은 시뮬레이션 설정이며 실물 사양이 아니다.**

현재 factory가 명시적으로 선택하는 값:

```text
arm_controller       = video
response_tau         = 0.1 s
response_at_physics  = True
fast_ik              = True
ik_policy_rate       = False        # 따라서 IK 120 Hz
velocity_path        = True
velocity_bounded_ik  = False        # 10절의 별도 후보는 사용하지 않음
ik_acceleration_limit= 0
num_envs             = 1
```

기본 dataclass의 `False`나 예전 JSON의 `not_default` 상태 문자열만 보면 현재 선택을
잘못 읽을 수 있다. 실제 launcher→공유 factory가 덮어쓴 **최종 설정**이 기준이다.

## 13. 서로 다른 “손목 오차”를 섞지 않는 법

| 수치의 종류 | 의미 | 대표 결과의 해석 |
|---|---|---|
| Reference pose→IK/FK | 기하학/장착 검증 | 매우 작아도 동적 추종을 보장하지 않음 |
| Filtered IK 입력→actual | 필터 이후 팔 실행 성능 | Raw policy 목표와 비교한 값이 아님 |
| FK(q_cmd)→actual, C | 최종 전달 명령에 대한 actuator 응답 | 보간/제한이 목표를 바꾼 B를 포함하지 않음 |
| Raw policy 평형점→actual | 필터의 지연까지 포함한 전체 차이 | 현재 평균 약 26 mm는 이 정의 |
| Floating actual→mounted actual | 같은 시각의 실제 동작 차이 | 최근 비교 평균 약 5–6 mm는 이 정의 |
| 정적/저속 benchmark | 고정된 부드러운 목표에서의 정밀도 | 약 0.7–0.8 mm를 빠른 파지 성적으로 쓰면 안 됨 |

위치·회전 오차는 각각 계산한다. 평균/P95/최대, 초기 정착 포함 여부,
실패 후 구간 포함 여부, 비교한 checkpoint와 배치가 같은지도 확인해야 한다.
회전·위치 오차 norm은 일반적으로 A+B+C의 스칼라 단순 합이 아니다.
시간축을 뒤로 이동시켜 lag를 지운 수치를 실제 추종오차로 보고하지 않았다.

## 14. 해결한 것과 아직 남은 것

### 확인·개선된 부분

- 현재 mount frame/FK/joint ordering의 일치와 물리 pose 측정 경로.
- 실제 residual 목표, q_ik, q_cmd, q_actual을 구분하는 계측.
- 초기 실패에서 명령 덮어쓰기보다 actuator 응답이 큰 오차를 만든다는 근거.
- 정적·저속 정밀도, 빠른 IK 처리량, 일부 closed-loop 파지 성능 개선.
- 작은 pose 변화에 큰 wrist 관절 변화가 필요한 특이점 메커니즘과 별도 개선 후보.
- 과거 영상과 다른 launcher/controller를 실행했던 문제의 재현·분리.
- 학습/평가 controller 공유, transfer 초기화·resume·normalizer 검증.
- 한 시간 추가 학습이 반드시 개선으로 이어지지는 않는다는 실제 비교 결과.

### 남은 한계

- 현재 기본값이 모든 순간 1 mm 추종을 보장하지 않는다.
- 높은 실제 관절 가속도와 wrist 속도 한계 근접 구간이 남아 있다.
- 별도 가속도 제약 IK 후보가 현재 10k+고무 조합의 기본으로 재검증·채택된 것은 아니다.
- 고무는 측정 기반 실물 동정이 아닌 접촉 근사다.
- 실제 solver drive 토크 포화는 UNKNOWN이며 “토크 부족이 유일 원인”은 미확정이다.
- 손가락의 작은 실제 joint-limit 초과도 기록돼 있어 전체 로봇을 완전 무위반으로 부르면 안 된다.
- 40개 저장 배치의 성공은 임의 위치·yaw·질량·다른 물체·긴 hold에 대한 보장이 아니다.
- 현재 video controller 학습/평가는 1 env이며, 대규모 병렬 mounted 학습을 검증한 상태가 아니다.
- 실물 RB3 내부 servo, 감속기/마찰, 실제 drive 제약과 이 시뮬레이터의 동등성은 검증되지 않았다.
- 물체 tracking/contact와 관측 변화까지 함께 보는 문제다. 손목 오차 하나만으로 파지 품질을 판단할 수 없다.

현재 상태를 한 문장으로 요약하면:

> 손목 pose를 IK에 넣는 계산 자체는 검증됐고, 그 뒤의 물리 응답과 정책의 시간적 협응을
> 맞추는 쪽으로 상당 부분 개선했다. 다만 floating과 완전히 같거나 실물에 바로 적용 가능한
> 제어기가 된 것은 아니며, 최신 추가 학습 모델도 기본 정책보다 낫다는 증거가 없었다.

## 15. 실행 명령과 코드·근거 찾아보기

### 현재 실행

프로젝트 root에서 실행한다. GUI가 가능한 Isaac 환경과 로컬 모델/reference가 필요하다.

```bash
# 기존 floating 정책
./scripts/rl.sh play

# 현재 기본: video controller + 고무 근사 + 원본 10k 정책
./scripts/rl.sh play-arm --episodes 20

# 이전 strict-IK controller 보존 경로
./scripts/rl.sh play-arm --arm-controller baseline --num_envs 1 --real_time

# 한 시간 추가 학습 모델만 명시적으로 선택
./scripts/rl.sh play-arm --episodes 20 --checkpoint \
  logs/rsl_rl/rb3_revo2_tuna_transfer_video/2026-09-09_20-43-36_video_rubber_hour/model_2678.pt
```

현재 arm 기본은 저장된 heldout 20배치를 순차 사용한다. 실행할 때마다 새로운
무제한 XY/yaw 배치를 무작위로 만드는 옵션이라고 이해하면 안 된다.
원본 10k checkpoint:
`logs/rsl_rl/floating_revo2_tuna/2026-09-08_01-28-29_floating_stable_ground_10000/model_9999.pt`.

파인튜닝 시작·resume·동일 상태 평가 명령은
[RL_TASK.md](RL_TASK.md#approved-video-controller-and-timed-transfer)에 있다.
과거 보고서의 `play-arm` 명령은 당시 기본 제어기를 전제하므로,
지금 복사할 때 `--arm-controller baseline`이 필요한지 확인한다.
과거 영상은 `--match-recording`으로 저장된 설정을 복원한다.

### 구현 위치

아래 `config/`와 `mdp/`의 공통 prefix는
`regrind/source/regrind/regrind/tasks/manager_based/dexterous/`다.

| 역할 | 코드 |
|---|---|
| 대표 실행·checkpoint 선택 | `scripts/rl.sh`, `scripts/_common.sh` |
| 현재 arm config 조립 | `regrind/source/regrind/regrind/utils/arm_execution_config.py` |
| 기존 결합형 RL | `config/rb3_revo2/rb3_revo2_tuna_env_cfg.py` |
| Floating RL | `config/revo2_floating/revo2_floating_tuna_env_cfg.py` |
| 손목 action decoder / floating wrench | `mdp/actions.py::SE3ImpedanceActionTerm` |
| 기존 팔 IK action / hand leader·mimic | `mdp/rb3_revo2_actions.py` |
| 현재 simple mounted 적용 | `mdp/simple_mounted_interface.py` |
| 실제 상태·reference·phase | `mdp/rb3_revo2_commands.py`, `mdp/observations.py` |
| 실제 정책 평가 / recorded profile | `tools/rb3_revo2_ik/evaluate_mounted_interface.py`, `recorded_run_profile.py` |
| FK·원본 bounded IK | `tools/rb3_revo2_ik/rb3_kinematics.py`, `rb3_model.json` |
| 빠른 기존 IK / 별도 제한 IK | `tools/rb3_revo2_ik/warm_start_ik.py`, `velocity_bounded_ik.py` |
| 모델 초기화·resume·시간 제한·PPO 계측 | `regrind/source/regrind/regrind/utils/arm_transfer.py`, `regrind/scripts/rsl_rl/train.py` |
| 고무 근사 설정 / 적용 | `config/experiments/revo2_rubber_contact.json`, `regrind/source/regrind/regrind/utils/revo2_contact_material.py` |
| 비교 통계 | `tools/arm_diagnostics/analyze_transfer_recovery.py` 및 같은 폴더의 단계별 분석기 |

역사별 근거 문서는 각 절의 링크에서 선택해서 읽으면 된다.
현재 전체 안내는 [architecture](architecture.md), [current-status](current-status.md),
[실행·진단 색인](../scripts/README.md)으로 연결된다.

최신 비교 자료는 로컬 `outputs/diagnostics/video_transfer_hour_20260909/`의
`comparison_old20/transfer_comparison.{json,png}` 및
`comparison_heldout20/transfer_comparison.{json,png}`에 있다.
GitHub에는 코드·문서·설정·테스트를 올렸지만 체크포인트·대형 로그·영상은 올리지 않았으므로,
다른 PC에서는 문서의 로컬 결과 경로가 없을 수 있다.
직전 기능 커밋 시점의 root 회귀 검증은 142개 통과였으며,
그 사실이 모든 실험 후보의 물리 성능을 보장하는 것은 아니다.
