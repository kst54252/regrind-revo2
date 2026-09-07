# Internal tools

이 디렉터리는 기능별 구현을 보관합니다. 일반 사용자는 프로젝트 루트의
`scripts/` 명령을 우선 사용하세요.

| 디렉터리 | 책임 |
|---|---|
| `dexycb_batch/` | 전체 sequence 전처리, REGRIND 호출, Isaac reference 생성 |
| `dexycb_world_transform/` | camera/world rigid transform과 시각화 |
| `revo2_kinematics/` | Revo2 FK, 21 semantic keypoint, joint limits |
| `rb3_revo2_ik/` | RB3 FK/IK, reference 생성, 진단, Isaac GUI replay |
| `arm_diagnostics/` | 기록된 팔 추종·접촉·정책 평가 로그의 분석과 비교; 시뮬레이션 실행과 분리 |

주 회귀 테스트는 최상위 `tests/`에 있으며 `scripts/run_tests.sh`가 실행합니다.
`regrind/source/regrind/test/`의 별도 패키지 테스트는 이 명령의 수집 대상이 아닙니다.
