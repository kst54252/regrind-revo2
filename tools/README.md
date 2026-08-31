# Internal tools

이 디렉터리는 기능별 구현을 보관합니다. 일반 사용자는 프로젝트 루트의
`scripts/` 명령을 우선 사용하세요.

| 디렉터리 | 책임 |
|---|---|
| `dexycb_batch/` | 전체 sequence 전처리, REGRIND 호출, Isaac reference 생성 |
| `dexycb_world_transform/` | camera/world rigid transform과 시각화 |
| `revo2_kinematics/` | Revo2 FK, 21 semantic keypoint, joint limits |
| `rb3_revo2_ik/` | RB3 FK/IK, reference 생성, 진단, Isaac GUI replay |

회귀 테스트는 구현과 섞이지 않도록 최상위 `tests/`에 모았습니다.
