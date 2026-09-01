# DexYCB-to-Revo2 batch pipeline

`dataset/` 원본은 변경하지 않고, 모든 sequence에서 **두 번째 카메라**의 MANO
21점과 grasp 대상 pose를 추출합니다. 이번 데이터의 실제 물체는 프로젝트의
`007_tuna_fish_can` asset으로 고정합니다. 잘못된 `joint_3d == -1` 프레임은 제거하고
원본 프레임 번호를 `source_frame_indices`에 보존합니다. 왼손 capture는 grasp
object-local X 평면에서 반사하여 Revo2용 오른손 입력을 함께 생성합니다. 원본의
sequential MANO21 순서를 검증된 Revo2 semantic `kp_00..kp_20` 순서로 명시적으로
재정렬하여 리타게팅 correspondence와 시각화 topology가 섞이지 않게 합니다.

```bash
cd /home/wanjunkim/ARSL/regrind-upload
./scripts/run_pipeline.sh
```

결과:

- `outputs/preprocessed/dexycb/<sequence>/dexycb_right_hand_preprocessed.npz`
- `outputs/preprocessed/dexycb/<sequence>/preprocess_summary.json`
- `outputs/preprocessed/dexycb/manifest.json`
- `outputs/retargeted/dexycb/<sequence>/revo2_retargeted.h5`
- `outputs/visualizations/dexycb/<sequence>/preprocessed_interactive.html`
- `outputs/visualizations/dexycb/<sequence>/retargeted_interactive.html`
- `outputs/visualizations/dexycb/index.html`

HTML은 Plotly JavaScript를 파일 안에 포함하므로 해당 HTML 하나만 복사해도
브라우저에서 열 수 있습니다.

최종 `retargeted_interactive.html`은 같은 화면에 오른손 MANO 21점, FK로 계산한
Revo2 semantic 21점, tuna fish can surface 50점을 표시합니다. 이 배치에는
Isaac world transform과 RB3 strict IK reference 생성도 포함됩니다.
Isaac 변환은 캔의 local Z가 아니라 두 번째 optical camera의 고정 중력축
(`camera -Y = world +Z`)을 사용합니다. 따라서 캔이 눕거나 local +Z가 아래를
향하는 자세에서도 물체·MANO·wrist의 상하 운동이 뒤집히지 않습니다.

Isaac GUI에서 사용할 sequence 확인과 실행:

```bash
./scripts/run_isaac_replay.sh --list-sequences
./scripts/run_isaac_replay.sh --sequence 20200709_143747_left
```

GUI에서는 RB3 6축과 Revo2 6축, tuna mesh와 원본 MANO21 스켈레톤을 같은 world
좌표계에서 표시합니다. 기본값은 순수 kinematic replay이며 `--physics-object`를
지정했을 때만 캔에 중력/contact를 적용합니다.
