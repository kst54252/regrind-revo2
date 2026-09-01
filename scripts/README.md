# User commands

Run commands from the repository root. The maintained entry points are:

| Command | Purpose |
|---|---|
| `./scripts/run_pipeline.sh` | DexYCB preprocessing through strict IK |
| `./scripts/run_isaac_replay.sh` | Isaac Sim kinematic/physics-object replay |
| `./scripts/rl.sh train` | PPO smoke or full training |
| `./scripts/rl.sh play` | Deterministic trained-policy replay |
| `./scripts/rl.sh zero` | Zero-residual reference validation |
| `./scripts/rl.sh debug` | Observation/reward/RSI diagnostics |
| `./scripts/run_tests.sh` | Shell syntax and Python regression tests |

Use `./scripts/rl.sh --help` for the shared sequence/reference options. The
older `train_rb3_revo2_ppo.sh`, `play_rb3_revo2_ppo.sh`, and
`run_rl_*.sh` names are compatibility wrappers; they contain no independent
launcher logic.
