---
name: trace-regrind-path
description: Trace an unfamiliar DexYCB, retargeting, Revo2, RB3, Isaac, or RL execution path, or scope a requested modification. Use for architecture/location questions and change planning; use diagnose-regrind-failure instead for a reproducible malfunction.
---

# Trace a REGRIND path

1. Read the matching section of `docs/architecture.md`. Consult
   `docs/current-status.md` only when support or legacy status matters, and open
   at most one linked subsystem document initially.
2. Start from the maintained root command, registered task ID, or concrete
   artifact. Search wrappers, imports, registrations, config inheritance, and
   call sites before opening implementation sections.
3. Trace inputs, transformations, outputs, configuration sources, and the
   nearest existing tests. Verify command names and flags in repository files.
4. Report the verified chain with file/symbol evidence. Separate facts,
   inferences, and unresolved links; do not expand into implementation unless
   requested.
