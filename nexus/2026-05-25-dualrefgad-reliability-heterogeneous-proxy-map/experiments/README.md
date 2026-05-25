# Experiments

This directory contains runner-registered pure probes for the reliability / heterogeneous-support proxy-map investigation.

Subdirectories:

- `configs/` — experiment-runner YAML configs.
- `scripts/` — probe scripts, designed to reuse upstream C-LEG3 helpers where possible.
- `outputs/` — aggregate JSON and progress JSON pulled back from runner execution.
- `logs/` — remote/local execution logs pulled back from runner execution.

Protocol boundary:

- no training in first probe;
- labels diagnostic-only;
- fixed C-LEG3 / `old_exact_080_regime` unless a later PROGRESS entry explicitly changes this.
