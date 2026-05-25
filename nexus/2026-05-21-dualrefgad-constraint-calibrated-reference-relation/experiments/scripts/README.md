# Experiment scripts

The first step intentionally reuses the already-tested Route2.5 decomposition probe script:

`../2026-05-19-dualrefgad-route25-matrix-autoencoder/experiments/scripts/route25_leg3_response_matrix_decomposition_probe.py`

Reason: the immediate goal is not to fork scientific logic, but to fix C-LEG3 / old_exact reference regime and audit available signal families. If R0/R4 justify pseudo-anomaly training, this investigation will add a dedicated R1–R3 training script here.

Execution rule: any real run must be registered or tracked through `experiment-runner`; ad-hoc results are smoke tests only.
