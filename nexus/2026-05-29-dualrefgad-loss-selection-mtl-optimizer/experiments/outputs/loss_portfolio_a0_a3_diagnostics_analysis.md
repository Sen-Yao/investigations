# DualRefGAD A0-A3 Loss Diagnostics Supplement

Base report to supplement: https://report.senyao.org/reports/2026/05/29/dualrefgad-loss-selection-mtl-optimizer-2026-05-29.html
Runner job: `exp_20260529_203308_dualrefgad_loss_portfolio_a0_a3_diagnost`
Status: `finished`; elapsed: 11.8 min

## Aggregate metric table
| variant | RIFT AUC | RIFT AP | mat_mean AUC | ΔAUC vs mat_mean | Spearman(RIFT, mat_mean) |
|---|---:|---:|---:|---:|---:|
| A0_rocc_mc | 0.7223 ± 0.0106 | 0.4106 ± 0.0515 | 0.8104 ± 0.0068 | -0.0881 ± 0.0120 | 0.1082 ± 0.0686 |
| A1_hinge_rank_barrier | 0.6554 ± 0.0400 | 0.1821 ± 0.0516 | 0.8104 ± 0.0068 | -0.1549 ± 0.0453 | -0.1667 ± 0.0926 |
| A2_view_consistency | 0.6554 ± 0.0400 | 0.1821 ± 0.0516 | 0.8104 ± 0.0068 | -0.1549 ± 0.0453 | -0.1667 ± 0.0926 |
| A3_pair_reliability | 0.6656 ± 0.0328 | 0.2243 ± 0.0769 | 0.8104 ± 0.0068 | -0.1448 ± 0.0378 | -0.2001 ± 0.0532 |

## Gradient audit: does each loss update encoder/reader?
Frequencies are averaged over audited batches. Encoder is the TransformerEncoder block; reader_non_encoder includes value/identity embeddings, layer norms, pooling/output parts.

### A0_rocc_mc
| component | total grad norm | encoder update freq | reader update freq | scalar active freq |
|---|---:|---:|---:|---:|
| collapse_or_rank_barrier | 0.000863 | 1.000 | 1.000 | 1.000 |
| known_normal_energy | 0.155607 | 1.000 | 1.000 | 1.000 |
| pair_reliability | 0.000000 | 0.000 | 0.000 | 0.000 |
| trimmed_unlabeled_energy | 0.016313 | 1.000 | 1.000 | 1.000 |
| view_consistency | 0.000000 | 0.000 | 0.000 | 0.000 |

### A1_hinge_rank_barrier
| component | total grad norm | encoder update freq | reader update freq | scalar active freq |
|---|---:|---:|---:|---:|
| collapse_or_rank_barrier | 0.000316 | 1.000 | 1.000 | 1.000 |
| known_normal_energy | 0.153917 | 1.000 | 1.000 | 1.000 |
| pair_reliability | 0.000000 | 0.000 | 0.000 | 0.000 |
| trimmed_unlabeled_energy | 0.001456 | 0.046 | 0.046 | 0.046 |
| view_consistency | 0.000000 | 0.000 | 0.000 | 0.000 |

### A2_view_consistency
| component | total grad norm | encoder update freq | reader update freq | scalar active freq |
|---|---:|---:|---:|---:|
| collapse_or_rank_barrier | 0.000316 | 1.000 | 1.000 | 1.000 |
| known_normal_energy | 0.153917 | 1.000 | 1.000 | 1.000 |
| pair_reliability | 0.000000 | 0.000 | 0.000 | 0.000 |
| trimmed_unlabeled_energy | 0.001456 | 0.046 | 0.046 | 0.046 |
| view_consistency | 0.000000 | 0.000 | 0.000 | 0.000 |

### A3_pair_reliability
| component | total grad norm | encoder update freq | reader update freq | scalar active freq |
|---|---:|---:|---:|---:|
| collapse_or_rank_barrier | 0.000338 | 1.000 | 1.000 | 1.000 |
| known_normal_energy | 0.143146 | 1.000 | 1.000 | 1.000 |
| pair_reliability | 0.000000 | 0.000 | 0.000 | 0.000 |
| trimmed_unlabeled_energy | 0.000511 | 0.031 | 0.031 | 0.031 |
| view_consistency | 0.000000 | 0.000 | 0.000 | 0.000 |

## Score decomposition Spearman summary

### A0_rocc_mc
| score component | Spearman vs mat_mean | Spearman vs RIFT score | diagnostic AUC |
|---|---:|---:|---:|
| known_normal_energy | 0.1082 ± 0.0686 | 1.0000 ± 0.0000 | 0.7223 ± 0.0106 |
| pair_reliable_response_energy | 0.9995 ± 0.0000 | 0.1079 ± 0.0688 | 0.8108 ± 0.0067 |
| pair_unreliable_response_energy | -0.2123 ± 0.0178 | 0.2348 ± 0.0448 | 0.6180 ± 0.0088 |
| trimmed_unlabeled_hinge_energy | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| view_cons_penalty | 0.0392 ± 0.0385 | 0.0083 ± 0.0093 | 0.5172 ± 0.0120 |

### A1_hinge_rank_barrier
| score component | Spearman vs mat_mean | Spearman vs RIFT score | diagnostic AUC |
|---|---:|---:|---:|
| known_normal_energy | -0.1667 ± 0.0926 | 1.0000 ± 0.0000 | 0.6554 ± 0.0400 |
| pair_reliable_response_energy | 0.9995 ± 0.0000 | -0.1696 ± 0.0928 | 0.8108 ± 0.0067 |
| pair_unreliable_response_energy | -0.2123 ± 0.0178 | 0.4049 ± 0.0421 | 0.6180 ± 0.0088 |
| trimmed_unlabeled_hinge_energy | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| view_cons_penalty | 0.0028 ± 0.1522 | 0.0131 ± 0.0492 | 0.5085 ± 0.0459 |

### A2_view_consistency
| score component | Spearman vs mat_mean | Spearman vs RIFT score | diagnostic AUC |
|---|---:|---:|---:|
| known_normal_energy | -0.1667 ± 0.0926 | 1.0000 ± 0.0000 | 0.6554 ± 0.0400 |
| pair_reliable_response_energy | 0.9995 ± 0.0000 | -0.1696 ± 0.0928 | 0.8108 ± 0.0067 |
| pair_unreliable_response_energy | -0.2123 ± 0.0178 | 0.4049 ± 0.0421 | 0.6180 ± 0.0088 |
| trimmed_unlabeled_hinge_energy | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| view_cons_penalty | 0.0028 ± 0.1522 | 0.0131 ± 0.0492 | 0.5085 ± 0.0459 |

### A3_pair_reliability
| score component | Spearman vs mat_mean | Spearman vs RIFT score | diagnostic AUC |
|---|---:|---:|---:|
| known_normal_energy | -0.2001 ± 0.0532 | 1.0000 ± 0.0000 | 0.6656 ± 0.0328 |
| pair_reliable_response_energy | 0.9995 ± 0.0000 | -0.2019 ± 0.0542 | 0.8108 ± 0.0067 |
| pair_unreliable_response_energy | -0.2123 ± 0.0178 | 0.4444 ± 0.0577 | 0.6180 ± 0.0088 |
| trimmed_unlabeled_hinge_energy | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| view_cons_penalty | 0.0189 ± 0.1528 | 0.0110 ± 0.0545 | 0.5121 ± 0.0447 |

## Mechanistic diagnosis
- A1/A2/A3 are judged by whether their named loss terms produce non-zero gradients on encoder/reader and whether the resulting RIFT score becomes a better or differently informative ordering than `mat_mean`.
- If pair_reliability shows zero direct gradient but changes scores, it should be interpreted as fixed input-side reliability weighting, not as a learned objective pressure.
- Promotion requires more than scalar loss activity: the update must hit trainable reader/encoder parameters and improve or complement anomaly ranking under label-free selection.